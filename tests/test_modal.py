"""Provider boundary checks without provisioning paid cloud resources."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml
from test_cli import config

from vllm_bench import bench_util
from vllm_bench.config import load_config
from vllm_bench.contracts import Trial
from vllm_bench.providers import stop_saved
from vllm_bench.providers.modal import ModalProvider


class ModalTests(unittest.TestCase):
    def tearDown(self):
        bench_util._context.cache_clear()

    def test_config_extends_environment_and_preserves_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = config().model_dump(exclude_none=True)
            (root / "base.yaml").write_text(yaml.safe_dump(base))
            (root / "local_cpu.yaml").write_text(
                yaml.safe_dump({"global_args": {"model": "test-model"}})
            )
            (root / "modal.yaml").write_text(
                yaml.safe_dump(
                    {
                        "extends": "local_cpu",
                        "provider": "modal",
                        "modal": {"gpu": "T4:1"},
                    }
                )
            )
            loaded = load_config(root / "modal")
            self.assertEqual(loaded.provider, "modal")
            self.assertEqual(loaded.modal.gpu, "T4:1")
            self.assertEqual(loaded.global_args, base["global_args"])
            self.assertEqual(len(loaded.benchmarks["first"].scenarios), 2)
            (root / "local_cpu.yaml").write_text("extends: modal\n")
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_config(root / "modal")

    def test_unavailable_sandbox_saves_logs_and_removes_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "provider": "modal",
                        "sandbox_id": "sb-test",
                        "directory": str(root),
                    }
                )
            )
            sandbox = MagicMock()
            sandbox.poll.side_effect = RuntimeError("The Sandbox is unavailable")
            sandbox.stdout.read.return_value = "service startup\n"
            sandbox.stderr.read.return_value = "Runner exceeded its memory request\n"
            sandbox.terminate.side_effect = RuntimeError("Already stopped")
            sdk = MagicMock()
            sdk.Sandbox.from_id.return_value = sandbox
            with patch("vllm_bench.providers.modal.modal_sdk", return_value=sdk):
                with self.assertRaises(RuntimeError):
                    stop_saved(state_path)
            self.assertIn("memory request", (root / "modal.log").read_text())
            self.assertFalse(state_path.exists())

    def test_modal_lifecycle_and_generated_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "results/run001/data/first/distributed/run_1"
            directory.mkdir(parents=True)
            with zipfile.ZipFile(root / "results/run001/project.zip", "w") as bundle:
                bundle.writestr("source.py", "# project snapshot")
            sandbox = MagicMock()
            sandbox.object_id = "sb-test"
            sandbox.poll.return_value = None
            sandbox.tunnels.return_value = {
                port: SimpleNamespace(url=f"https://service-{port}.test")
                for port in range(8000, 8003)
            }
            command = sandbox.exec.return_value
            command.stdout.read.return_value = ""
            command.stderr.read.return_value = ""
            command.returncode = 0
            sdk = MagicMock()
            sdk.Sandbox.create.return_value = sandbox
            sdk.Sandbox.from_id.return_value = sandbox
            with patch("vllm_bench.providers.modal.modal_sdk", return_value=sdk):
                provider = ModalProvider(
                    config(provider="modal", modal={"gpu": "T4:1"}), root
                )
                context, endpoints = provider.start(
                    Trial(benchmark="first", scenario="distributed", repeat=1),
                    directory,
                )
                creation = sdk.Sandbox.create.call_args.kwargs
                self.assertEqual(creation["gpu"], "T4:1")
                self.assertEqual(list(creation["volumes"]), ["/mnt/vllm-bench-cache"])
                self.assertEqual(
                    creation["env"]["HF_HOME"], "/mnt/vllm-bench-cache/huggingface"
                )
                self.assertEqual(
                    creation["env"]["VLLM_CACHE_ROOT"], "/mnt/vllm-bench-cache/vllm"
                )
                self.assertEqual(
                    creation["env"]["XDG_CACHE_HOME"], "/mnt/vllm-bench-cache"
                )
                self.assertEqual(creation["encrypted_ports"], [8000, 8001, 8002])
                self.assertEqual(len(endpoints), 3)
                data = json.loads(context.read_text())
                self.assertEqual(
                    data["internal_services"][1]["url"], "http://127.0.0.1:8001"
                )
                launches = [
                    json.loads(call.args[0])
                    for call in sandbox.filesystem.write_text.call_args_list
                    if call.args[1].endswith("launch.json")
                ][0]
                self.assertEqual(len(launches), 3)
                self.assertNotIn("HOME", launches[0]["environment"])
                dataset = directory / "dataset.csv"
                dataset.write_text("id\n1\n")
                with patch.dict("os.environ", {"VLLM_BENCH_CONTEXT": str(context)}):
                    self.assertEqual(
                        bench_util.get_service_path(dataset),
                        "/run/vllm-bench/trial/dataset.csv",
                    )
                sandbox.filesystem.copy_from_local.assert_any_call(
                    dataset, "/run/vllm-bench/trial/dataset.csv"
                )
                sandbox.filesystem.read_text.return_value = json.dumps(
                    {"leader-1": None, "worker-1": 1}
                )
                with self.assertRaisesRegex(RuntimeError, "worker-1"):
                    provider.check_running()

                def download(remote, local):
                    with zipfile.ZipFile(local, "w") as bundle:
                        bundle.writestr(
                            "services/leader-1/container.log", "service output"
                        )

                sandbox.filesystem.copy_to_local.side_effect = download
                stop_saved(root / ".vllm_bench/state.json")
                sandbox.terminate.assert_called_once()
                self.assertTrue(
                    (directory / "services/leader-1/container.log").is_file()
                )
                self.assertFalse((root / ".vllm_bench/state.json").exists())


if __name__ == "__main__":
    unittest.main()
