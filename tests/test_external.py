"""Service ownership and execution checks for external and managed trials."""

import json
import shlex
import tempfile
import threading
import unittest
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from test_cli import config

from vllm_bench.cli import main, run_only_command
from vllm_bench.contracts import Trial, TrialResult
from vllm_bench.external import service_endpoints
from vllm_bench.reports.data import load_data


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(
            f"vllm:request_success_total {self.server.requests}\n".encode()
        )

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        path = Path(payload["path"])
        self.server.last_path = str(path)
        if project := getattr(self.server, "project_dir", None):
            path = project / path.relative_to("/workspace/project")
        if path.read_text() != "shared input":
            self.send_error(400)
            return
        self.server.requests += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"latency_sec": 0.01, "results": [1]}).encode())

    def log_message(self, *args):
        pass


class ExternalTests(unittest.TestCase):
    def test_external_success_and_failure_leave_services_and_active_state_alone(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.requests = 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for fail in [False, True]:
                with (
                    self.subTest(fail=fail),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    root = Path(temporary)
                    folder = root / ".vllm_bench/configs"
                    folder.mkdir(parents=True)
                    data = config(runs=3).model_dump()
                    data.pop("base_image")
                    for service in data["services"].values():
                        service.pop("startup_script")
                    data["benchmarks"]["first"]["script_args"] = {"fail": fail}
                    (folder / "base.yaml").write_text(yaml.safe_dump(data))
                    state = root / ".vllm_bench/state.json"
                    state.write_text("existing environment")
                    (root / "benchmark.py").write_text(
                        (Path(__file__).parent / "fixtures/benchmark.py").read_text()
                    )
                    with (
                        patch(
                            "sys.argv",
                            [
                                "vllm_bench",
                                "run_only",
                                "--config",
                                "base",
                                "--current-dir",
                                str(root),
                                "--benchmark",
                                "first",
                                "--scenario",
                                "baseline",
                                "--service-url",
                                f"leader=http://127.0.0.1:{server.server_port}",
                            ],
                        ),
                        patch("vllm_bench.cli.get_provider") as provider,
                        patch("vllm_bench.cli.stop_saved") as stop,
                        patch("vllm_bench.reports.generate_reports") as report,
                    ):
                        self.assertEqual(main(), 1 if fail else 0)
                        provider.assert_not_called()
                        stop.assert_not_called()
                        report.assert_called_once()
                    self.assertEqual(state.read_text(), "existing environment")
                    output = next((root / "results/external-test").iterdir())
                    records = load_data(output).records
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]["repeat"], 1)
                    self.assertEqual(records[0]["status"], "failed" if fail else "ok")
                    trial = output / "data/first/baseline/run_1"
                    context = json.loads((trial / "client.json").read_text())
                    self.assertEqual(context["services"], context["internal_services"])
                    self.assertTrue(context["external_services"])
            self.assertEqual(server.requests, 1)
            self.assertTrue(thread.is_alive())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_up_command_maps_sampled_inputs_into_the_shared_project_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            folder = root / ".vllm_bench/configs"
            folder.mkdir(parents=True)
            settings = config()
            config_path = folder / "base.yaml"
            config_path.write_text(yaml.safe_dump(settings.model_dump()))
            (root / "benchmark.py").write_text(
                (Path(__file__).parent / "fixtures/benchmark.py").read_text()
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.requests = 0
            server.project_dir = root
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                trial = Trial(benchmark="first", scenario="baseline", repeat=1)
                endpoints = service_endpoints(
                    settings, trial, [f"leader=http://127.0.0.1:{server.server_port}"]
                )
                command = shlex.split(
                    run_only_command(settings, config_path, root, trial, endpoints)
                )
                self.assertEqual(
                    command[command.index("--service-project-dir") + 1],
                    "/workspace/project",
                )
                with (
                    patch("sys.argv", command),
                    patch("vllm_bench.reports.generate_reports"),
                ):
                    self.assertEqual(main(), 0)
                self.assertEqual(server.requests, 1)
                self.assertEqual(
                    server.last_path,
                    "/workspace/project/results/external-test/run001/"
                    "data/first/baseline/run_1/inputs.txt",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_service_urls_validate_required_dependencies_and_keep_https_paths(self):
        settings = config()
        settings.benchmarks["first"].services = ["worker"]
        trial = Trial(benchmark="first", scenario="baseline", repeat=1)
        with self.assertRaisesRegex(ValueError, "Missing service URLs: worker"):
            service_endpoints(settings, trial, ["leader=http://localhost:8000"])
        with self.assertRaisesRegex(ValueError, "Repeated service URL"):
            service_endpoints(settings, trial, ["leader=http://a", "leader=http://b"])
        with self.assertRaisesRegex(ValueError, "not required"):
            service_endpoints(settings, trial, ["unknown=http://a"])
        endpoints = service_endpoints(
            settings,
            trial,
            [
                "leader=https://example.com:8443/kalypso/",
                "worker=http://[::1]:8000",
            ],
        )
        self.assertEqual(endpoints[0]["url"], "https://example.com:8443/kalypso")
        self.assertEqual(endpoints[1]["role"], "worker")

    def test_managed_trials_start_execute_and_stop_for_every_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / ".vllm_bench/configs"
            folder.mkdir(parents=True)
            (folder / "base.yaml").write_text(
                yaml.safe_dump(config(runs=2).model_dump())
            )
            for script in ["benchmark.py", "worker.py", "leader.py"]:
                (root / script).touch()
            events = []
            provider = Mock()
            provider.run_session.return_value = nullcontext()

            def start(trial, directory):
                events.append(("up", trial.scenario, trial.repeat))
                return directory / "client.json", []

            def execute(
                settings, project, trial, directory, context, endpoints, **kwargs
            ):
                events.append(("run_only", trial.scenario, trial.repeat))
                # A failed trial must also stop services and allow the next trial.
                return TrialResult(**trial.model_dump(), status="failed")

            provider.start.side_effect = start
            provider.stop.side_effect = lambda: events.append(("down",))
            with (
                patch(
                    "sys.argv",
                    [
                        "vllm_bench",
                        "run_scenarios",
                        "--config",
                        "base",
                        "--current-dir",
                        str(root),
                        "--benchmark",
                        "first",
                        "--scenario",
                        "*",
                    ],
                ),
                patch("vllm_bench.cli.get_provider", return_value=provider),
                patch("vllm_bench.runner.run_only", side_effect=execute),
                patch("vllm_bench.reports.generate_reports"),
            ):
                self.assertEqual(main(), 1)
            self.assertEqual(
                events,
                [
                    event
                    for scenario in ["baseline", "distributed"]
                    for repeat in [1, 2]
                    for event in [
                        ("up", scenario, repeat),
                        ("run_only", scenario, repeat),
                        ("down",),
                    ]
                ],
            )
