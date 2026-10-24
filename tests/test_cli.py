"""CPU checks for the public script contract and Compose lifecycle boundaries."""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from vllm_bench import Service, bench_util, enviroment
from vllm_bench.artifacts import prepare_run, project_files, trials
from vllm_bench.cli import main
from vllm_bench.config import Config, load_config, select_benchmarks
from vllm_bench.contracts import Trial, TrialResult
from vllm_bench.metrics import deltas, parse_metrics
from vllm_bench.providers.compose import ComposeProvider
from vllm_bench.providers.context import script_context, service_instances
from vllm_bench.reports.data import load_data


def config(**changes):
    return Config.model_validate(
        {
            "name": "test",
            "base_image": "test-image",
            "global_args": {"model": "test-model"},
            "services": {
                "worker": {
                    "role": "worker",
                    "startup_script": "worker.py",
                    "replicas": 2,
                },
                "leader": {
                    "role": "leader",
                    "startup_script": "leader.py",
                    "metrics_path": "/metrics",
                },
            },
            "scenarios": {
                "baseline": {
                    "args": {"coordinator": "single_worker"},
                    "services": ["leader"],
                },
                "distributed": {
                    "args": {"coordinator": "round_robin"},
                    "services": ["leader", "worker"],
                },
            },
            "benchmarks": {
                "first": {
                    "script": "benchmark.py",
                    "scenarios": ["baseline", "distributed"],
                },
            },
            **changes,
        }
    )


class CliTests(unittest.TestCase):
    def tearDown(self):
        bench_util._context.cache_clear()

    def test_config_rejects_unknown_services_and_duplicate_scenarios(self):
        for scenarios in [
            {"test": {"services": ["missing"]}},
            {"empty": {"services": []}},
            {"missing_services": {}},
            {"repeated": {"services": ["leader", "leader"]}},
        ]:
            with self.subTest(scenarios=scenarios), self.assertRaises(ValidationError):
                config(scenarios=scenarios)
        for names in [["baseline", "baseline"], ["missing"]]:
            with self.subTest(names=names), self.assertRaises(ValidationError):
                config(
                    benchmarks={"first": {"script": "benchmark.py", "scenarios": names}}
                )
        with self.assertRaises(ValidationError):
            config(
                benchmarks={
                    "first": {
                        "script": "benchmark.py",
                        "scenarios": ["baseline"],
                        "services": ["missing"],
                    }
                }
            )
        with self.assertRaises(ValidationError):
            config(
                services={"bad": {"role": "leader", "startup_script": "../secret.py"}}
            )

    def test_named_config_inherits_base_and_merges_nested_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = config().model_dump(exclude_none=True)
            (root / "base.yaml").write_text(yaml.safe_dump(base))
            (root / "first.yaml").write_text(
                yaml.safe_dump(
                    {
                        "global_args": {"max_model_len": 2048},
                    }
                )
            )
            selected = load_config(root / "first")
            self.assertEqual(
                selected.global_args, {"model": "test-model", "max_model_len": 2048}
            )
            self.assertEqual(
                selected.get_scenario("first", "baseline").services, ["leader"]
            )

    def test_current_dir_uses_project_configs_scripts_results_and_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            folder = root / ".vllm_bench/configs"
            folder.mkdir(parents=True)
            data = config().model_dump(exclude_none=True)
            (folder / "base.yaml").write_text(yaml.safe_dump(data))
            (folder / "first.yaml").write_text(
                yaml.safe_dump({"global_args": {"max_model_len": 2048}})
            )
            for script in ["leader.py", "benchmark.py"]:
                (root / script).touch()
            with (
                patch(
                    "sys.argv",
                    [
                        "vllm_bench",
                        "--current-dir",
                        str(root),
                        "up",
                        "--config",
                        "first",
                        "--benchmarks",
                        "first",
                        "--scenarios",
                        "baseline",
                    ],
                ),
                patch("vllm_bench.cli.get_provider") as provider,
                patch("vllm_bench.cli.show_environment", return_value=0),
            ):
                provider.return_value.start.return_value = (None, [])
                self.assertEqual(main(), 0)
                self.assertEqual(provider.call_args.args[1], root)
                directory = provider.return_value.start.call_args.args[1]
                self.assertTrue(directory.is_relative_to(root / "results"))
            (root / ".vllm_bench/state.json").write_text("{}")
            archived = {name for _, name in project_files(root)}
            self.assertIn(".vllm_bench/configs/base.yaml", archived)
            self.assertNotIn(".vllm_bench/state.json", archived)
            with (
                patch("sys.argv", ["vllm_bench", "down", "--current-dir", str(root)]),
                patch("vllm_bench.cli.stop_saved") as stop,
            ):
                self.assertEqual(main(), 0)
                stop.assert_called_once_with(root / ".vllm_bench/state.json")

    def test_select_multiple_benchmarks_and_scenarios(self):
        data = config().model_dump()
        data["scenarios"]["other"] = {"services": ["worker"]}
        data["benchmarks"]["second"] = {
            "script": "other.py",
            "scenarios": ["baseline", "other"],
            "services": ["worker", "leader"],
        }
        settings = Config.model_validate(data)
        selected = select_benchmarks(settings, ["*"], ["baseline"])
        self.assertEqual(
            [(t.benchmark, t.scenario) for t in trials(selected)],
            [("first", "baseline"), ("second", "baseline")],
        )
        second = Trial(benchmark="second", scenario="baseline", repeat=1)
        instances = service_instances(selected, second)
        self.assertEqual(list(instances), ["leader-1", "worker-1", "worker-2"])
        self.assertEqual(settings.scenarios["baseline"].services, ["leader"])
        context = script_context(selected, second, Path("/project"), instances)
        self.assertEqual(context["scenario_args"], {"coordinator": "single_worker"})
        self.assertEqual(len(trials(select_benchmarks(settings, ["*"], ["*"]))), 4)
        self.assertEqual(
            [
                t.scenario
                for t in trials(select_benchmarks(settings, ["second"], ["*"]))
            ],
            ["baseline", "other"],
        )
        with self.assertRaisesRegex(ValueError, "benchmark 'second'.*distributed"):
            select_benchmarks(settings, ["*"], ["distributed"])
        with self.assertRaisesRegex(ValueError, "Unknown benchmarks"):
            select_benchmarks(settings, ["unknown"], ["*"])
        with self.assertRaisesRegex(ValueError, r"Use '\*' alone"):
            select_benchmarks(settings, ["*", "first"], ["*"])

    def test_invalid_selection_does_not_stop_active_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / ".vllm_bench/configs"
            folder.mkdir(parents=True)
            data = config().model_dump()
            data["benchmarks"]["second"] = {
                "script": "second.py",
                "scenarios": ["baseline"],
            }
            (folder / "base.yaml").write_text(yaml.safe_dump(data))
            (folder / "local_cpu.yaml").write_text("{}\n")
            state = root / ".vllm_bench/state.json"
            state.write_text("active")
            for command, scenarios in [
                ("run_scenarios", "missing"),
                ("run_scenarios", "distributed"),
                ("up", "*"),
            ]:
                with (
                    patch(
                        "sys.argv",
                        [
                            "vllm_bench",
                            command,
                            "--current-dir",
                            str(root),
                            "--benchmark",
                            "*",
                            "--scenario",
                            scenarios,
                        ],
                    ),
                    patch("vllm_bench.cli.get_provider") as provider,
                    patch("vllm_bench.cli.stop_saved") as stop,
                ):
                    self.assertEqual(main(), 1)
                    stop.assert_not_called()
                    provider.assert_not_called()
                    self.assertEqual(state.read_text(), "active")
                    self.assertFalse((root / "results").exists())

    def test_up_does_not_attach_to_a_different_environment_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / ".vllm_bench/configs"
            folder.mkdir(parents=True)
            (folder / "base.yaml").write_text(yaml.safe_dump(config().model_dump()))
            (folder / "local_cpu.yaml").write_text("{}\n")
            for script in ["leader.py", "benchmark.py"]:
                (root / script).touch()
            directory = root / "active"
            (directory / ".provider").mkdir(parents=True)
            (directory / ".provider/client.json").write_text(
                json.dumps(
                    {
                        "benchmark": "first",
                        "scenario": "baseline",
                        "global_args": {"model": "test-model"},
                        "scenario_args": {"coordinator": "single_worker"},
                    }
                )
            )
            state = root / ".vllm_bench/state.json"
            state.write_text(
                json.dumps(
                    {
                        "provider": "local",
                        "image": "arm-image",
                        "directory": str(directory),
                    }
                )
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "vllm_bench",
                        "up",
                        "--current-dir",
                        str(root),
                        "--benchmark",
                        "first",
                        "--scenario",
                        "baseline",
                    ],
                ),
                patch("vllm_bench.cli.get_provider") as provider,
                patch("vllm_bench.cli.show_environment") as show,
            ):
                self.assertEqual(main(), 1)
                provider.assert_not_called()
                show.assert_not_called()
                self.assertTrue(state.exists())

    def test_snapshot_numbering_exclusions_and_skip_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.py").write_text("# current source")
            (root / ".env").write_text("SECRET=secret")
            first = prepare_run(config(), root)
            with zipfile.ZipFile(first / "project.zip") as archive:
                self.assertEqual(archive.namelist(), ["source.py"])
            second = prepare_run(config(skip_volume_mount=True), root)
            self.assertEqual(second.name, "run002")
            self.assertFalse((second / "project.zip").exists())

    def test_scenario_selects_services_and_expands_replicas(self):
        for scenario, expected in [("baseline", 1), ("distributed", 3)]:
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                directory = root / "trial"
                directory.mkdir()
                provider = ComposeProvider(config(), root)
                with patch(
                    "vllm_bench.providers.compose.compose",
                    return_value="127.0.0.1:12345\n",
                ):
                    context, endpoints = provider.start(
                        Trial(benchmark="first", scenario=scenario, repeat=2), directory
                    )
                compose = yaml.safe_load(
                    (directory / ".provider/compose.yaml").read_text()
                )
                self.assertEqual(len(compose["services"]), expected)
                self.assertEqual(compose["services"]["leader-1"]["image"], "test-image")
                client = json.loads(context.read_text())
                self.assertEqual(len(client["services"]), expected)
                self.assertEqual(client["services"][0]["url"], "http://127.0.0.1:12345")
                self.assertEqual(
                    client["internal_services"][0]["url"], "http://leader-1:8000"
                )
                self.assertEqual(client["repeat"], 2)
                self.assertEqual(len(endpoints), expected)

    def test_skip_mount_keeps_context_and_output_but_excludes_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = ComposeProvider(
                config(skip_volume_mount=True, image_workdir="/app"), root
            )
            directory = root / "trial"
            directory.mkdir()
            with patch(
                "vllm_bench.providers.compose.compose", return_value="127.0.0.1:12345\n"
            ):
                provider.start(
                    Trial(benchmark="first", scenario="baseline", repeat=1), directory
                )
            definition = yaml.safe_load(
                (directory / ".provider/compose.yaml").read_text()
            )
            service = definition["services"]["leader-1"]
            self.assertEqual(service["working_dir"], "/app")
            self.assertEqual(
                service["entrypoint"], ["python3", "-m", "vllm_bench.service"]
            )
            self.assertEqual(
                service["environment"]["PYTHONPATH"], "/opt/vllm-bench-lib"
            )
            mounts = [item for item in service["volumes"] if isinstance(item, dict)]
            self.assertTrue(mounts[0]["read_only"])
            self.assertTrue((Path(mounts[0]["source"]) / "service.py").is_file())
            self.assertEqual(
                [item["target"] for item in mounts],
                ["/opt/vllm-bench-lib/vllm_bench", "/run/vllm-bench/trial"],
            )

    def test_script_helpers_distinguish_client_and_internal_addresses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "trial"
            directory.mkdir()
            provider = ComposeProvider(config(), root)
            with patch(
                "vllm_bench.providers.compose.compose", return_value="127.0.0.1:12345\n"
            ):
                context, _ = provider.start(
                    Trial(benchmark="first", scenario="baseline", repeat=2), directory
                )
            with patch.dict(os.environ, {"VLLM_BENCH_CONTEXT": str(context)}):
                self.assertEqual(enviroment.benchmark, "first")
                self.assertEqual(enviroment.scenario, "baseline")
                self.assertEqual(enviroment.repeat, 2)
                self.assertIsInstance(enviroment.project_dir, Path)
                self.assertEqual(enviroment.output_dir, directory)
                self.assertIsNone(enviroment.service)
                with self.assertRaisesRegex(ValueError, "no current service"):
                    enviroment.get_service()
                self.assertIsInstance(enviroment.leader, Service)
                self.assertEqual(enviroment.leader.port, 12345)
                self.assertEqual(enviroment.leader.url, "http://127.0.0.1:12345")
                self.assertEqual(
                    enviroment.get_service("leader", internal=True).url,
                    "http://leader-1:8000",
                )
                self.assertEqual(enviroment.workers, ())
                self.assertEqual(enviroment.get_arg("missing", 0.5), 0.5)
                self.assertEqual(
                    enviroment.get_arg("model", scope="global"), "test-model"
                )
                self.assertEqual(
                    enviroment.get_arg("coordinator", scope="scenario"),
                    "single_worker",
                )
                self.assertEqual(bench_util.current_run, 2)
                self.assertEqual(bench_util.get_global_arg("model"), "test-model")
                self.assertEqual(
                    bench_util.get_scenario_arg("coordinator"), "single_worker"
                )
                self.assertEqual(bench_util.get_worker_urls(), [])
                self.assertEqual(bench_util.get_leader_url(), "http://127.0.0.1:12345")
                self.assertEqual(
                    bench_util.get_service_url("leader", internal=True),
                    "http://leader-1:8000",
                )
                self.assertEqual(
                    bench_util.get_service_path(directory / "inputs/file.csv"),
                    "/run/vllm-bench/trial/inputs/file.csv",
                )

    def test_environment_returns_typed_replicas_with_isolated_arguments(self):
        settings = config()
        settings.services["worker"].service_args = {"model": "worker-model"}
        trial = Trial(benchmark="first", scenario="distributed", repeat=1)
        instances = service_instances(settings, trial)
        context = script_context(settings, trial, Path("/project"), instances)
        context.update(
            service=instances["worker-2"], output_dir="/trial/services/worker-2"
        )
        with patch("vllm_bench.enviroment._context", return_value=context):
            service = enviroment.get_service()
            self.assertIsInstance(service, Service)
            self.assertEqual(service.name, "worker-2")
            self.assertEqual(service.group, "worker")
            self.assertEqual(service.role, "worker")
            self.assertEqual(service.replica, 2)
            self.assertEqual(service.args["model"], "worker-model")
            self.assertEqual(len(enviroment.workers), 2)
            with self.assertRaisesRegex(ValueError, "found 2"):
                enviroment.get_service("worker")
            self.assertEqual(enviroment.get_service("worker-2"), service)
            self.assertEqual(enviroment.leader.args["model"], "test-model")
            service.args["model"] = "changed"
            self.assertEqual(enviroment.get_service().args["model"], "worker-model")
            with self.assertRaisesRegex(ValueError, "found 0"):
                enviroment.get_service("missing")

    def test_service_args_override_globals_without_affecting_other_services(self):
        data = config().model_dump()
        data["global_args"].update(max_model_len=8192, max_num_seqs=2)
        data["services"]["leader"]["service_args"] = {
            "model": "helper-model",
            "max_num_seqs": 1,
        }
        settings = Config.model_validate(data)
        trial = Trial(benchmark="first", scenario="distributed", repeat=1)
        for shared_network in [False, True]:
            instances = service_instances(
                settings, trial, shared_network=shared_network
            )
            context = script_context(settings, trial, Path("/project"), instances)
            with patch.object(bench_util, "_context", return_value=context):
                leader = bench_util.get_service_args("leader")
                self.assertEqual(leader["model"], "helper-model")
                self.assertEqual(leader["max_num_seqs"], 1)
                self.assertEqual(leader["max_model_len"], 8192)
                self.assertEqual(
                    bench_util.get_service_args("worker"), settings.global_args
                )
                context["service"] = instances["leader-1"]
                self.assertEqual(bench_util.get_service_args(), leader)
                leader["max_model_len"] = 1
                self.assertEqual(bench_util.get_global_arg("max_model_len"), 8192)
                self.assertEqual(bench_util.get_global_arg("model"), "test-model")

    def test_benchmark_global_overrides_are_isolated_and_service_args_apply_last(self):
        settings = config(
            global_args={"model": "test-model", "gpu_memory_utilization": 0.9}
        )
        settings.benchmarks["second"] = settings.benchmarks["first"].model_copy(
            deep=True
        )
        settings.benchmarks["first"].overwrite_global_args = {
            "gpu_memory_utilization": 0.8,
            "icl_gpu": True,
        }
        settings.services["worker"].service_args = {"gpu_memory_utilization": 0.5}
        for shared_network in [False, True]:
            for benchmark, expected in [("first", 0.8), ("second", 0.9)]:
                trial = Trial(benchmark=benchmark, scenario="distributed", repeat=1)
                instances = service_instances(
                    settings, trial, shared_network=shared_network
                )
                context = script_context(settings, trial, Path("/project"), instances)
                context.update(
                    service=instances["leader-1"], output_dir="/trial/leader"
                )
                with patch("vllm_bench.enviroment._context", return_value=context):
                    self.assertEqual(
                        enviroment.global_args["gpu_memory_utilization"], expected
                    )
                    self.assertEqual(
                        enviroment.leader.args["gpu_memory_utilization"], expected
                    )
                    self.assertEqual(
                        enviroment.get_service("worker-1").args[
                            "gpu_memory_utilization"
                        ],
                        0.5,
                    )
                    self.assertEqual(
                        enviroment.global_args.get("icl_gpu", False),
                        benchmark == "first",
                    )
        self.assertEqual(settings.global_args["gpu_memory_utilization"], 0.9)
        self.assertNotIn("icl_gpu", settings.global_args)

    def test_missing_metrics_and_counter_resets_are_not_zero(self):
        self.assertEqual(
            parse_metrics(
                'vllm:prompt_tokens_total{model="a"} 3\n'
                'vllm:prompt_tokens_total{model="b"} 4\n'
            ),
            {"vllm:prompt_tokens_total": 7},
        )
        rows = deltas(
            {"leader": {"values": {"vllm:prompt_tokens_total": 10}}},
            {"leader": {"values": {"vllm:prompt_tokens_total": 2}}},
            1,
        )
        self.assertIsNone(rows[0]["prompt_tokens"])
        self.assertIsNone(rows[0]["requests"])

    def test_reports_keep_missing_trials_and_distinct_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = prepare_run(config(), root)
            trial = Trial(benchmark="first", scenario="baseline", repeat=1)
            directory = output / trial.directory
            directory.mkdir(parents=True)
            TrialResult(**trial.model_dump(), status="ok", runtime_sec=2).save(
                directory
            )
            records = load_data(output).records
            self.assertEqual([record["status"] for record in records], ["ok", "failed"])
            self.assertEqual(load_data(output).directory(records[0]), directory)
