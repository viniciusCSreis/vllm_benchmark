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

from vllm_bench import bench_util
from vllm_bench.artifacts import prepare_run
from vllm_bench.config import Config
from vllm_bench.contracts import Trial, TrialResult
from vllm_bench.metrics import deltas, parse_metrics
from vllm_bench.providers.compose import ComposeProvider
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
            "benchmarks": {
                "first": {"script": "benchmark.py", "services": ["leader", "worker"]}
            },
            "benchmark_scenarios": [
                {
                    "name": "baseline",
                    "args": {"coordinator": "single_worker"},
                    "disabled_services": ["worker"],
                },
                {"name": "distributed", "args": {"coordinator": "round_robin"}},
            ],
            **changes,
        }
    )


class CliTests(unittest.TestCase):
    def tearDown(self):
        bench_util._context.cache_clear()

    def test_config_rejects_unknown_services_and_duplicate_scenarios(self):
        with self.assertRaises(ValidationError):
            config(benchmarks={"test": {"script": "test.py", "services": ["missing"]}})
        with self.assertRaises(ValidationError):
            config(benchmark_scenarios=[{"name": "same"}, {"name": "same"}])
        with self.assertRaises(ValidationError):
            config(
                services={"bad": {"role": "leader", "startup_script": "../secret.py"}}
            )

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

    def test_topology_disables_workers_and_expands_replicas(self):
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
            mounts = [item for item in service["volumes"] if isinstance(item, dict)]
            self.assertEqual(
                [item["target"] for item in mounts], ["/run/vllm-bench/trial"]
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
