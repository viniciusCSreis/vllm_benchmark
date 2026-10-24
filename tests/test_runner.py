"""Benchmark timing works independently of optional response artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

from test_cli import config

from vllm_bench.contracts import Trial
from vllm_bench.runner import run_only


class TimingTests(unittest.TestCase):
    def test_successful_script_latency_defaults_to_runtime(self):
        for payload in [None, {"results": [1]}, {"latency_sec": 0.01}]:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                directory = root / "trial"
                directory.mkdir()
                if payload is not None:
                    (directory / "response.json").write_text(json.dumps(payload))
                (root / "benchmark.py").write_text("print('Benchmark completed')\n")
                result = run_only(
                    config(), root,
                    Trial(benchmark="first", scenario="baseline", repeat=1),
                    directory, directory / "client.json", [],
                )
                self.assertEqual(result.status, "ok", result.error)
                self.assertGreater(result.runtime_sec, 0)
                expected = (
                    payload["latency_sec"]
                    if payload and "latency_sec" in payload else result.runtime_sec
                )
                self.assertEqual(result.latency_sec, expected)
                self.assertEqual(
                    result.output_rows, 1 if payload and "results" in payload else None
                )

    def test_failed_script_retains_runtime_without_success_latency(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "benchmark.py").write_text("raise RuntimeError('Failed')\n")
            result = run_only(
                config(), root,
                Trial(benchmark="first", scenario="baseline", repeat=1),
                root / "trial", root / "client.json", [],
            )
            self.assertEqual(result.status, "failed")
            self.assertGreater(result.runtime_sec, 0)
            self.assertIsNone(result.latency_sec)
