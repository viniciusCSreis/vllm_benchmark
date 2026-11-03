"""Check metric weighting and offline report generation through the CLI."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_bench.cli import main
from vllm_bench.contracts import Trial, TrialResult
from vllm_bench.reports.data import ReportData
from vllm_bench.reports.v2 import trial_rows


class ReportTests(unittest.TestCase):
    def test_rates_pool_engine_counts_and_preserve_missing_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = Trial(benchmark="example", scenario="distributed", repeat=1)
            directory = root / trial.directory
            directory.mkdir(parents=True)
            engines = [
                {
                    "engine": "leader",
                    "cache_queries": 100,
                    "cache_hits": 90,
                    "prompt_tokens": 100,
                    "requests": 1000,
                    "generation_tokens": 0,
                },
                {
                    "engine": "worker",
                    "cache_queries": 900,
                    "cache_hits": 450,
                    "prompt_tokens": 900,
                    "requests": 1,
                    "generation_tokens": None,
                },
            ]
            record = TrialResult(
                **trial.model_dump(), status="ok", latency_sec=2, engines=engines
            ).model_dump()
            before, after = {}, {}
            metric = "vllm:time_to_first_token_seconds"
            for name, total, count in [("leader", 2, 1), ("worker", 90, 9)]:
                before[name] = {"values": {metric + "_sum": 10, metric + "_count": 2}}
                after[name] = {
                    "values": {
                        metric + "_sum": 10 + total,
                        metric + "_count": 2 + count,
                    }
                }
            (directory / "metrics_before.json").write_text(json.dumps(before))
            (directory / "metrics_after.json").write_text(json.dumps(after))
            data = ReportData(root, root / "reports", [record])
            row = trial_rows(data)[0]
            self.assertEqual(row["cache_hit_pct"], 54)
            self.assertEqual(row["uncached_query_tokens"], 460)
            self.assertEqual(row["ttft_mean_sec"], 9.2)
            self.assertEqual(row["prompt_tokens"], 1000)
            self.assertIsNone(row["generation_tokens"])
            after["worker"]["values"][metric + "_count"] = 1
            (directory / "metrics_after.json").write_text(json.dumps(after))
            self.assertIsNone(trial_rows(data)[0]["ttft_mean_sec"])
            (directory / "metrics_after.json").unlink()
            self.assertIsNone(trial_rows(data)[0]["ttft_mean_sec"])

    def test_report_command_generates_both_formats_for_multiple_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trials = [
                Trial(benchmark="first", scenario="baseline", repeat=1),
                Trial(benchmark="first", scenario="baseline", repeat=2),
                Trial(benchmark="first", scenario="not_completed", repeat=1),
                Trial(benchmark="second", scenario="baseline", repeat=1),
            ]
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "trials": [trial.model_dump() for trial in trials],
                    }
                )
            )
            for trial in [trials[0], trials[1], trials[3]]:
                directory = root / trial.directory
                directory.mkdir(parents=True)
                TrialResult(
                    **trial.model_dump(),
                    status="ok",
                    latency_sec=trial.repeat,
                    runtime_sec=trial.repeat,
                ).save(directory)
            with patch("sys.argv", ["vllm_bench", "report", str(root)]):
                self.assertEqual(main(), 0)
            self.assertTrue((root / "reports/summary.json").is_file())
            output = root / "reports_v2"
            self.assertTrue((output / "report.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertTrue((output / "summary.csv").is_file())
            rows = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                [row["status"] for row in rows], ["ok", "ok", "failed", "ok"]
            )
            self.assertIsNone(rows[0]["ttft_mean_sec"])
            for benchmark in ["first", "second"]:
                self.assertEqual(
                    {
                        path.stem
                        for path in (output / "images" / benchmark).glob("*.png")
                    },
                    {
                        "query_latency_sec",
                        "cache_hit_pct",
                        "ttft_mean_sec",
                        "prompt_tokens",
                        "uncached_query_tokens",
                        "requests",
                        "generation_tokens",
                    },
                )
