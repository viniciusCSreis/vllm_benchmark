"""Opt-in real Compose check with tiny HTTP engines, no models or cloud jobs.

Run: python tests/compose_smoke.py --image kalypso-vllm-benchmark:local
The image must already contain this package and its dependencies.
"""

import argparse
import copy
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="vllm-bench-compose-smoke-"))
    (root / "startup.py").write_text(
        (Path(__file__).parent / "fixtures/startup.py").read_text()
    )
    (root / "benchmark.py").write_text(
        (Path(__file__).parent / "fixtures/benchmark.py").read_text()
    )
    config = {
        "name": "smoke",
        "provider": "local",
        "base_image": args.image,
        "global_args": {"model": "stand-in"},
        "services": {
            name: {
                "role": name,
                "startup_script": "startup.py",
                "metrics_path": "/metrics",
            }
            for name in ["leader", "worker"]
        },
        "benchmarks": {
            name: {"script": "benchmark.py", "services": ["leader", "worker"]}
            for name in ["first", "second"]
        },
        "benchmark_scenarios": [
            {
                "name": "baseline",
                "args": {"coordinator": "single"},
                "disabled_services": ["worker"],
            },
            {"name": "distributed", "args": {"coordinator": "distributed"}},
        ],
        "startup_timeout": 30,
        "benchmark_timeout": 10,
        "sample_interval": 0.1,
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config))

    def cli(*command, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "vllm_bench.cli", *command], cwd=root
        )
        assert result.returncode == expected, result.returncode

    try:
        cli("up", "-d")
        assert (root / ".vllm-bench/state.json").exists()
        cli("down")
        assert not (root / ".vllm-bench/state.json").exists()
        cli("run", "scenarios")
        output = root / "results/local-smoke/run002"
        records = json.loads((output / "reports/summary.json").read_text())
        assert len(records) == 4 and all(
            record["status"] == "ok" for record in records
        ), records
        for record in records:
            directory = (
                output / "data" / record["benchmark"] / record["scenario"] / "run_1"
            )
            assert (directory / "metrics_samples.jsonl").exists()
            assert (directory / "services/leader-1/container.log").exists()
            startup = json.loads(
                (directory / "services/leader-1/startup.json").read_text()
            )
            assert len(startup["workers"]) == (
                0 if record["scenario"] == "baseline" else 1
            )
            assert record["engines"][0]["requests"] == 1, record
        assert (output / "reports/runtime.png").exists()
        assert (output / "reports/throughput.png").exists()
        cli("report", str(output))
        assert not (root / ".vllm-bench/state.json").exists()

        for name in ["startup_failure", "benchmark_failure", "timeout"]:
            failure = copy.deepcopy(config)
            failure["name"] = name
            failure["benchmarks"] = {"first": failure["benchmarks"]["first"]}
            failure["benchmark_scenarios"] = failure["benchmark_scenarios"][:1]
            if name == "startup_failure":
                failure["global_args"]["fail_startup"] = True
            elif name == "benchmark_failure":
                failure["benchmarks"]["first"]["script_args"] = {"fail": True}
            else:
                failure["benchmarks"]["first"]["script_args"] = {"delay": 10}
                failure["benchmark_timeout"] = 0.2
            (root / "config.yaml").write_text(yaml.safe_dump(failure))
            cli("run", "scenarios", expected=1)
            saved = root / f"results/local-{name}/run001"
            record = json.loads((saved / "reports/summary.json").read_text())[0]
            assert record["status"] == "failed" and record["error"], record
            assert not (root / ".vllm-bench/state.json").exists()

        cancellation = copy.deepcopy(failure)
        cancellation["name"] = "cancellation"
        cancellation["benchmark_timeout"] = 60
        (root / "config.yaml").write_text(yaml.safe_dump(cancellation))
        saved = root / "results/local-cancellation/run001"
        marker = saved / "data/first/baseline/run_1/benchmark.started"
        with (root / "cancel.log").open("w") as stream:
            process = subprocess.Popen(
                [sys.executable, "-m", "vllm_bench.cli", "run", "scenarios"],
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 60
                while not marker.exists():
                    assert process.poll() is None, (root / "cancel.log").read_text()
                    assert time.monotonic() < deadline, "Benchmark did not start"
                    time.sleep(0.1)
                process.send_signal(signal.SIGINT)
                assert process.wait(timeout=60) == 130
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        record = json.loads((saved / "reports/summary.json").read_text())[0]
        assert record["status"] == "cancelled", record
        assert not (root / ".vllm-bench/state.json").exists()
        print(f"Compose smoke check passed: {root}")
    finally:
        cli("down")


if __name__ == "__main__":
    main()
