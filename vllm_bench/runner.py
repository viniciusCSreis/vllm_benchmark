"""The host-side trial lifecycle, independent of provider implementation."""

import json
import subprocess
import sys
import time

import requests

from .contracts import TrialResult
from .metrics import deltas, monitor
from .processes import running_process
from .progress import log


def wait_healthy(provider, endpoints, timeout):
    started = time.monotonic()
    deadline = time.monotonic() + timeout
    next_update = started + 10
    pending = {item["name"]: item["url"] + item["health_path"] for item in endpoints}
    log(
        f"Waiting for {len(pending)} service(s) to become healthy "
        f"(timeout: {timeout:g}s)."
    )
    for name, url in pending.items():
        log(f"Health check {name}: {url}")
    with requests.Session() as session:
        session.trust_env = False
        while pending:
            if provider is not None:
                provider.check_running()
            for name, url in list(pending.items()):
                try:
                    if session.get(url, timeout=2).status_code == 200:
                        del pending[name]
                        log(f"{name} is healthy ({time.monotonic() - started:.1f}s).")
                except requests.RequestException:
                    pass
            if time.monotonic() >= deadline and pending:
                raise TimeoutError(f"Services not healthy: {', '.join(pending)}")
            if pending:
                if time.monotonic() >= next_update:
                    log(
                        f"Still waiting for {', '.join(pending)} "
                        f"({time.monotonic() - started:.0f}s elapsed). "
                        "Models may still be loading; see the service startup logs."
                    )
                    next_update = time.monotonic() + 10
                time.sleep(0.5)


def start_environment(config, provider, trial, directory, *, wait=True):
    """Shared startup and readiness checks for up and managed trials."""
    directory.mkdir(parents=True, exist_ok=True)
    context, endpoints = provider.start(trial, directory)
    if wait:
        wait_healthy(provider, endpoints, config.startup_timeout)
    return context, endpoints


def run_only(config, root, trial, directory, context, endpoints, *, provider=None):
    """Execute one trial against supplied endpoints without changing services."""
    directory.mkdir(parents=True, exist_ok=True)
    scenario = config.get_scenario(trial.benchmark, trial.scenario)
    result = TrialResult(
        **trial.model_dump(),
        global_args=config.global_args,
        scenario_args=scenario.args,
    )
    samples = []
    print(f"Running {trial.benchmark}/{trial.scenario}/{trial.repeat}", flush=True)
    try:
        if provider is None:
            wait_healthy(None, endpoints, config.startup_timeout)
        metrics = {
            item["name"]: item["url"] + item["metrics_path"]
            for item in endpoints
            if item["metrics_path"]
        }
        with monitor(metrics, directory, config.sample_interval) as samples:
            started = time.perf_counter()
            try:
                with running_process(
                    [sys.executable, config.benchmarks[trial.benchmark].script],
                    root,
                    log=directory / "benchmark.log",
                    environment={"VLLM_BENCH_CONTEXT": str(context)},
                ) as process:
                    deadline = time.monotonic() + config.benchmark_timeout
                    while process.poll() is None:
                        if provider is not None:
                            provider.check_running()
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Benchmark timed out")
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    if process.returncode:
                        raise RuntimeError(
                            f"Benchmark exited with {process.returncode}; "
                            "see benchmark.log"
                        )
            finally:
                result.runtime_sec = time.perf_counter() - started
        result.latency_sec = result.runtime_sec
        response = directory / "response.json"
        if response.exists():
            payload = json.loads(response.read_text())
            if isinstance(payload, dict):
                latency = payload.get("latency_sec")
                if isinstance(latency, (int, float)) and not isinstance(latency, bool):
                    result.latency_sec = latency
                if isinstance(payload.get("results"), list):
                    result.output_rows = len(payload["results"])
        result.status = "ok"
    except KeyboardInterrupt:
        result.status, result.error = "cancelled", "Interrupted"
    except Exception as exc:
        result.error = str(exc)
        print(f"Trial failed: {exc}", flush=True)
    finally:
        if samples:
            before, after = samples[0]["engines"], samples[-1]["engines"]
            (directory / "metrics_before.json").write_text(json.dumps(before, indent=2))
            (directory / "metrics_after.json").write_text(json.dumps(after, indent=2))
            if result.runtime_sec:
                result.engines = deltas(before, after, result.runtime_sec)
        result.save(directory)
    return result


def run_trial(config, root, provider, trial, directory):
    """Create fresh services, run the shared executor, then stop the services."""
    directory.mkdir(parents=True, exist_ok=True)
    result = TrialResult(
        **trial.model_dump(),
        global_args=config.global_args,
        scenario_args=config.get_scenario(trial.benchmark, trial.scenario).args,
    )
    try:
        context, endpoints = start_environment(config, provider, trial, directory)
        result = run_only(
            config, root, trial, directory, context, endpoints, provider=provider
        )
    except KeyboardInterrupt:
        result.status, result.error = "cancelled", "Interrupted"
    except Exception as exc:
        result.error = str(exc)
        log(f"Trial startup failed: {exc}")
    finally:
        try:
            provider.stop()
        except Exception as exc:
            result.status = "failed"
            result.error = (
                f"{result.error + '; ' if result.error else ''}Cleanup failed: {exc}"
            )
        result.save(directory)
    return result
