"""The user's machine owns scheduling, benchmark scripts, metrics, and reports."""

import argparse
import json
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from . import __version__
from .artifacts import prepare_run, trials
from .config import load_config, resolve_config, select_benchmarks
from .progress import log
from .providers import get_provider, stop_saved
from .runner import run_trial, wait_healthy


def interrupt(signum, frame):
    raise KeyboardInterrupt


def show_environment(provider, endpoints, directory, timeout, started, detach):
    ready = False
    try:
        logs = nullcontext() if detach else provider.follow_logs()
        with logs as process:
            wait_healthy(provider, endpoints, timeout)
            for item in endpoints:
                log(f"{item['name']} ({item['role']}): {item['url']}")
            log(f"Environment ready in {time.monotonic() - started:.1f}s.")
            log(f"Service output: {directory / 'services'}")
            ready = True
            if detach:
                log("Containers are running. Stop with: vllm_bench down")
                return 0
            log("Following container logs. Press Ctrl+C to stop the environment.")
            while process.poll() is None:
                provider.check_running()
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(
                    f"Container log stream exited with {process.returncode}"
                )
            return 0
    finally:
        if not detach or not ready:
            log("Stopping the environment and saving container logs...")
            provider.stop()


def main():
    parser = argparse.ArgumentParser(description="Run script-defined vLLM benchmarks.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path, default=Path("local_cpu.yaml"))
    parser.add_argument(
        "--current-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root containing .vllm_bench",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("up", help="Start services and follow their logs")
    up.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Leave services running in the background",
    )
    down = commands.add_parser(
        "down", help="Stop the active environment and collect output"
    )
    run = commands.add_parser("run", help="Run the experiment on this client")
    report = commands.add_parser("report", help="Regenerate reports from saved files")
    report.add_argument("directory", type=Path)
    for command in [up, down, run, report]:
        command.add_argument(
            "--current-dir",
            type=Path,
            default=argparse.SUPPRESS,
            help="Project root containing .vllm_bench",
        )
    for command in [up, run]:
        command.add_argument(
            "--config",
            type=Path,
            default=argparse.SUPPRESS,
            help="Config name or path, merged over base.yaml (e.g. modal)",
        )
    for command in [up, run]:
        command.add_argument(
            "--benchmark",
            "--benchmarks",
            dest="benchmarks",
            nargs="+",
            required=True,
            help="Benchmark names, or '*' for all benchmarks",
        )
        command.add_argument(
            "--scenario",
            "--scenarios",
            dest="scenarios",
            nargs="+",
            required=True,
            help="Scenario names, or '*' for every scenario in each selected benchmark",
        )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    root = args.current_dir.resolve()
    config_path = resolve_config(root / ".vllm_bench/configs" / args.config)
    state_path = root / ".vllm_bench/state.json"
    try:
        if args.command == "report":
            from .reports import generate_reports

            generate_reports((root / args.directory).resolve())
            return 0
        if not (root / ".vllm_bench").is_dir():
            raise FileNotFoundError(f"Project must contain .vllm_bench: {root}")
        if args.command == "down":
            stop_saved(state_path)
            return 0
        started = time.monotonic()
        if args.command == "up":
            log(f"Reading config: {config_path}")
        config = select_benchmarks(
            load_config(config_path), args.benchmarks, args.scenarios
        )
        experiment = trials(config)
        first = experiment[0]
        if args.command == "up":
            pairs = sum(
                len(benchmark.scenarios) for benchmark in config.benchmarks.values()
            )
            if pairs != 1:
                raise ValueError("up requires exactly one benchmark and one scenario")
            log(f"Configuration valid. Provider: {config.provider}")
            log(f"Starting benchmark '{first.benchmark}', scenario '{first.scenario}'.")
            if model := config.global_args.get("model"):
                log(f"Model: {model}")
        scripts = [benchmark.script for benchmark in config.benchmarks.values()]
        if not config.skip_volume_mount:
            groups = dict.fromkeys(
                name
                for benchmark in config.benchmarks.values()
                for scenario in benchmark.scenarios
                for name in scenario.services
            )
            scripts += [config.services[name].startup_script for name in groups]
        for script in scripts:
            if not (root / script).is_file():
                raise FileNotFoundError(f"Script not found: {root / script}")
        if state_path.exists():
            if args.command == "up":
                state = json.loads(state_path.read_text())
                directory = Path(state["directory"])
                active = json.loads((directory / ".provider/client.json").read_text())
                if (
                    state.get("provider", "local") != config.provider
                    or state.get("image") != config.base_image
                    or (active["benchmark"], active["scenario"])
                    != (first.benchmark, first.scenario)
                    or active["global_args"] != config.global_args
                    or active["scenario_args"]
                    != config.benchmarks[first.benchmark].scenarios[0].args
                ):
                    raise ValueError(
                        f"An environment for {active['benchmark']}/"
                        f"{active['scenario']} "
                        "is already active with different selection "
                        "or environment settings. "
                        "Run 'vllm_bench down' before switching."
                    )
                log(f"Attaching to {first.benchmark}/{first.scenario}: {directory}")
                endpoints = json.loads((directory / "endpoints.json").read_text())
                return show_environment(
                    get_provider(config, root),
                    endpoints,
                    directory,
                    config.startup_timeout,
                    started,
                    args.detach,
                )
            stop_saved(state_path)
        output = prepare_run(config, root)
        provider = get_provider(config, root)
        if args.command == "up":
            trial = experiment[0]
            directory = output / trial.directory
            directory.mkdir(parents=True)
            try:
                _, endpoints = provider.start(trial, directory)
            except BaseException:
                log("Startup did not complete. Stopping containers and saving logs...")
                provider.stop()
                log(f"Startup output: {directory}")
                raise
            return show_environment(
                provider,
                endpoints,
                directory,
                config.startup_timeout,
                started,
                args.detach,
            )
        from .reports import generate_reports

        failed = False
        try:
            for trial in experiment:
                result = run_trial(
                    config, root, provider, trial, output / trial.directory
                )
                if result.status == "cancelled":
                    return 130
                failed = failed or result.status != "ok"
                if state_path.exists():
                    # Preserve failed cleanup state for recovery; stop scheduling.
                    return 1
            return 1 if failed else 0
        finally:
            generate_reports(output)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
