"""The user's machine owns scheduling, benchmark scripts, metrics, and reports."""

import argparse
import json
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .artifacts import prepare_run, trials
from .config import load_config
from .progress import log
from .providers import get_provider
from .providers.compose import stop_saved
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
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("up", help="Start services and follow their logs")
    up.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Leave services running in the background",
    )
    commands.add_parser("down", help="Stop the active environment and collect output")
    run = commands.add_parser("run", help="Run the experiment on this client")
    run.add_argument("target", choices=["scenarios"])
    report = commands.add_parser("report", help="Regenerate reports from saved files")
    report.add_argument("directory", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    config_path = args.config.resolve()
    root = config_path.parent
    state_path = root / ".vllm-bench/state.json"
    try:
        if args.command == "report":
            from .reports import generate_reports

            generate_reports(args.directory.resolve())
            return 0
        if args.command == "down":
            stop_saved(state_path)
            return 0
        started = time.monotonic()
        if args.command == "up":
            log(f"Reading config: {config_path}")
        config = load_config(config_path)
        if args.command == "up":
            first = trials(config)[0]
            log(f"Configuration valid. Provider: {config.provider}")
            log(
                f"Starting benchmark '{first.benchmark}', scenario '{first.scenario}' "
                f"(repeat {first.repeat})."
            )
            if model := config.global_args.get("model"):
                log(f"Model: {model}")
        # Check scripts before reserving a run or creating resources.
        scripts = [benchmark.script for benchmark in config.benchmarks.values()]
        if not config.skip_volume_mount:
            scripts += [service.startup_script for service in config.services.values()]
        for script in scripts:
            if not (root / script).is_file():
                raise FileNotFoundError(f"Script not found: {root / script}")
        if state_path.exists():
            if args.command == "up":
                state = json.loads(state_path.read_text())
                log(f"An environment is already active: {state['directory']}")
                directory = Path(state["directory"])
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
        experiment = trials(config)
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
