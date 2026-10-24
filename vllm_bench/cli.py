"""The user's machine owns scheduling, benchmark scripts, metrics, and reports."""

import argparse
import json
import shlex
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from . import __version__
from .artifacts import prepare_run, trials
from .config import load_config, resolve_config, select_benchmarks
from .external import prepare_context, service_endpoints
from .progress import log
from .providers import get_provider, stop_saved
from .providers.paths import CONTAINER_PROJECT
from .runner import run_only, run_trial, start_environment, wait_healthy


def interrupt(signum, frame):
    raise KeyboardInterrupt


def run_only_command(config, config_path, root, trial, endpoints):
    command = [
        "vllm_bench",
        "run_only",
        "--current-dir",
        str(root),
        "--config",
        str(config_path),
        "--benchmark",
        trial.benchmark,
        "--scenario",
        trial.scenario,
    ]
    if config.provider == "local" and not config.skip_volume_mount:
        command.extend(["--service-project-dir", str(CONTAINER_PROJECT)])
    urls = {}
    for item in endpoints:
        urls.setdefault(item["group"], item["url"])
    for name, url in urls.items():
        command.extend(["--service-url", f"{name}={url}"])
    if dataset := config.benchmarks[trial.benchmark].script_args.get("dataset_path"):
        command.extend(["--dataset-path", str(dataset)])
    return shlex.join(command)


def show_environment(
    provider, endpoints, directory, timeout, started, detach, run_command
):
    ready = False
    try:
        logs = nullcontext() if detach else provider.follow_logs()
        with logs as process:
            wait_healthy(provider, endpoints, timeout)
            log(f"Environment ready in {time.monotonic() - started:.1f}s.")
            log("Run the benchmark from another terminal:")
            print(run_command, flush=True)
            for item in endpoints:
                log(f"{item['name']} ({item['role']}): {item['url']}")
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
    run = commands.add_parser(
        "run_scenarios",
        help="Run selected trials with fresh managed services each time",
    )
    single = commands.add_parser(
        "run_only", help="Run one benchmark/scenario once against existing service URLs"
    )
    single.add_argument("--benchmark", required=True)
    single.add_argument("--scenario", required=True)
    single.add_argument(
        "--service-url",
        action="append",
        required=True,
        metavar="NAME=URL",
        help="Existing service base URL; repeat for each required service",
    )
    single.add_argument(
        "--service-project-dir",
        help="Project mount path visible to the services (e.g. /workspace/project)",
    )
    single.add_argument(
        "--dataset-path",
        help="Existing dataset path visible to the services (passed to the benchmark)",
    )
    report = commands.add_parser("report", help="Regenerate reports from saved files")
    report.add_argument("directory", type=Path)
    for command in [up, down, run, single, report]:
        command.add_argument(
            "--current-dir",
            type=Path,
            default=argparse.SUPPRESS,
            help="Project root containing .vllm_bench",
        )
    for command in [up, run, single]:
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
        external = args.command == "run_only"
        config = select_benchmarks(
            load_config(config_path),
            [args.benchmark] if external else args.benchmarks,
            [args.scenario] if external else args.scenarios,
        )
        if external:
            config = config.model_copy(
                update={
                    "provider": "external",
                    "runs": 1,
                    "skip_volume_mount": True,
                }
            )
            if args.dataset_path:
                benchmark = config.benchmarks[args.benchmark]
                benchmark.script_args = {
                    **benchmark.script_args,
                    "dataset_path": args.dataset_path,
                }
        experiment = trials(config)
        first = experiment[0]
        if args.command in {"up", "run_only"}:
            pairs = sum(
                len(benchmark.scenarios) for benchmark in config.benchmarks.values()
            )
            if pairs != 1:
                raise ValueError(
                    f"{args.command} requires exactly one benchmark and one scenario"
                )
        if external:
            endpoints = service_endpoints(config, first, args.service_url)
        if args.command == "up":
            log(f"Configuration valid. Provider: {config.provider}")
            log(f"Starting benchmark '{first.benchmark}', scenario '{first.scenario}'.")
            if model := config.get_global_args(first.benchmark).get("model"):
                log(f"Model: {model}")
        scripts = [benchmark.script for benchmark in config.benchmarks.values()]
        if not external:
            if config.provider == "external":
                raise ValueError("Use run_only for external services")
            if not config.base_image:
                raise ValueError(f"{args.command} requires base_image in the config")
            groups = dict.fromkeys(
                name
                for benchmark_name, benchmark in config.benchmarks.items()
                for scenario in benchmark.scenarios
                for name in config.get_scenario(benchmark_name, scenario).services
            )
            for name in groups:
                script = config.services[name].startup_script
                if not script:
                    raise ValueError(
                        f"Managed service '{name}' requires startup_script"
                    )
                if not config.skip_volume_mount:
                    scripts.append(script)
        for script in scripts:
            if not (root / script).is_file():
                raise FileNotFoundError(f"Script not found: {root / script}")
        if external:
            from .reports import generate_reports

            output = prepare_run(config, root)
            directory = output / first.directory
            context = prepare_context(
                config,
                first,
                root,
                directory,
                endpoints,
                service_project_dir=args.service_project_dir,
            )
            try:
                result = run_only(config, root, first, directory, context, endpoints)
                if result.status == "cancelled":
                    return 130
                return 0 if result.status == "ok" else 1
            finally:
                generate_reports(output)
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
                    or active["global_args"] != config.get_global_args(first.benchmark)
                    or active.get("service_args", {})
                    != {
                        group: config.services[group].service_args
                        for group in config.get_scenario(
                            first.benchmark, first.scenario
                        ).services
                    }
                    or active["scenario_args"]
                    != config.get_scenario(first.benchmark, first.scenario).args
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
                    run_only_command(config, config_path, root, first, endpoints),
                )
            stop_saved(state_path)
        output = prepare_run(config, root)
        provider = get_provider(config, root)
        if args.command == "up":
            trial = experiment[0]
            directory = output / trial.directory
            directory.mkdir(parents=True)
            try:
                _, endpoints = start_environment(
                    config, provider, trial, directory, wait=False
                )
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
                run_only_command(config, config_path, root, first, endpoints),
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
