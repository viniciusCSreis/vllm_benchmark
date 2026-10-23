"""Run each service replica in its own Compose container."""

import json
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import yaml

from ..artifacts import write_json
from ..processes import running_process
from ..progress import log
from .base import Provider

CONTAINER_TRIAL = Path("/run/vllm-bench/trial")
CONTAINER_PROJECT = Path("/workspace/project")


def compose(state, *arguments, stream=False):
    command = [
        "docker",
        "compose",
        "-p",
        state["project"],
        "-f",
        state["compose_file"],
        *arguments,
    ]
    result = subprocess.run(command, capture_output=not stream, text=True)
    if result.returncode:
        raise RuntimeError(
            (result.stderr or result.stdout or "").strip()
            or f"Docker Compose {' '.join(arguments)} failed (exit {result.returncode})"
        )
    return result.stdout or ""


def stop_saved(state_path):
    """Also used by `down`, without depending on the current config contents."""
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    try:
        log(f"Stopping Compose project {state['project']}...")
        compose(state, "stop", "--timeout", "30")
        log("Saving container logs and service output...")
        for service in state["services"]:
            output = Path(state["directory"]) / "services" / service
            output.mkdir(parents=True, exist_ok=True)
            (output / "container.log").write_text(
                compose(state, "logs", "--no-color", "--timestamps", service)
            )
    finally:
        log("Removing the Compose containers and network (keeping the model cache)...")
        compose(state, "down", "--timeout", "30", "--remove-orphans")
        state_path.unlink(missing_ok=True)
    log(f"Environment stopped. Output: {state['directory']}")


class ComposeProvider(Provider):
    def __init__(self, config, root):
        self.config = config
        self.root = root
        self.state_path = root / ".vllm_bench/state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else None
        )

    def start(self, trial, directory):
        if self.state_path.exists():
            raise RuntimeError("An environment is already active; run vllm_bench down")
        scenario = next(
            s
            for s in self.config.benchmark_scenarios.scenarios
            if s.name == trial.scenario
        )
        benchmark = self.config.benchmarks[trial.benchmark]
        log(f"Preparing Docker Compose with image: {self.config.base_image}")
        instances = {}
        for group in scenario.services:
            service = self.config.services[group]
            devices = f"GPUs: {service.gpus}" if service.gpus else "no GPUs requested"
            log(
                f"Service {group}: {service.replicas} replica(s), {devices}; "
                f"startup: {service.startup_script}"
            )
            for index in range(1, service.replicas + 1):
                name = f"{group}-{index}"
                instances[name] = {
                    "name": name,
                    "group": group,
                    "role": service.role,
                    "replica": index,
                    "url": f"http://{name}:{service.health_port}",
                    "health_path": service.health_path,
                    "metrics_path": service.metrics_path,
                }
        project = (
            Path(self.config.image_workdir)
            if self.config.skip_volume_mount
            else CONTAINER_PROJECT
        )
        if self.config.skip_volume_mount:
            log(f"Using project code already installed in the image at {project}.")
        else:
            log(f"Mounting project read-only: {self.root} -> {project}")
        context = {
            **trial.model_dump(),
            "sample_interval": self.config.sample_interval,
            "global_args": self.config.global_args,
            "scenario_args": scenario.args,
            "benchmark_args": benchmark.script_args,
            "project_dir": str(project),
            "service_project_dir": str(project),
            "trial_dir": str(CONTAINER_TRIAL),
            "services": list(instances.values()),
            "internal_services": list(instances.values()),
        }
        compose_services = {}
        for name, instance in instances.items():
            service = self.config.services[instance["group"]]
            service_output = directory / "services" / name
            service_output.mkdir(parents=True)
            context_file = directory / ".provider" / f"{name}.json"
            write_json(
                context_file,
                {
                    **context,
                    "service": instance,
                    "output_dir": str(CONTAINER_TRIAL / "services" / name),
                },
            )
            volumes = [
                {
                    "type": "bind",
                    "source": str(directory),
                    "target": str(CONTAINER_TRIAL),
                },
                "model_cache:/kalypso-cache",
            ]
            if not self.config.skip_volume_mount:
                volumes.append(
                    {
                        "type": "bind",
                        "source": str(self.root),
                        "target": str(project),
                        "read_only": True,
                    }
                )
            definition = {
                "image": self.config.base_image,
                "working_dir": str(project),
                "entrypoint": ["python", "-m", "vllm_bench.service"],
                "command": [service.startup_script],
                "init": True,
                "shm_size": "2gb",
                "volumes": volumes,
                "environment": {
                    **service.environment,
                    "VLLM_BENCH_CONTEXT": str(
                        CONTAINER_TRIAL / ".provider" / f"{name}.json"
                    ),
                    "PYTHONUNBUFFERED": "1",
                    "HF_HOME": "/kalypso-cache/huggingface",
                    "VLLM_CACHE_ROOT": "/kalypso-cache/vllm",
                    "NO_PROXY": "*",
                },
                "ports": [f"127.0.0.1::{service.health_port}"],
            }
            if service.gpus:
                device = {"driver": "nvidia", "capabilities": ["gpu"]}
                if service.gpus == "all":
                    device["count"] = "all"
                else:
                    device["device_ids"] = service.gpus
                definition["deploy"] = {
                    "resources": {"reservations": {"devices": [device]}}
                }
            compose_services[name] = definition
        compose_file = directory / ".provider/compose.yaml"
        compose_file.write_text(
            yaml.safe_dump(
                {
                    "services": compose_services,
                    "volumes": {"model_cache": {"name": "vllm-bench-model-cache"}},
                },
                sort_keys=False,
            )
        )
        self.state = {
            "project": f"vllmbench-{uuid4().hex[:12]}",
            "compose_file": str(compose_file),
            "directory": str(directory),
            "services": list(instances),
        }
        write_json(self.state_path, self.state)
        write_json(directory / ".provider/provider.json", self.state)
        log(f"Generated Compose file: {compose_file}")
        log(f"Starting {len(instances)} container(s) in parallel...")
        compose(self.state, "up", "-d", stream=True)
        log("Containers started. Discovering published service ports...")
        reachable = []
        for name, instance in instances.items():
            service = self.config.services[instance["group"]]
            address = compose(
                self.state, "port", name, str(service.health_port)
            ).strip()
            reachable.append({**instance, "url": f"http://{address}"})
        client_context = directory / ".provider/client.json"
        write_json(
            client_context,
            {
                **context,
                "services": reachable,
                "project_dir": str(self.root),
                "trial_dir": str(directory),
                "output_dir": str(directory),
            },
        )
        write_json(directory / "endpoints.json", reachable)
        log(
            "Follow service startup logs: "
            + shlex.join(
                [
                    "docker",
                    "compose",
                    "-p",
                    self.state["project"],
                    "-f",
                    str(compose_file),
                    "logs",
                    "-f",
                ]
            )
        )
        return client_context, reachable

    def check_running(self):
        text = compose(self.state, "ps", "--all", "--format", "json").strip()
        rows = (
            json.loads(text)
            if text.startswith("[")
            else [json.loads(line) for line in text.splitlines() if line]
        )
        running = {row["Service"] for row in rows if row["State"] == "running"}
        missing = set(self.state["services"]) - running
        if missing:
            raise RuntimeError(
                f"Services stopped: {', '.join(sorted(missing))}; see container logs"
            )

    @contextmanager
    def follow_logs(self):
        command = [
            "docker",
            "compose",
            "-p",
            self.state["project"],
            "-f",
            self.state["compose_file"],
            "logs",
            "--follow",
            "--no-color",
        ]
        with running_process(command, self.root) as process:
            yield process

    def stop(self):
        if self.state is not None:
            stop_saved(self.state_path)
            self.state = None
