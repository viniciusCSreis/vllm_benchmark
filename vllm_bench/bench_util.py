"""The small public interface used by startup and benchmark scripts."""

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _context():
    return json.loads(Path(os.environ["VLLM_BENCH_CONTEXT"]).read_text())


def get_global_args():
    return dict(_context()["global_args"])


def get_global_arg(name, default=None):
    return _context()["global_args"].get(name, default)


def get_scenario_arg(name, default=None):
    return _context()["scenario_args"].get(name, default)


def get_benchmark_arg(name, default=None):
    return _context()["benchmark_args"].get(name, default)


def get_output_dir():
    path = Path(_context()["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_dir():
    return Path(_context()["project_dir"])


def get_service():
    """Name, role, and replica of the calling startup script (None on the client)."""
    return _context().get("service")


def get_service_url(name, *, internal=False):
    services = _context()["internal_services" if internal else "services"]
    matches = [
        item for item in services if item["group"] == name or item["name"] == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one service named {name}; found {len(matches)}")
    return matches[0]["url"]


def get_leader_url():
    return next(
        item["url"] for item in _context()["services"] if item["role"] == "leader"
    )


def get_worker_urls():
    return [item["url"] for item in _context()["services"] if item["role"] == "worker"]


def get_service_path(path):
    """Translate a client project/trial path into the path visible to services."""
    path = Path(path).resolve()
    context = _context()
    for local, remote in [
        (context["trial_dir"], "/run/vllm-bench/trial"),
        (context["project_dir"], context["service_project_dir"]),
    ]:
        local = Path(local).resolve()
        if path.is_relative_to(local):
            destination = str(Path(remote) / path.relative_to(local))
            if sandbox_id := context.get("modal_sandbox_id"):
                from .providers.modal import upload_path

                upload_path(sandbox_id, path, destination)
            return destination
    raise ValueError(f"Path must be inside the project or trial directory: {path}")


@lru_cache(maxsize=1)
def get_gpu_count():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return len(result.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        return 0


def is_gpu_available():
    return get_gpu_count() > 0


def is_cpu():
    return not is_gpu_available()


def __getattr__(name):
    if name == "current_run":
        return _context()["repeat"]
    if name == "current_benchmark":
        return _context()["benchmark"]
    raise AttributeError(name)
