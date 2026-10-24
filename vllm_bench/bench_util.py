"""Runtime utilities and compatibility getters; use enviroment for typed state."""

import csv
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

from .enviroment import _context as _context


def get_global_args():
    return dict(_context()["global_args"])


def get_service_args(name=None):
    """Global arguments with overrides for the current or named service group."""
    if name is None and (service := get_service()):
        name = service["group"]
    overrides = _context()["service_args"][name] if name is not None else {}
    return {**get_global_args(), **overrides}


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


def prepare_dataset(archive, records, fraction=1.0):
    """Sample a ZIP's CSV rows or matching files, keeping supporting files.

    ``records`` is a CSV member path or a glob for record files. Files use
    natural numeric order. Write inputs and dataset.json under the trial output
    directory and return the inputs directory without changing the archive.
    """
    if not 0 < fraction <= 1:
        raise ValueError("dataset_percentage must be greater than zero and at most one")
    folder = get_output_dir() / "inputs"
    with zipfile.ZipFile(archive) as bundle:
        if records.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(bundle.read(records).decode("utf-8"))))
            total = max(0, len(rows) - 1)
            selected = math.ceil(total * fraction)
            if not total:
                raise ValueError(f"No records in {archive}")
            path = folder / records
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows[: selected + 1])
        else:
            files = sorted(
                (name for name in bundle.namelist() if fnmatch(name, records)),
                key=lambda name: [
                    int(part) if part.isdigit() else part
                    for part in re.split(r"(\d+)", name)
                ],
            )
            total = len(files)
            selected = math.ceil(total * fraction)
            if not total:
                raise ValueError(f"No records in {archive}")
            excluded = set(files[selected:])
            for item in bundle.infolist():
                if item.filename not in excluded:
                    bundle.extract(item, folder)
    get_output_dir().joinpath("dataset.json").write_text(
        json.dumps(
            {
                "dataset_percentage": fraction,
                "total_records": total,
                "selected_records": selected,
            },
            indent=2,
        )
    )
    return folder


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
    if context.get("external_services"):
        # A shared Docker mount may expose the project under a different path.
        project = Path(context["project_dir"]).resolve()
        if path.is_relative_to(project):
            return str(Path(context["service_project_dir"]) / path.relative_to(project))
        return str(path)
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


def source_environment(package):
    """Use current Python sources with the image's installed native libraries."""
    project = get_project_dir()
    native = Path(importlib.util.find_spec(package).origin).parent
    overlay = Path(tempfile.mkdtemp(prefix="vllm-bench-source-"))
    shutil.copytree(
        project / package,
        overlay / package,
        copy_function=os.symlink,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
    )
    # Wheels also supply generated Python packages such as FlashAttention.
    # Keep fork files first, then fill missing files from the installed wheel.
    for file in native.rglob("*"):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
            continue
        target = overlay / package / file.relative_to(native)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(file.resolve())
    return {
        **os.environ,
        "PYTHONPATH": str(overlay),
        # `python -m` must not prepend the mounted working directory: its native
        # libraries may have been compiled for the host rather than the container.
        "PYTHONSAFEPATH": "1",
    }


def redact_environment(environment):
    """Copy launch environment for metadata, masking common credential names."""
    return {
        name: "<redacted>"
        if re.search(
            r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|"
            r"CREDENTIALS?|AUTHORIZATION)(?:_|$)",
            name.upper(),
        )
        else value
        for name, value in sorted(environment.items())
    }


def start_engine(command, options):
    """Launch a vLLM engine with source overlay, GPU checks, and startup metadata.

    ``command`` supplies the project's launcher; ``options`` supplies its vLLM
    flags, including model. The launcher receives MODEL and VLLM_PYTHON.
    Leaders use the last GPU; other replicas rotate over available GPUs.
    """
    command = list(command)
    options = dict(options)
    model = options.pop("model")
    service = get_service()
    environment = source_environment("vllm")
    environment.update(MODEL=model, VLLM_PYTHON=sys.executable)
    if is_gpu_available():
        count = get_gpu_count()
        index = (
            count - 1
            if service["role"] == "leader"
            else (service["replica"] - 1) % count
        )
        environment["CUDA_VISIBLE_DEVICES"] = str(index)
        # nvidia-smi sees allocated hardware even when the image runs CPU vLLM.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.metadata as metadata; import torch; "
                "from vllm.platforms import current_platform; "
                "print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, "
                "'vllm:', metadata.version('vllm'), "
                "'platform:', current_platform.device_type, flush=True); "
                "assert torch.cuda.is_available() and current_platform.is_cuda(), "
                "'An NVIDIA GPU is allocated, but PyTorch/vLLM cannot use CUDA. "
                "Use a CUDA-enabled image and check its NVIDIA driver support.'",
            ],
            cwd=get_project_dir(),
            env=environment,
            check=True,
        )
    for name, value in options.items():
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            command.extend(
                [
                    flag,
                    json.dumps(value)
                    if isinstance(value, (dict, list))
                    else str(value),
                ]
            )
    get_output_dir().joinpath("startup.json").write_text(
        json.dumps(
            {
                "command": command,
                "model": model,
                "gpu_count": get_gpu_count(),
                "environment": redact_environment(environment),
            },
            indent=2,
        )
    )
    os.execvpe(command[0], command, environment)


def __getattr__(name):
    if name == "current_run":
        return _context()["repeat"]
    if name == "current_benchmark":
        return _context()["benchmark"]
    raise AttributeError(name)
