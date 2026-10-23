"""Number result directories and save the exact experiment inputs."""

import hashlib
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import yaml

from . import __version__
from .contracts import Trial
from .json_io import write_json
from .progress import log

EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "results",
    ".vllm-bench",
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".wheels",
}


def project_files(root):
    try:
        names = (
            subprocess.check_output(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .split("\0")
        )
    except subprocess.CalledProcessError:
        names = []
        for folder, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in EXCLUDED]
            names.extend(str((Path(folder) / name).relative_to(root)) for name in files)
    for name in sorted(set(names)):
        path = root / name
        if name == ".vllm_bench/state.json":
            continue
        if not name or not path.is_file() or path.is_symlink():
            continue
        if any(
            part in EXCLUDED or part.startswith(".env") or part.endswith(".egg-info")
            for part in Path(name).parts
        ):
            continue
        if any((root / parent).is_symlink() for parent in Path(name).parents):
            continue
        if path.suffix not in {".pyc", ".so", ".dylib", ".whl"}:
            yield path, name


def trials(config):
    return [
        Trial(benchmark=name, scenario=scenario.name, repeat=repeat)
        for name, benchmark in config.benchmarks.items()
        for scenario in benchmark.scenarios
        for repeat in range(1, config.runs + 1)
    ]


def prepare_run(config, root):
    parent = root / "results" / f"{config.provider}-{config.name}"
    parent.mkdir(parents=True, exist_ok=True)
    last = max(
        (
            int(match[1])
            for path in parent.iterdir()
            if path.is_dir() and (match := re.fullmatch(r"run(\d+)", path.name))
        ),
        default=0,
    )
    output = parent / f"run{last + 1:03d}"
    output.mkdir()
    log(f"Result directory: {output}")
    (output / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(exclude_none=True), sort_keys=False)
    )
    if not config.skip_volume_mount:
        archive = output / "project.zip"
        log("Saving project snapshot (excluding results, environments, and caches)...")
        count = 0
        last_update = time.monotonic()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path, name in project_files(root):
                bundle.write(path, name)
                count += 1
                if time.monotonic() - last_update >= 5:
                    log(f"Project snapshot: {count} files archived so far...")
                    last_update = time.monotonic()
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        (output / "project.sha256").write_text(f"{digest}  project.zip\n")
        log(
            f"Snapshot saved: {count} files, {archive.stat().st_size / 1024**2:.1f} MiB"
        )
    else:
        log("Project snapshot skipped: skip_volume_mount is enabled.")
    write_json(
        output / "run.json",
        {
            "schema_version": 1,
            "cli_version": __version__,
            "provider": config.provider,
            "image": config.base_image,
            "trials": [trial.model_dump() for trial in trials(config)],
        },
    )
    log("Saved effective config and run manifest.")
    return output
