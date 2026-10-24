"""Run a scenario's services together on Modal; keep benchmarks on the client."""

import json
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

from ..artifacts import write_json
from ..processes import running_process
from ..progress import log
from .base import Provider
from .context import script_context, service_instances
from .paths import CONTAINER_PROJECT, CONTAINER_TRIAL

CONTROL = CONTAINER_TRIAL / ".provider"
CACHE = Path("/mnt/vllm-bench-cache")


def modal_sdk():
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError(
            'Install Modal support: pip install "modal>=1.6.2,<2"'
        ) from exc
    return modal


def remote_command(sandbox, *command):
    process = sandbox.exec(*command)
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    process.wait()
    if process.returncode:
        raise RuntimeError(f"Modal command failed: {stderr or stdout}")
    return stdout


def upload_path(sandbox_id, source, destination):
    """Publish benchmark inputs created locally after service startup."""
    sandbox = modal_sdk().Sandbox.from_id(sandbox_id)
    source = Path(source)
    if source.is_dir():
        log(f"Packing benchmark inputs: {source}")
        with tempfile.TemporaryDirectory(prefix="vllm-bench-input-") as folder:
            archive = Path(folder) / "inputs.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for path in source.rglob("*"):
                    if path.is_file() and not path.is_symlink():
                        bundle.write(path, path.relative_to(source))
                count = len(bundle.infolist())
            remote_archive = f"/tmp/{Path(folder).name}.zip"
            log(f"Uploading {count} input files as one archive to {destination}")
            sandbox.filesystem.copy_from_local(archive, remote_archive)
            remote_command(
                sandbox,
                "python3",
                "-c",
                "import pathlib, sys, zipfile; "
                "pathlib.Path(sys.argv[2]).mkdir(parents=True, exist_ok=True); "
                "zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2]); "
                "pathlib.Path(sys.argv[1]).unlink()",
                remote_archive,
                destination,
            )
        log(f"Benchmark inputs ready: {destination}")
    else:
        log(f"Uploading benchmark input: {source}")
        sandbox.filesystem.copy_from_local(source, destination)


def stop_saved(state_path):
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    sandbox = modal_sdk().Sandbox.from_id(state["sandbox_id"])
    directory = Path(state["directory"])
    log(f"Stopping Modal sandbox {sandbox.object_id}...")
    try:
        if sandbox.poll() is None:
            sandbox.filesystem.write_text("stop", str(CONTROL / "stop"))
            deadline = time.monotonic() + 60
            while True:
                try:
                    sandbox.filesystem.read_text(str(CONTROL / "stopped.json"))
                    break
                except (
                    FileNotFoundError,
                    modal_sdk().exception.SandboxFilesystemNotFoundError,
                ):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Modal services did not stop within 60 seconds"
                        )
                    time.sleep(0.5)
            log("Downloading Modal service output and logs...")
            remote_command(
                sandbox,
                "python3",
                "-c",
                "import shutil; shutil.make_archive('/tmp/vllm-bench-output', 'zip', "
                "'/run/vllm-bench/trial', 'services')",
            )
            archive = directory / ".provider/modal-output.zip"
            sandbox.filesystem.copy_to_local("/tmp/vllm-bench-output.zip", archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(directory)
            archive.unlink()
        else:
            raise RuntimeError(
                "Modal sandbox exited before output could be collected; see modal.log"
            )
    except Exception:
        # Filesystem access (or even poll) can race with an OOM termination.
        # Modal's output streams can still contain the reason for the shutdown.
        try:
            (directory / "modal.log").write_text(
                sandbox.stdout.read() + sandbox.stderr.read()
            )
        except Exception as exc:
            log(f"Could not download Modal logs: {exc}")
        raise
    finally:
        try:
            sandbox.terminate()
        finally:
            state_path.unlink(missing_ok=True)
    log(f"Modal environment stopped. Output: {directory}")


class ModalProvider(Provider):
    def __init__(self, config, root):
        self.config = config
        self.root = root
        self.app = None
        self.state_path = root / ".vllm_bench/state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else None
        )

    @contextmanager
    def run_session(self):
        modal = modal_sdk()
        with modal.enable_output(), modal.App(self.config.modal.app).run() as app:
            self.app = app
            log(f"Started fresh Modal app: {app.app_id}")
            try:
                yield
            finally:
                self.app = None
                log(f"Stopping Modal app: {app.app_id}")

    def start(self, trial, directory):
        if self.state_path.exists():
            raise RuntimeError("An environment is already active; run vllm_bench down")
        modal = modal_sdk()
        settings = self.config.modal
        instances = service_instances(self.config, trial, shared_network=True)
        project = (
            Path(self.config.image_workdir)
            if self.config.skip_volume_mount
            else CONTAINER_PROJECT
        )
        context = script_context(self.config, trial, project, instances)
        log(f"Starting Modal with image: {self.config.base_image}")
        log(
            f"Resources: {settings.cpu:g} CPUs, {settings.memory} MiB RAM, "
            f"GPU: {settings.gpu or 'none'}"
        )
        # Ship the current library without installing packages at service startup.
        image = (
            modal.Image.from_registry(self.config.base_image)
            .entrypoint([])
            .add_local_dir(
                Path(__file__).parents[1],
                "/opt/vllm-bench-lib/vllm_bench",
                copy=True,
                ignore=["**/__pycache__/**"],
            )
        )
        environment = {
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/opt/vllm-bench-lib:" + self.config.image_workdir,
            "HF_HOME": str(CACHE / "huggingface"),
            "VLLM_CACHE_ROOT": str(CACHE / "vllm"),
            "XDG_CACHE_HOME": str(CACHE),
            "NO_PROXY": "*",
        }
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                "python3",
                "-m",
                "vllm_bench.providers.modal_runtime",
                app=self.app
                if self.app is not None
                else modal.App.lookup(settings.app, create_if_missing=True),
                image=image,
                env=environment,
                gpu=settings.gpu,
                cpu=settings.cpu,
                memory=settings.memory,
                timeout=settings.timeout,
                encrypted_ports=[urlsplit(i["url"]).port for i in instances.values()],
                volumes={
                    str(CACHE): modal.Volume.from_name(
                        settings.cache_volume, create_if_missing=True
                    )
                },
            )
        self.state = {
            "provider": "modal",
            "image": self.config.base_image,
            "sandbox_id": sandbox.object_id,
            "directory": str(directory),
            "services": list(instances),
        }
        write_json(self.state_path, self.state)
        write_json(directory / ".provider/provider.json", self.state)
        if not self.config.skip_volume_mount:
            archive = next(
                parent / "project.zip"
                for parent in directory.parents
                if (parent / "project.zip").is_file()
            )
            log("Uploading project snapshot to Modal...")
            sandbox.filesystem.copy_from_local(archive, "/tmp/vllm-bench-project.zip")
            remote_command(
                sandbox,
                "python3",
                "-c",
                "import zipfile; zipfile.ZipFile('/tmp/vllm-bench-project.zip')"
                ".extractall('/workspace/project')",
            )
        launch = []
        for name, instance in instances.items():
            service = self.config.services[instance["group"]]
            output = CONTAINER_TRIAL / "services" / name
            remote_command(sandbox, "mkdir", "-p", str(output))
            service_context = {
                **context,
                "service": instance,
                "output_dir": str(output),
            }
            remote_context = CONTROL / f"{name}.json"
            write_json(directory / ".provider" / f"{name}.json", service_context)
            sandbox.filesystem.write_text(
                json.dumps(service_context), str(remote_context)
            )
            launch.append(
                {
                    "name": name,
                    "script": service.startup_script,
                    "project": str(project),
                    "environment": {
                        **environment,
                        **service.environment,
                        "VLLM_BENCH_CONTEXT": str(remote_context),
                    },
                }
            )
        sandbox.filesystem.write_text(json.dumps(launch), str(CONTROL / "launch.json"))
        tunnels = sandbox.tunnels(timeout=int(self.config.startup_timeout))
        reachable = [
            {**instance, "url": tunnels[urlsplit(instance["url"]).port].url}
            for instance in instances.values()
        ]
        client = directory / ".provider/client.json"
        write_json(
            client,
            {
                **context,
                "modal_sandbox_id": sandbox.object_id,
                "services": reachable,
                "project_dir": str(self.root),
                "trial_dir": str(directory),
                "output_dir": str(directory),
            },
        )
        write_json(directory / "endpoints.json", reachable)
        log(f"Started {len(instances)} services in parallel on Modal.")
        return client, reachable

    def check_running(self):
        sandbox = modal_sdk().Sandbox.from_id(self.state["sandbox_id"])
        if sandbox.poll() is not None:
            raise RuntimeError("Modal sandbox stopped")
        try:
            status = json.loads(
                sandbox.filesystem.read_text(str(CONTROL / "status.json"))
            )
        except (
            FileNotFoundError,
            modal_sdk().exception.SandboxFilesystemNotFoundError,
        ):
            return
        stopped = [name for name, code in status.items() if code is not None]
        if stopped:
            raise RuntimeError(
                f"Services stopped: {', '.join(stopped)}; see container logs"
            )

    @contextmanager
    def follow_logs(self):
        with running_process(
            [sys.executable, "-m", "vllm_bench.providers.modal", str(self.state_path)],
            self.root,
        ) as process:
            yield process

    def stop(self):
        if self.state is not None:
            stop_saved(self.state_path)
            self.state = None


def stream_logs(state_path):
    state = json.loads(Path(state_path).read_text())
    sandbox = modal_sdk().Sandbox.from_id(state["sandbox_id"])

    def forward(stream):
        for line in stream:
            print(line, end="", flush=True)

    stderr = Thread(target=forward, args=(sandbox.stderr,), daemon=True)
    stderr.start()
    forward(sandbox.stdout)
    stderr.join()


if __name__ == "__main__":
    stream_logs(sys.argv[1])
