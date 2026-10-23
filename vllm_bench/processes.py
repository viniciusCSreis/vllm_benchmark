"""Own child process groups so cancellation also stops their children."""

import os
import signal
import subprocess
from contextlib import contextmanager, nullcontext, suppress


@contextmanager
def running_process(command, cwd, *, log=None, environment=None):
    with log.open("w") if log else nullcontext() as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **(environment or {})},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
