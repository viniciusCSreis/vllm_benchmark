"""Container entry point: execute the user's Python startup and sample local GPUs."""

import signal
import sys
from pathlib import Path

from .bench_util import _context, get_output_dir, get_project_dir
from .gpu import gpu_monitor
from .processes import running_process


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (
            gpu_monitor(
                get_output_dir() / "gpu_samples.jsonl", _context()["sample_interval"]
            ),
            running_process(
                [sys.executable, str(Path(sys.argv[1]))], get_project_dir()
            ) as process,
        ):
            return process.wait()
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
