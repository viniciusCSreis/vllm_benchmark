"""Sample physical GPU utilization and framebuffer memory with nvidia-smi."""

import csv
import io
import json
import subprocess
import time
from contextlib import contextmanager
from threading import Event, Thread

FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)
KEYS = (
    "gpu_index",
    "gpu_uuid",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_w",
)


def gpu_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(FIELDS),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    devices = []
    for values in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        row = dict(zip(KEYS[:2], (v.strip() for v in values[:2])))
        for key, value in zip(KEYS[2:], values[2:]):
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = None  # Some devices do not expose power/temperature.
        devices.append(row)
    return devices


@contextmanager
def gpu_monitor(path, interval=1.0):
    """Capture startup, query, and shutdown; CPU-only runs record unavailability."""
    stop = Event()
    samples = []
    started = time.monotonic()

    with path.open("w") as log:

        def capture():
            sample = {"elapsed_sec": time.monotonic() - started, "devices": []}
            try:
                sample["devices"] = gpu_snapshot()
            except (OSError, subprocess.SubprocessError) as exc:
                sample["error"] = str(exc)
            samples.append(sample)
            log.write(json.dumps(sample) + "\n")
            log.flush()
            return "error" not in sample

        available = capture()

        def monitor():
            while not stop.wait(interval):
                capture()

        thread = Thread(target=monitor) if available else None
        if thread:
            thread.start()
        try:
            yield samples
        finally:
            stop.set()
            if thread:
                thread.join()
                capture()
