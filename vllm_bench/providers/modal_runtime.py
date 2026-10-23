"""Start service scripts together inside one Modal sandbox."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

from ..artifacts import write_json

TRIAL = Path("/run/vllm-bench/trial")
CONTROL = TRIAL / ".provider"


def forward_logs(process, name):
    output = TRIAL / "services" / name / "container.log"
    with output.open("w") as stream:
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(f"{name} | {line}", end="", flush=True)


def main():
    (TRIAL / "services").mkdir(parents=True, exist_ok=True)
    # The provider uploads project and context files before releasing startup.
    while not (CONTROL / "launch.json").exists():
        if (CONTROL / "stop").exists():
            write_json(CONTROL / "stopped.json", True)
            while True:
                time.sleep(60)
        time.sleep(0.2)
    settings = json.loads((CONTROL / "launch.json").read_text())
    processes = {}
    threads = []
    try:
        for item in settings:
            name = item["name"]
            process = subprocess.Popen(
                [sys.executable, "-m", "vllm_bench.service", item["script"]],
                cwd=item["project"],
                env={**os.environ, **item["environment"]},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            processes[name] = process
            thread = Thread(target=forward_logs, args=(process, name), daemon=True)
            thread.start()
            threads.append(thread)
        while not (CONTROL / "stop").exists():
            write_json(
                CONTROL / "status.json",
                {name: p.poll() for name, p in processes.items()},
            )
            time.sleep(0.5)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for thread in threads:
            thread.join(timeout=5)
        write_json(CONTROL / "stopped.json", True)
    # Keep the filesystem available while the client downloads results.
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
