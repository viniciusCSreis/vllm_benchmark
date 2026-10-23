import json
import time

import requests

from vllm_bench import bench_util

output = bench_util.get_output_dir()
(output / "benchmark.started").write_text("started")
if bench_util.get_benchmark_arg("fail", False):
    raise RuntimeError("Deliberate benchmark failure")
time.sleep(bench_util.get_benchmark_arg("delay", 0))
path = output / "inputs.txt"
path.write_text("shared input")
response = requests.post(
    bench_util.get_leader_url() + "/query",
    json={
        "path": bench_util.get_service_path(path),
    },
    timeout=5,
)
response.raise_for_status()
(output / "response.json").write_text(json.dumps(response.json()))
