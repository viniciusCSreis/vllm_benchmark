"""Call the example service and save its response for the report."""

from urllib.request import Request, urlopen

from vllm_bench import bench_util

request = Request(bench_util.get_leader_url() + "/query", data=b"", method="POST")
with urlopen(request, timeout=10) as response:
    (bench_util.get_output_dir() / "response.json").write_bytes(response.read())
