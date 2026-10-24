"""Call the example service and save its response for the report."""

from urllib.request import Request, urlopen

from vllm_bench import enviroment

request = Request(enviroment.leader.url + "/query", data=b"", method="POST")
with urlopen(request, timeout=10) as response:
    (enviroment.output_dir / "response.json").write_bytes(response.read())
