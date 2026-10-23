import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from vllm_bench import bench_util

if bench_util.get_global_arg("fail_startup", False):
    raise RuntimeError("Deliberate startup failure")

output = bench_util.get_output_dir()
output.joinpath("startup.json").write_text(
    json.dumps(
        {
            "role": bench_util.get_service()["role"],
            "workers": bench_util.get_worker_urls(),
            "scenario": bench_util.get_scenario_arg("coordinator"),
        }
    )
)


class Handler(BaseHTTPRequestHandler):
    count = 0

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == "/metrics":
            text = (
                "vllm:request_success_total %d\n"
                "vllm:prompt_tokens_total %d\n"
                "vllm:generation_tokens_total %d\n"
                "vllm:prefix_cache_queries_total %d\n"
                "vllm:prefix_cache_hits_total %d\n"
                "vllm:num_preemptions_total 0\n"
                "vllm:num_requests_running 0\n"
                "vllm:num_requests_waiting 0\n"
                "vllm:kv_cache_usage_perc 0.1\n"
            ) % (
                self.count,
                self.count * 10,
                self.count * 2,
                self.count * 10,
                self.count * 5,
            )

            self.wfile.write(text.encode())
        else:
            self.wfile.write(b"ok")

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert Path(payload["path"]).read_text() == "shared input"
        type(self).count += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(
            json.dumps({"latency_sec": 0.01, "results": [{"ok": True}]}).encode()
        )

    def log_message(self, *args):
        pass


HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
