"""Tiny HTTP stand-in demonstrating startup without a custom container image."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

from vllm_bench import bench_util


class Handler(BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == "/metrics":
            self.wfile.write(f"vllm:request_success_total {self.requests}\n".encode())
        else:
            self.wfile.write(b"ok")

    def do_POST(self):
        type(self).requests += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"results": [{"message": "hello"}]}).encode())


port = urlsplit(bench_util.get_service()["url"]).port
HTTPServer(("0.0.0.0", port), Handler).serve_forever()
