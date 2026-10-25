FROM python:3.12-slim

COPY pyproject.toml README.md LICENSE /tmp/vllm-benchmark/
COPY vllm_bench /tmp/vllm-benchmark/vllm_bench
RUN python -m pip install /tmp/vllm-benchmark
