"""Benchmark latency and per-interval inference latency means."""

from .charts import comparison, counter_timeline


def generate(data):
    comparison(
        data, "latency.png", {"Benchmark latency (s)": lambda r: r["latency_sec"]}
    )
    for filename, title, metric in [
        ("request_latency", "Request latency", "vllm:e2e_request_latency_seconds"),
        ("queue_wait", "Inference queue wait", "vllm:request_queue_time_seconds"),
        (
            "time_to_first_token",
            "Time to first token",
            "vllm:time_to_first_token_seconds",
        ),
    ]:
        counter_timeline(
            data,
            filename + ".png",
            {
                title + " (s, mean of observations in sample interval)": (
                    metric + "_sum",
                    metric + "_count",
                )
            },
        )
