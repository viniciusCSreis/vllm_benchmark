"""Per-engine throughput over each sampling interval, for every trial."""

from .charts import counter_timeline


def generate(data):
    counter_timeline(
        data,
        "throughput.png",
        {
            "Inference requests / s (sample interval)": (
                "vllm:request_success_total",
                None,
            ),
            "Prompt tokens / s (sample interval)": ("vllm:prompt_tokens_total", None),
            "Generated tokens / s (sample interval)": (
                "vllm:generation_tokens_total",
                None,
            ),
        },
    )
