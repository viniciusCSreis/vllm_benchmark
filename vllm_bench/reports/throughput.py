"""Rates are per-engine counter differences divided by client runtime."""

from .charts import comparison
from .data import engine_total


def generate(data):
    comparison(
        data,
        "throughput.png",
        {
            "Inference requests / s": lambda r: engine_total(r, "requests_per_sec"),
            "Prompt tokens / s": lambda r: engine_total(r, "prompt_tokens_per_sec"),
            "Generated tokens / s": lambda r: engine_total(
                r, "generation_tokens_per_sec"
            ),
        },
    )
