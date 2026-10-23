"""Compare existing vLLM preemption counter differences."""

from .charts import comparison
from .data import engine_total


def generate(data):
    comparison(
        data,
        "preemptions.png",
        {"Preemptions": lambda r: engine_total(r, "preemptions")},
    )
