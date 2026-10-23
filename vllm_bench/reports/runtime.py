"""Compare client runtimes, excluding service startup and metric collection."""

from .charts import comparison


def generate(data):
    comparison(data, "runtime.png", {"Client runtime (s)": lambda r: r["runtime_sec"]})
