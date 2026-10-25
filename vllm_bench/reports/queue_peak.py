"""Observed peak inference queue across engines for each trial."""

from .charts import comparison
from .data import read_samples


def peak_queue(data, record):
    peaks = []
    for sample in read_samples(data.directory(record) / "metrics_samples.jsonl"):
        readings = [
            engine.get("values", {}).get("vllm:num_requests_waiting")
            if not engine.get("error")
            else None
            for engine in sample["engines"].values()
        ]
        if readings and all(value is not None for value in readings):
            peaks.append(sum(readings))
    return max(peaks) if peaks else None


def generate(data):
    comparison(
        data,
        "queue_peak.png",
        {
            "Peak sampled waiting requests (sum across engines)": lambda r: peak_queue(
                data, r
            )
        },
    )
