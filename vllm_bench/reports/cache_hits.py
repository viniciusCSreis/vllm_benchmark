"""Prefix-cache hits and query-weighted hit rate for each sampling interval."""

from .charts import comparison, counter_timeline
from .data import engine_total


def hit_rate(record):
    queries = engine_total(record, "cache_queries")
    hits = engine_total(record, "cache_hits")
    return hits / queries if hits is not None and queries else None


def generate(data):
    comparison(
        data, "cache_hit_rate.png", {"Prefix-cache hit rate (fraction)": hit_rate}
    )
    counter_timeline(
        data,
        "cache_hits.png",
        {
            "Prefix-cache hits / s (sample interval)": (
                "vllm:prefix_cache_hits_total",
                None,
            ),
            "Prefix-cache hit rate (fraction, sample interval)": (
                "vllm:prefix_cache_hits_total",
                "vllm:prefix_cache_queries_total",
            ),
        },
    )
