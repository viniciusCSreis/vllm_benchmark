"""Prefix-cache hit rate is weighted by queries across all engines."""

from .charts import comparison
from .data import engine_total


def hit_rate(record):
    queries = engine_total(record, "cache_queries")
    hits = engine_total(record, "cache_hits")
    return hits / queries if hits is not None and queries else None


def generate(data):
    comparison(data, "cache_hits.png", {"Prefix-cache hit rate": hit_rate})
