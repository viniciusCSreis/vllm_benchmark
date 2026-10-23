"""One sampler for existing vLLM counters, histograms, and engine gauges."""

import json
import time
from contextlib import contextmanager
from threading import Event, Thread

import requests
from prometheus_client.parser import text_string_to_metric_families

COUNTERS = {
    "requests": "vllm:request_success_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "cache_queries": "vllm:prefix_cache_queries_total",
    "cache_hits": "vllm:prefix_cache_hits_total",
    "preemptions": "vllm:num_preemptions_total",
}
HISTOGRAMS = {
    "request_latency": "vllm:e2e_request_latency_seconds",
    "queue_time": "vllm:request_queue_time_seconds",
    "time_to_first_token": "vllm:time_to_first_token_seconds",
}
GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}


def parse_metrics(text):
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.startswith("vllm:"):
                values[sample.name] = values.get(sample.name, 0) + sample.value
    return values


def snapshot(endpoints):
    engines = {}
    with requests.Session() as session:
        session.trust_env = False
        for name, url in endpoints.items():
            try:
                response = session.get(url, timeout=2)
                response.raise_for_status()
                engines[name] = {"values": parse_metrics(response.text)}
            except (requests.RequestException, ValueError) as exc:
                engines[name] = {"values": {}, "error": str(exc)}
    return engines


def difference(start, end, metric):
    # Counter labels can first appear after the first inference.
    if start.get("error") or end.get("error") or metric not in end["values"]:
        return None
    value = end["values"][metric] - start["values"].get(metric, 0)
    return value if value >= 0 else None


def deltas(before, after, elapsed):
    rows = []
    for name, end in after.items():
        start = before.get(name, {"values": {}, "error": "No initial snapshot"})
        error = start.get("error") or end.get("error")
        row = {"engine": name, "error": error}

        for key, metric in COUNTERS.items():
            row[key] = difference(start, end, metric)
        for key, metric in HISTOGRAMS.items():
            total, count = (
                difference(start, end, metric + "_sum"),
                difference(start, end, metric + "_count"),
            )
            row[key + "_mean_sec"] = (
                total / count if total is not None and count else None
            )
        for key in ["requests", "prompt_tokens", "generation_tokens"]:
            row[key + "_per_sec"] = row[key] / elapsed if row[key] is not None else None
        queries, hits = row["cache_queries"], row["cache_hits"]
        row["cache_hit_rate"] = hits / queries if hits is not None and queries else None
        rows.append(row)
    return rows


@contextmanager
def monitor(endpoints, folder, interval):
    stop = Event()
    samples = []
    started = time.monotonic()
    with (folder / "metrics_samples.jsonl").open("w") as log:

        def capture():
            sample = {
                "elapsed_sec": time.monotonic() - started,
                "engines": snapshot(endpoints),
            }
            samples.append(sample)
            log.write(json.dumps(sample) + "\n")
            log.flush()

        def poll():
            while not stop.wait(interval):
                capture()

        capture()
        thread = Thread(target=poll)
        thread.start()
        try:
            yield samples
        finally:
            stop.set()
            thread.join()
            capture()
