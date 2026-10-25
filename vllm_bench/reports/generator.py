"""Regenerate reports using only the saved result folder."""

import csv
import json

from . import (
    cache_hits,
    engine_activity,
    engine_work,
    gpu_usage,
    latency,
    queue_peak,
    runtime,
    throughput,
    tokens,
)
from .data import load_data

REPORTS = [
    runtime,
    latency,
    queue_peak,
    tokens,
    throughput,
    cache_hits,
    engine_work,
    engine_activity,
    gpu_usage,
]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_reports(result_dir):
    data = load_data(result_dir)
    (data.output / "preemptions.png").unlink(missing_ok=True)
    (data.output / "summary.json").write_text(json.dumps(data.records, indent=2))
    fields = [
        "benchmark",
        "scenario",
        "repeat",
        "status",
        "runtime_sec",
        "latency_sec",
        "output_rows",
        "error",
    ]
    write_csv(
        data.output / "runs.csv",
        [{key: record[key] for key in fields} for record in data.records],
    )
    engines = []
    for record in data.records:
        engines.extend(
            {
                "benchmark": record["benchmark"],
                "scenario": record["scenario"],
                "repeat": record["repeat"],
                "status": record["status"],
                **engine,
            }
            for engine in record["engines"]
        )
    write_csv(data.output / "engines.csv", engines)
    for report in REPORTS:
        report.generate(data)
    print(f"Reports: {data.output}", flush=True)
    return data.output
