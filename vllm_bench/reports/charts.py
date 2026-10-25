"""Shared helpers for individual trial charts and sampled counter rates."""

from matplotlib.figure import Figure

from ..metrics import difference
from .data import read_samples


def trial_label(record):
    return f"{record['benchmark']}/{record['scenario']}/trial {record['repeat']}"


def comparison(data, filename, panels):
    records = [record for record in data.records if record["status"] == "ok"]
    if not records:
        return
    figure = Figure(
        figsize=(max(7, len(records) * 0.8) * len(panels), 4), layout="constrained"
    )
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    available = False
    for axis, (title, measure) in zip(axes, panels.items()):
        for index, record in enumerate(records):
            value = measure(record)
            if value is not None:
                available = True
                axis.bar(index, value)
        axis.set_title(title)
        axis.set_xticks(
            range(len(records)),
            [trial_label(r) for r in records],
            rotation=25,
            ha="right",
        )
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Individual successful trials")
    if available:
        figure.savefig(data.output / filename, dpi=150)


def counter_timeline(data, filename, panels):
    figure = Figure(figsize=(12, 3 * len(panels)), layout="constrained")
    axes = figure.subplots(len(panels), 1, squeeze=False)[:, 0]
    available = False
    for record in data.records:
        samples = read_samples(data.directory(record) / "metrics_samples.jsonl")
        engines = sorted(set().union(*(sample["engines"] for sample in samples)))
        for engine in engines:
            label = f"{trial_label(record)}/{engine} ({record['status']})"
            for axis, (title, (metric, denominator)) in zip(axes, panels.items()):
                times, values = [], []
                for before, after in zip(samples, samples[1:]):
                    elapsed = after["elapsed_sec"] - before["elapsed_sec"]
                    missing = {"values": {}, "error": "Missing engine"}
                    start = before["engines"].get(engine, missing)
                    end = after["engines"].get(engine, missing)
                    delta = difference(start, end, metric)
                    divisor = (
                        difference(start, end, denominator) if denominator else elapsed
                    )
                    value = (
                        delta / divisor
                        if delta is not None and divisor and divisor > 0
                        else float("nan")
                    )
                    times.append(after["elapsed_sec"])
                    values.append(value)
                    available = available or value == value
                if times:
                    axis.stairs(
                        values,
                        [samples[0]["elapsed_sec"], *times],
                        baseline=None,
                        label=label,
                    )
    for axis, title in zip(axes, panels):
        axis.set_title(title)
        axis.set_xlabel("Seconds since initial metric snapshot")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    if available:
        axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        figure.savefig(data.output / filename, dpi=150)
