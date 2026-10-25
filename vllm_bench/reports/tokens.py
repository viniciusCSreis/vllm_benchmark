"""Per-engine token counter growth relative to the initial trial snapshot."""

from matplotlib.figure import Figure

from ..metrics import difference
from .charts import trial_label
from .data import read_samples


def generate(data):
    figure = Figure(figsize=(12, 6), layout="constrained")
    axes = figure.subplots(2, 1)
    available = False
    for record in data.records:
        samples = read_samples(data.directory(record) / "metrics_samples.jsonl")
        if not samples:
            continue
        engines = sorted(set().union(*(sample["engines"] for sample in samples)))
        for engine in engines:
            missing = {"values": {}, "error": "Missing engine"}
            start = samples[0]["engines"].get(engine, missing)
            for axis, metric in zip(
                axes, ["vllm:prompt_tokens_total", "vllm:generation_tokens_total"]
            ):
                values = []
                for sample in samples:
                    delta = difference(
                        start, sample["engines"].get(engine, missing), metric
                    )
                    values.append(delta if delta is not None else float("nan"))
                    available = available or delta is not None
                axis.plot(
                    [sample["elapsed_sec"] for sample in samples],
                    values,
                    label=f"{trial_label(record)}/{engine} ({record['status']})",
                    drawstyle="steps-post",
                )
    for axis, title in zip(
        axes, ["Cumulative prompt tokens", "Cumulative generated tokens"]
    ):
        axis.set_title(title + " since initial snapshot")
        axis.set_xlabel("Seconds since initial metric snapshot")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    if available:
        axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        figure.savefig(data.output / "tokens_timeline.png", dpi=150)
