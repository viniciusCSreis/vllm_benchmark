"""Plot existing vLLM engine gauges, including partial trial histories."""

from matplotlib.figure import Figure

from .data import read_samples

GAUGES = {
    "Running inference requests": "vllm:num_requests_running",
    "Waiting inference requests": "vllm:num_requests_waiting",
    "KV-cache usage (fraction)": "vllm:kv_cache_usage_perc",
}


def generate(data):
    figure = Figure(figsize=(12, 9), layout="constrained")
    axes = figure.subplots(3, 1)
    available = False
    for record in data.records:
        samples = read_samples(data.directory(record) / "metrics_samples.jsonl")
        engines = set().union(*(sample["engines"] for sample in samples))
        for engine in sorted(engines):
            label = (
                f"{record['benchmark']}/{record['scenario']}/"
                f"{record['repeat']}/{engine} "
                f"({record['status']})"
            )
            for axis, (title, metric) in zip(axes, GAUGES.items()):
                values = [
                    sample["engines"]
                    .get(engine, {})
                    .get("values", {})
                    .get(metric, float("nan"))
                    for sample in samples
                ]
                if any(
                    metric in sample["engines"].get(engine, {}).get("values", {})
                    for sample in samples
                ):
                    available = True
                axis.plot(
                    [sample["elapsed_sec"] for sample in samples], values, label=label
                )
                axis.set_title(title)
                axis.set_xlabel("Seconds since initial metric snapshot")
                axis.grid(alpha=0.2)
    if available:
        axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        figure.savefig(data.output / "engine_activity.png", dpi=150)
