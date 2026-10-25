"""Request and token totals per individual trial and engine."""

from matplotlib.figure import Figure

from .charts import trial_label


def generate(data):
    rows = [
        (f"{trial_label(record)}/{engine['engine']}", engine)
        for record in data.records
        if record["status"] == "ok"
        for engine in record["engines"]
    ]
    if not rows:
        return
    figure = Figure(figsize=(max(9, len(rows) * 0.8), 9), layout="constrained")
    axes = figure.subplots(3, 1)
    available = False
    for axis, (key, title) in zip(
        axes,
        [
            ("requests", "Completed inference requests"),
            ("prompt_tokens", "Prompt tokens"),
            ("generation_tokens", "Generated tokens"),
        ],
    ):
        for index, (_, engine) in enumerate(rows):
            value = engine.get(key)
            if value is not None:
                axis.bar(index, value)
                available = True
        axis.set_xticks(
            range(len(rows)), [label for label, _ in rows], rotation=25, ha="right"
        )
        axis.set_title(title + " per trial and engine")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
    if available:
        figure.savefig(data.output / "engine_work.png", dpi=150)
