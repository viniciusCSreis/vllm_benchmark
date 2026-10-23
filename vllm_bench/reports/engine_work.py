"""Show mean prompt-token work per scenario and engine."""

import statistics

from matplotlib.figure import Figure


def generate(data):
    groups = {}
    for record in data.records:
        if record["status"] != "ok":
            continue
        for engine in record["engines"]:
            if engine.get("prompt_tokens") is not None:
                label = f"{record['scenario']}/{engine['engine']}"
                groups.setdefault(label, []).append(engine["prompt_tokens"])
    if not groups:
        return
    figure = Figure(figsize=(9, 5), layout="constrained")
    axis = figure.subplots()
    axis.bar(list(groups), [statistics.mean(values) for values in groups.values()])
    axis.tick_params(axis="x", labelrotation=30)
    axis.set_title("Mean prompt tokens per engine (successful repeats)")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(data.output / "engine_work.png", dpi=150)
