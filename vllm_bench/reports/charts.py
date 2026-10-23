"""Small shared chart helpers; each report module chooses its measurements."""

import statistics

from matplotlib.figure import Figure


def comparison(data, filename, panels):
    records = [record for record in data.records if record["status"] == "ok"]
    names = list(
        dict.fromkeys((record["benchmark"], record["scenario"]) for record in records)
    )
    if not names:
        return
    figure = Figure(figsize=(5 * len(panels), 4), layout="constrained")
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    available = False
    for axis, (title, measure) in zip(axes, panels.items()):
        for index, name in enumerate(names):
            values = [
                value
                for record in records
                if (record["benchmark"], record["scenario"]) == name
                and (value := measure(record)) is not None
            ]
            if values:
                available = True
                axis.bar(
                    index,
                    statistics.mean(values),
                    yerr=statistics.stdev(values) if len(values) > 1 else 0,
                    capsize=3,
                )
        axis.set_title(title)
        axis.set_xticks(
            range(len(names)),
            ["/".join(name) for name in names],
            rotation=25,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Successful repeats: mean and sample standard deviation")
    if available:
        figure.savefig(data.output / filename, dpi=150)
