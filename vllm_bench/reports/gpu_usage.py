"""Physical GPU utilization and framebuffer memory timelines, when available."""

from matplotlib.figure import Figure

from .data import read_samples


def generate(data):
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 1)
    available = False
    for record in data.records:
        for path in sorted(data.directory(record).glob("services/*/gpu_samples.jsonl")):
            samples = read_samples(path)
            devices = {
                device["gpu_uuid"] for sample in samples for device in sample["devices"]
            }
            for device in sorted(devices):
                label = (
                    f"{record['benchmark']}/{record['scenario']}/{record['repeat']}/"
                    f"{path.parent.name}/{device}"
                )
                for axis, key in zip(axes, ["gpu_utilization_pct", "memory_used_mib"]):
                    values = []
                    for sample in samples:
                        reading = next(
                            (
                                item
                                for item in sample["devices"]
                                if item["gpu_uuid"] == device
                            ),
                            {},
                        )
                        value = reading.get(key)
                        values.append(value if value is not None else float("nan"))
                        available = available or value is not None
                    axis.plot(
                        [sample["elapsed_sec"] for sample in samples],
                        values,
                        label=label,
                    )
    if available:
        axes[0].set_ylabel("GPU utilization (%)")
        axes[1].set_ylabel("Framebuffer memory (MiB)")
        axes[1].set_xlabel("Seconds since service startup")
        axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        figure.savefig(data.output / "gpu_usage.png", dpi=150)
