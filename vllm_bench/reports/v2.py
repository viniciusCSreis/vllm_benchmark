"""Compact scenario comparisons, a combined PDF, and standalone chart images."""

import csv
import json
from statistics import mean
from textwrap import fill

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from ..metrics import difference
from .data import engine_total

INK = "#17343D"
TEAL = "#167C78"
MUTED = "#526471"
COLORS = ["#7C8795", "#4C78A8", "#24968D", "#B78037", "#8B6BA6", "#C36E64"]
ENGINE_COLORS = COLORS[1:] + COLORS[:1]
CHARTS = [
    ("query_latency_sec", "Complete-query latency", "Seconds", 1),
    ("cache_hit_pct", "Prefix-cache hit rate", "Percent of queried tokens", 1),
    ("ttft_mean_sec", "Mean time to first token", "Seconds per request", 1),
    ("prompt_tokens", "Logical prompt-token volume", "Millions of tokens", 1e6),
    (
        "uncached_query_tokens",
        "Uncached prefix-query tokens",
        "Millions of tokens",
        1e6,
    ),
    ("requests", "Completed internal LLM requests", "Requests", 1),
    ("generation_tokens", "Generated-token volume", "Thousands of tokens", 1e3),
]
STACKED = {"prompt_tokens", "requests"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def time_to_first_token(directory, engines):
    """Pool histogram observations, rather than averaging per-engine means."""
    before = read_json(directory / "metrics_before.json")
    after = read_json(directory / "metrics_after.json")
    total = count = 0
    if not engines:
        return None
    for engine in engines:
        missing = {"values": {}, "error": "Missing engine"}
        start = before.get(engine["engine"], missing)
        end = after.get(engine["engine"], missing)
        prefix = "vllm:time_to_first_token_seconds"
        elapsed = difference(start, end, prefix + "_sum")
        observations = difference(start, end, prefix + "_count")
        if elapsed is None or observations is None:
            return None
        total += elapsed
        count += observations
    return total / count if count else None


def trial_rows(data):
    rows = []
    for record in data.records:
        hits = engine_total(record, "cache_hits")
        queries = engine_total(record, "cache_queries")
        row = {
            key: record[key]
            for key in [
                "benchmark",
                "scenario",
                "repeat",
                "status",
                "output_rows",
                "error",
            ]
        }
        row.update(
            query_latency_sec=record["latency_sec"],
            cache_hit_pct=100 * hits / queries
            if hits is not None and queries
            else None,
            ttft_mean_sec=time_to_first_token(
                data.directory(record), record["engines"]
            ),
            prompt_tokens=engine_total(record, "prompt_tokens"),
            uncached_query_tokens=queries - hits
            if queries is not None and hits is not None and queries >= hits
            else None,
            requests=engine_total(record, "requests"),
            generation_tokens=engine_total(record, "generation_tokens"),
            engines={
                engine["engine"]: {
                    key: engine.get(key) for key in ["prompt_tokens", "requests"]
                }
                for engine in record["engines"]
            },
        )
        rows.append(row)
    return rows


def label(name):
    return fill(name.replace("_", " ").capitalize(), width=9)


def format_value(value):
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def draw_chart(axis, chart, groups, scenario_colors, engine_colors):
    key, title, unit, scale = chart
    axis.set_title(title, loc="left", color=INK, weight="bold", fontsize=10, pad=16)
    axis.set_ylabel(unit, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#A5AFB7")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E4E8EB", linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8, length=0, pad=5)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    labels, annotations, maximum = [], [], 0
    for index, (scenario, rows) in enumerate(groups.items()):
        trials = [r for r in rows if r["status"] == "ok" and r[key] is not None]
        labels.append(f"{label(scenario)}\n(n={len(trials)})")
        if not trials:
            axis.text(
                index,
                0.04,
                "N/A",
                transform=axis.get_xaxis_transform(),
                ha="center",
                color=MUTED,
                fontsize=9,
            )
            continue
        values = [row[key] / scale for row in trials]
        value = mean(values)
        if key in STACKED:
            bottom = 0
            for engine, color in engine_colors.items():
                height = mean(
                    row["engines"].get(engine, {}).get(key, 0) / scale for row in trials
                )
                axis.bar(index, height, bottom=bottom, color=color, width=0.64)
                bottom += height
        else:
            axis.bar(index, value, color=scenario_colors[scenario], width=0.64)
        if len(values) > 1:
            positions = [
                index + (i / (len(values) - 1) - 0.5) * 0.22 for i in range(len(values))
            ]
            axis.scatter(
                positions,
                values,
                s=15,
                facecolor="white",
                edgecolor=INK,
                linewidth=0.7,
                zorder=3,
            )
        annotations.append((index, value))
        maximum = max(maximum, *values)
    upper = max(112 if key == "cache_hit_pct" else maximum * 1.25, 1)
    axis.set_ylim(0, upper)
    axis.set_xlim(-0.65, len(groups) - 0.35)
    axis.set_xticks(range(len(groups)), labels)
    if key == "cache_hit_pct":
        axis.set_yticks([0, 25, 50, 75, 100])
    for index, value in annotations:
        axis.text(
            index,
            value + upper * 0.025,
            format_value(value),
            ha="center",
            color=INK,
            fontsize=8.5,
        )
    if key in STACKED and annotations:
        handles = [
            mpl.patches.Patch(color=color, label=engine)
            for engine, color in engine_colors.items()
        ]
        axis.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=min(3, len(handles)),
            frameon=False,
            fontsize=7,
            borderaxespad=0,
            handlelength=1,
        )


def page(data, benchmark, records, title, page_number, page_count, manifest):
    figure = Figure(figsize=(8.5, 11), facecolor="white")
    figure.text(
        0.075,
        0.965,
        f"VLLM BENCH / {data.root.name.upper()}",
        fontsize=8.5,
        weight="bold",
        color=TEAL,
    )
    figure.text(
        0.925,
        0.965,
        f"{page_number} / {page_count}",
        fontsize=8.5,
        ha="right",
        color=INK,
    )
    figure.text(0.075, 0.925, title, fontsize=21, weight="bold", color=INK)
    models = list(
        dict.fromkeys(
            record["global_args"]["model"].split("/")[-1]
            for record in records
            if record["global_args"].get("model")
        )
    )
    description = f"{benchmark} | {', '.join(models) or 'Model not recorded'}"
    figure.text(0.075, 0.89, fill(description, 95), fontsize=9, color=INK, va="top")
    provider = manifest.get("provider", "Not recorded")
    figure.text(
        0.075,
        0.845,
        f"Provider: {provider} | Saved metrics | Scenario comparisons",
        fontsize=9,
        color=MUTED,
    )
    figure.add_artist(
        mpl.lines.Line2D(
            [0.075, 0.925],
            [0.825, 0.825],
            transform=figure.transFigure,
            color="#DDE5E9",
            linewidth=0.8,
        )
    )
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(
        left=0.105, right=0.94, bottom=0.18, top=0.755, wspace=0.4, hspace=0.65
    )
    figure.add_artist(
        mpl.lines.Line2D(
            [0.075, 0.925],
            [0.115, 0.115],
            transform=figure.transFigure,
            color="#DDE5E9",
            linewidth=0.8,
        )
    )
    notes = [
        "Bars: mean of available successful trials. Dots: trials when n > 1.",
        "Cache rates use query weights; TTFT uses observation counts across engines.",
        "Prompts include cached tokens. Uncached tokens = prefix queries - cache hits.",
        "N/A: missing data. Failed/cancelled trials appear only in summaries.",
    ]
    for index, note in enumerate(notes):
        figure.text(0.075, 0.095 - index * 0.017, note, fontsize=7.2, color=MUTED)
    return figure, axes.flat


def generate(data):
    output = data.root / "reports_v2"
    output.mkdir(exist_ok=True)
    rows = trial_rows(data)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        fields = [key for key in rows[0] if key != "engines"]
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    benchmarks = list(dict.fromkeys(record["benchmark"] for record in data.records))
    scenarios = list(dict.fromkeys(record["scenario"] for record in data.records))
    scenario_colors = {
        name: COLORS[i % len(COLORS)] for i, name in enumerate(scenarios)
    }
    manifest = read_json(data.root / "run.json")
    pages = output / "pages"
    pages.mkdir(exist_ok=True)
    with mpl.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        with PdfPages(
            output / "report.pdf",
            metadata={
                "Title": f"{data.root.name}: benchmark comparisons",
                "Author": "vllm_bench",
            },
        ) as pdf:
            for index, benchmark in enumerate(benchmarks):
                records = [r for r in data.records if r["benchmark"] == benchmark]
                selected = [row for row in rows if row["benchmark"] == benchmark]
                engines = sorted(
                    {engine for row in selected for engine in row["engines"]}
                )
                engine_colors = {
                    name: ENGINE_COLORS[i % len(ENGINE_COLORS)]
                    for i, name in enumerate(engines)
                }
                groups = {
                    scenario: [r for r in selected if r["scenario"] == scenario]
                    for scenario in dict.fromkeys(r["scenario"] for r in selected)
                }
                images = output / "images" / benchmark
                images.mkdir(parents=True, exist_ok=True)
                for chart in CHARTS:
                    figure = Figure(
                        figsize=(max(6, len(groups) * 1.25), 4.6), layout="constrained"
                    )
                    draw_chart(
                        figure.subplots(), chart, groups, scenario_colors, engine_colors
                    )
                    figure.savefig(images / f"{chart[0]}.png", dpi=200)
                    figure.clear()
                for offset, (title, charts) in enumerate(
                    [
                        ("Coordinator performance", CHARTS[:3]),
                        ("Execution and result validity", CHARTS[3:]),
                    ]
                ):
                    number = index * 2 + offset + 1
                    figure, axes = page(
                        data,
                        benchmark,
                        records,
                        title,
                        number,
                        len(benchmarks) * 2,
                        manifest,
                    )
                    for axis, chart in zip(axes, charts):
                        draw_chart(axis, chart, groups, scenario_colors, engine_colors)
                    if offset == 0:
                        figure.delaxes(figure.axes[3])
                    pdf.savefig(figure)
                    # The same page as an image, for documents that cannot
                    # embed a PDF.
                    figure.savefig(pages / f"page_{number:02d}.png", dpi=200)
                    figure.clear()
    print(f"Reports v2: {output}", flush=True)
