# Result contract, version 1

Runs are numbered per provider/experiment:
`results/{provider}-{name}/run001/`, `run002/`, and so on.

Each run saves the merged base and benchmark settings as `config.yaml`, `run.json` with expected trials and CLI
version/image reference, and `project.zip` plus SHA256 unless
`skip_volume_mount: true`.

Each `data/<benchmark>/<scenario>/run_<repeat>/` contains:

- `result.json`: identity, `status` (`ok`, `failed`, `cancelled`), `error`,
  `runtime_sec`, benchmark `latency_sec`, optional `output_rows`, resolved shared
  and scenario arguments, and engine counter differences/rates.
- `response.json`: optional unmodified benchmark output.
- `benchmark.log`: client benchmark stdout/stderr.
- `endpoints.json`: published addresses for the trial's service replicas.
- `metrics_before.json`, `metrics_after.json`: per-engine `values` and optional
  scrape `error`.
- `metrics_samples.jsonl`: `{elapsed_sec, engines: {replica: {values, error?}}}`.
  Each `values` mapping stores existing Prometheus vLLM sample names, aggregated
  across labels. Time starts at the initial pre-benchmark snapshot.
- `services/<replica>/`: service output, `container.log`, and
  `gpu_samples.jsonl` when the wrapper starts.
- `.provider/`: generated Compose file, resolved script contexts, and provider
  metadata needed to understand/recover the execution.

`runtime_sec` measures the client benchmark process, excluding service startup
and metric setup. It includes Python imports and any dataset preparation in the
benchmark script. For successful trials, `latency_sec` defaults to this measured
runtime. A numeric `latency_sec` in the optional response overrides it.

Engine counter differences cover requests, prompt/generated tokens, prefix-cache
queries/hits, and preemptions. Rates are differences divided by client runtime.
Histogram means are differences in `_sum` divided by differences in `_count`;
units are seconds. Cache hit rate is a fraction. Missing metrics, scrape errors,
and counter resets produce `null`, not zero. Newly created counter labels may
start at zero when the initial scrape succeeded.

Raw vLLM gauges include running/waiting requests and KV-cache usage (a fraction).
Each GPU sample stores elapsed seconds and physical device utilization (%),
framebuffer memory (MiB), temperature (C), and power (W). CPU-only/unavailable
systems record an error and empty device list. GPU samples are measured inside
the service container, not on the client. Shared GPUs may be sampled by multiple
containers; their readings must not be summed as separate devices.

Startup failures may have no metrics. Missing trial results remain failed/not
completed in reports. Partial JSONL samples are retained; incomplete lines are
ignored. A successful benchmark need not produce Prometheus metrics or a response
file; those reports are omitted when no corresponding data exists.

`reports/` contains `summary.json`, `runs.csv`, `engines.csv`, and available PNGs:
runtime, throughput, cache hits, per-engine work, engine activity,
and GPU timelines. Every chart identifies individual trials; no repeats are averaged.
Throughput and prefix-cache hits use counter differences between adjacent snapshots
divided by the sampling interval, per engine. Cache hit rate divides interval hits
by interval queries; intervals without queries are omitted. Missing samples and
counter resets leave gaps. Engine activity shows running/waiting requests and
KV-cache usage over time. Runtime and engine work show individual successful trials. Failed trials remain in
tables and may appear in timelines. Reports read saved files only.

Modal runs keep the same trial and report layout. `.provider/provider.json`
records the sandbox ID, and `.provider/client.json` records public endpoints and
remote path mappings. Service output and logs are downloaded into `services/`
before reports run. If the sandbox exits before collection, `modal.log` preserves
its available entrypoint logs and the trial is marked failed.

Additional per-trial reports include benchmark latency (`latency.png`), interval
means for request latency, inference queue wait, and time to first token
(`request_latency.png`, `queue_wait.png`, `time_to_first_token.png`), cumulative
prompt/generated tokens (`tokens_timeline.png`), peak sampled waiting requests
summed across engines (`queue_peak.png`), and query-weighted cache hit rate
(`cache_hit_rate.png`). Engine work includes request and both token totals.
Histogram timelines divide adjacent sum differences by count differences;
intervals without observations leave gaps. These are observation means, not
percentiles or individual request traces. GPU timelines require valid GPU samples.

`reports_v2/` adds a compact comparison report alongside `reports/`:

- `report.pdf`: two pages per benchmark in one combined PDF, with three
  performance charts and four execution charts.
- `images/<benchmark>/`: seven separate PNG charts, using the same values and
  colors as the PDF.
- `pages/page_NN.png`: each page of `report.pdf` as a 200 dpi PNG, numbered as
  in the PDF, for documents that cannot embed a PDF.
- `summary.json` and `summary.csv`: derived values for every expected trial,
  including failed or cancelled trials. JSON also includes per-engine prompt
  token and completed-request totals for the stacked charts.

The charts cover complete-query latency (`query_latency_sec`), prefix-cache hit
rate (`cache_hit_pct`), mean time to first token (`ttft_mean_sec`), logical prompt
tokens (`prompt_tokens`), uncached prefix-query tokens (`uncached_query_tokens`),
completed internal LLM requests (`requests`), and generated tokens
(`generation_tokens`). Summary token counts are unscaled; chart axes use millions
for prompt/uncached tokens and thousands for generated tokens.

Each bar is the arithmetic mean of available successful trials for that scenario
and benchmark, with individual points when more than one trial is available.
The `n` under each bar is the number of available successful trials for that
metric. Missing data is marked `N/A`, including scenarios with no successful
trials. Cache rates pool hits and queries across engines within each trial.
Time to first token pools histogram sum/count differences from
`metrics_before.json` and `metrics_after.json` across all trial engines; missing
histograms or counter resets leave it unavailable. Uncached prefix-query tokens
are cache queries minus cache hits. Logical prompt tokens include cached tokens.
Query latency uses saved `latency_sec`, including its runtime fallback when no
benchmark response supplied a latency. Reported output rows describe completion
and do not establish result quality.
