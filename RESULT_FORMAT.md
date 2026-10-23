# Result contract, version 1

Runs are numbered per provider/experiment:
`results/{provider}-{name}/run001/`, `run002/`, and so on.

Each run saves the merged base and benchmark settings as `config.yaml`, `run.json` with expected trials and CLI
version/image reference, and `project.zip` plus SHA256 unless
`skip_volume_mount: true`.

Each `data/<benchmark>/<scenario>/run_<repeat>/` contains:

- `result.json`: identity, `status` (`ok`, `failed`, `cancelled`), `error`,
  `runtime_sec`, optional response `latency_sec`/`output_rows`, resolved shared
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
benchmark script. `latency_sec` comes from its response, when available.

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
runtime, throughput, cache hits, per-engine work, preemptions, engine activity,
and GPU timelines. Comparisons group by benchmark and scenario, with means and
sample standard deviations across successful repeats. Failed trials remain in
tables and may appear in timelines. Reports read saved files only.
