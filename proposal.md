# vllm_bench proposal

A Python CLI and library for benchmarking a user's vLLM project. The user keeps
their own service startup and benchmark scripts. The CLI provides execution,
scenario scheduling, metrics collection, saved results, and reports.

Start with the `local` provider using Docker Compose. Add Modal next. Keep
RunPod in the provider design, but decide its implementation later.

## What users add to their project

```text
multi-worker-kalypso/
  config.yaml
  scripts/
    start_worker.py
    start_leader.py
    start_icp.py
    benchmark_medec.py
    benchmark_biodex.py
```

Users install the `vllm_bench` package in their client environment and include
the same version in their Docker image's requirements. Python startup and
benchmark scripts can then import `vllm_bench.bench_util`.

The image installs dependencies at build time. Providers select the image and
override its startup command; they do not run pip installs during startup.
The existing image reference is `viniciuscsreis/kalypso-bench:latest`.

## One config describes the experiment

Based on `multi-worker-kalypso/vllm_benchmark_config/config.yaml`, use mappings
for services and keep shared model settings in `global_args`:

```yaml
name: compare_coordinators
provider: local
base_image: viniciuscsreis/kalypso-bench:latest

# Omitted or false: make the current project available to the services.
skip_volume_mount: false

global_args:
  model: Qwen/Qwen2.5-0.5B-Instruct
  max_model_len: 8192
  max_num_seqs: 2
  max_num_batched_tokens: 512
  gpu_memory_utilization: 0.45
  dtype: float16

services:
  worker:
    role: worker
    replicas: 1 # 2 will start 2 workers
    startup_script: scripts/start_worker.py
    health_port: 8000
    health_path: /health
    metrics_path: /metrics

  leader:
    role: leader
    startup_script: scripts/start_leader.py
    health_port: 8000
    health_path: /health
    metrics_path: /metrics

  icp:
    role: icp
    startup_script: scripts/start_icp.py
    health_port: 8000
    health_path: /health

benchmarks:
  medec:
    script: scripts/benchmark_medec.py
    script_args:
      dataset_folder: data/full/medec
      dataset_percentage: 0.01
    services: [leader, worker]

  biodex:
    script: scripts/benchmark_biodex.py
    script_args:
      dataset_folder: data/full/biodex
      dataset_percentage: 0.01
    services: [leader, worker, icp]

benchmark_scenarios:
  - name: baseline
    args:
      coordinator: single_worker
    disabled_services: [worker]
  - name: round_robin
    args:
      coordinator: round_robin
  - name: affinity_hash
    args:
      coordinator: affinity_hash
  - name: cost_aware
    args:
      coordinator: cost_aware

runs: 1
startup_timeout: 1800
benchmark_timeout: 3600
sample_interval: 1
```

Run every benchmark × scenario × repeat. This example produces eight trials.
Scenario arguments such as `coordinator` are user-defined; the CLI does not
maintain a list of Kalypso coordinators. `disabled_services` is a scheduling
field: the baseline starts no workers, and the leader receives an empty worker
URL list. ICP is selected by the benchmark's service list.

The startup script will decide to start vllm with gpu args or not
util will have invo about the inviroment like is_gpu available, is_cpu this help when starting scripts

`dataset_percentage` is a benchmark script argument. Each project decides how
to interpret it; the generic CLI does not assume a CSV or dataset layout.

## Commands and lifecycle

### `vllm_bench up`

Read and validate `config.yaml`, prepare the provider, and start the services
Start independent services together
and wait for their health endpoints. Return when ready, leaving the environment
running for manual use, like `docker compose up -d`.

Save local runtime state containing the selected benchmark/scenario, provider
resource IDs, service addresses, and output paths. Print the leader URL.
Initial implementation uses config order to select this manual environment;
selection flags can follow later.

### `vllm_bench run scenarios`

The runner always runs on the user's machine:

1. Read config and reserve a result directory. Save the effective configuration
   and project snapshot before starting the trials.
2. For each benchmark, scenario, and repeat, ask the provider to start a fresh
   set of selected services with the resolved trial context.
3. Wait for health and start collecting service metrics.
4. Run the benchmark's Python script locally. It obtains the reachable leader
   URL through `bench_util`.
5. Measure the benchmark process duration and save its exit status. Take final
   metrics snapshots, then stop that trial's services and collect their files.
6. After all trials and cleanup, generate reports from the saved data.

`run scenarios` can provision an environment itself. A prior `up` is optional.
If `up` already created an environment, reuse its provider setup while restarting
the trial services. Each repeat starts fresh services so prefix caches and
counters do not carry over between comparisons.

### `vllm_bench down`

Stop the environment recorded by `up` and collect available output. A benchmark
run also cleans up the resources it owns on completion, failure, or Ctrl+C.
Keep partial measurements and logs when a trial fails.

## The Python script interface

Startup scripts and client benchmark scripts import the same lightweight library:

```python
from vllm_bench import bench_util


def main():
    model = bench_util.get_global_arg("model")
    coordinator = bench_util.get_scenario_arg("coordinator")
    run_number = bench_util.current_run
    output_dir = bench_util.get_output_dir()
```

Also provide:

- `get_global_args()`: all shared model settings.
- `get_benchmark_arg(name)`: values from this benchmark's `script_args`.
- `get_leader_url()`: leader address appropriate to the calling environment.
- `get_worker_urls()`: the configured worker addresses, or an empty list.
- `get_service_url(name)`: another configured service, such as ICP.

Pass the resolved trial context as a JSON file, with one environment variable
pointing to it. The library reads this file; scripts do not parse the whole
experiment config or load provider internals. Provider-generated context has
internal addresses for service containers and client-reachable addresses for
benchmark scripts running on the user's machine.

A startup script starts its service in the foreground. The provider owns the
container and its cleanup. Scripts can detect CPU/GPU availability and use
different startup settings without requiring separate YAML files.

Benchmark scripts write their response to `get_output_dir() / "response.json"`.
In client scripts this is a local directory; in startup scripts it is a writable
provider directory that will be collected later. Service outputs are kept in
separate subdirectories to avoid filename collisions.

## Provider boundary and project files

Every provider must start services, expose their addresses to the client, stop
services, and collect logs/output. The shared runner owns scenario iteration,
benchmark execution, timing, metrics, and report generation.

- **Local:** generate a Compose file, create a container per service replica,
  bind-mount the project at `/workspace/project`, mount writable output, and
  publish service ports to the client. Container-to-container addresses use
  Compose service names; the host client uses published ports.
- **Modal:** upload the same project snapshot, start service containers from the
  configured image, provide client-reachable endpoints, and download output
  after shutdown. Endpoint exposure must support long benchmark requests as
  well as health checks and metrics.
- **RunPod:** implement the same interface later; machine allocation and
  connectivity need their own design.

For local execution, a bind mount makes project files available without rebuilding
the image. For cloud execution, upload the saved snapshot. By default, the CLI
includes the current project, excluding results, Git metadata, virtual
environments, caches, and secrets files such as `.env`.

When `skip_volume_mount: true`, rely on project code and startup scripts already
installed in the image. Do not mount/upload the user's project or save a project
ZIP. The small resolved context and writable output are still provided. The
benchmark script continues to execute from the user's local project.

## Results and reports

Use a readable run number to avoid overwriting previous experiments:

```text
results/local-compare_coordinators/run001/
  config.yaml
  project.zip                   # Unless skip_volume_mount is true
  project.sha256                # When a ZIP is saved
  run.json                      # Expected trials and source/image details
  data/
    medec/baseline/run_1/
      result.json               # Status, duration, arguments, and any error
      response.json             # Benchmark script's response
      benchmark.log
      metrics_before.json
      metrics_after.json
      metrics_samples.jsonl
      services/
        leader/                 # Downloaded logs and service output
        worker_group_a-1/
        icp/
  reports/
```

The service directories shown are illustrative: a baseline has no worker
directory, and MedEC has no ICP directory. Each started replica gets its own
service output directory.

Record benchmark wall time separately from latency returned by the benchmark.
Collect existing vLLM metrics per engine: request/token counters, prefix-cache
queries/hits, preemptions, running/waiting requests, KV-cache usage, and available
latency histograms. Save raw timestamped measurements. Record missing metrics
as unavailable and detect counter resets when calculating differences.

ICP can have health checks and logs without exposing Prometheus metrics.
Only scrape services configured with a metrics endpoint. For remote GPU usage,
an optional sampler must run inside the provider environment and save samples
for collection; the client cannot use its own GPU readings for a remote engine.

Reports read saved results only. Reuse the existing runtime, throughput,
prefix-cache, engine activity, preemption, and GPU reports where their input
data is available. Preserve a documented result schema so new graphs do not
depend on provider or runner implementation details.

## First implementation

Build the installable Python package and `bench_util`, config validation,
Compose provider, `up`/`down`, local scenario runner, and file-based reports.
Use the existing CLI branches as references for metrics and reports, and the
Modal branches for image settings and resource cleanup. Keep Kalypso-specific
startup commands and benchmark behavior in the user's scripts.

Modal and RunPod follow after the local lifecycle and script interface work.
