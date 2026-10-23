# vllm_bench

Run user-defined vLLM benchmark scenarios through Docker Compose or Modal. Benchmark
scripts and the runner execute on your machine; startup scripts execute in
containers. The same Python library supplies arguments and service addresses to
both. The provider owns service infrastructure; the runner and reports are shared.

## Install

Requires Python 3.11 or newer on the host and in service images. Local runs
require Docker Compose v2; Modal runs require a Modal account and the optional
SDK. Once published on PyPI:

```bash
python -m pip install vllm-benchmark
# Or, with Modal support:
python -m pip install 'vllm-benchmark[modal]'
```

For development, clone this repository and run `python -m pip install -e '.[dev]'`.
No vLLM installation or sibling repository is required on the host unless your
own benchmark scripts need it. Project service images supply their own runtime
dependencies, such as PyTorch, vLLM, or FAISS.

The local provider mounts the installed `vllm_bench` package read-only into every
service container. Modal copies it into the remote image. Providers override
startup to run a wrapper that launches your configured script, samples GPUs,
and handles shutdown. Images need no CLI installation or special entrypoint;
no packages are installed at service startup. Service scripts can use
`from vllm_bench import bench_util` without an initialization call.

Try the [minimal example](examples/minimal/README.md) with an ordinary Python
image and no model download:

```bash
cd examples/minimal
vllm_bench run --benchmark example --scenario baseline
```

## Commands and selection

A project supplies `.vllm_bench/configs/base.yaml` plus environment configs:
`local_cpu.yaml`, `local_cpu_arm.yaml`, and `modal.yaml`. The default environment
is `local_cpu`; choose another with `--config`. Environment selection is separate
from benchmark and scenario selection.

Both `up` and `run` require `--benchmark` and `--scenario`. The plural aliases
`--benchmarks` and `--scenarios` work too. Each accepts one or more names or `'*'`
on its own. Quote the wildcard so your shell does not expand it into filenames.

```bash
# Every scenario for one benchmark:
vllm_bench run --config local_cpu --benchmark example --scenarios '*'
# A named scenario across all benchmarks:
vllm_bench run --benchmark '*' --scenario baseline
# Every benchmark and its own scenarios:
vllm_bench run --benchmark '*' --scenario '*'
# Explicit subsets:
vllm_bench run --benchmark example other --scenario baseline distributed
# Start one environment and follow its logs:
vllm_bench up --config local_cpu_arm --benchmark example --scenario baseline
vllm_bench down
vllm_bench report results/local-comparison/run001
```

A named scenario must exist in every selected benchmark. Unknown names fail
before starting containers, stopping an active environment, or saving results.
A scenario wildcard uses each benchmark's own scenarios; they need not match.
`run` creates fresh services for every selected benchmark/scenario/repeat and
then generates reports. The old `run scenarios` positional syntax is removed.

`up` must resolve to exactly one benchmark/scenario pair. It only attaches to an
active environment when provider, image, benchmark, scenario, and arguments match.
It streams logs;
Ctrl+C stops services and saves output. Use `up -d` to leave it running, then
`down` to stop it. `down` uses saved state, and `report` uses its result directory,
so neither needs benchmark/scenario selectors or an environment config.

`run` stops an existing environment only after validating the complete selection.
Failures retain partial results and continue; Ctrl+C stops the experiment.
Cleanup failures preserve recovery state for `down` and stop further scheduling.
Reports can be regenerated offline.

`--current-dir /path/to/project` works before or after any command and sets the
root for config lookup, scripts, snapshots, runtime state, and results. Relative
report paths also resolve from this root. `--config` works before or after `up`
or `run`; it accepts an environment name or YAML path.

## Configuration

`base.yaml` defines services and every benchmark's own scenarios. For example:

```yaml
name: comparison
services:
  leader:
    role: leader
    startup_script: scripts/start_leader.py
    health_port: 8000
    metrics_path: /metrics
  worker:
    role: worker
    startup_script: scripts/start_worker.py
    health_port: 8000
    metrics_path: /metrics
benchmarks:
  example:
    script: scripts/benchmark.py
    script_args:
      dataset_percentage: 0.01
    scenarios:
      - name: baseline
        args:
          coordinator: single_worker
        services: [leader]
      - name: distributed
        args:
          coordinator: round_robin
        services: [leader, worker]
runs: 1
startup_timeout: 1800
benchmark_timeout: 3600
sample_interval: 1
```

`local_cpu.yaml` selects the local image and model:

```yaml
provider: local
base_image: my-project:cpu
global_args:
  model: Qwen/Qwen2.5-0.5B-Instruct
```

`local_cpu_arm.yaml` uses the same structure with an ARM64 CPU image.
An environment config automatically merges over its sibling `base.yaml`.
Mappings merge recursively; child lists and scalar values replace base values.
`extends: other_config` can inherit another environment before applying overrides.
Saved runs contain the resolved environment and only the selected benchmarks and
scenarios. There are no separate benchmark YAML files or `benchmark_scenarios` field.

Scenario names are unique within a benchmark, and may repeat across benchmarks.
Every scenario declares a nonempty list of known service names. Only listed
services start, each expanded to its replica count. The CLI does not infer
service dependencies. Scenario and benchmark arguments are arbitrary project data.
Names use lowercase letters, digits, underscores, or hyphens, starting with a
letter. Script paths are relative to the project root.

Optional settings:

- Service `replicas`: number of instances; defaults to one.
- Service `gpus: all` or `gpus: ["0"]`: local NVIDIA GPU allocation. Omit for CPU.
- Service `environment`: extra string environment values.
- Service `health_path`: defaults to `/health`; omit `metrics_path` if unavailable.
- `skip_volume_mount: true`: use project code already inside the image.
- `image_workdir`: project directory for that mode; defaults to `/workspace/project`.

Startup scripts bind to `0.0.0.0` on their assigned service port and run in the
foreground. Returning stops the service. Images must support the selected
hardware; a CUDA-only vLLM build cannot automatically run on CPU.

## Modal

Install `python -m pip install 'vllm-benchmark[modal]'` and authenticate with
`modal setup`. Define `.vllm_bench/configs/modal.yaml`:

```yaml
provider: modal
base_image: your-registry/benchmark:cuda
global_args:
  model: Qwen/Qwen2.5-0.5B-Instruct
modal:
  app: vllm-bench
  gpu: T4:1
  cpu: 8
  memory: 16384
  timeout: 14400
  cache_volume: vllm-bench-model-cache
```

```bash
vllm_bench run --config modal --benchmark example --scenario '*'
vllm_bench up --config modal --benchmark example --scenario baseline
vllm_bench down
```

The registry image contains project dependencies; Modal copies the installed CLI
library into it without installing packages at startup. Each trial uses one
sandbox in the named Modal App. Services share its CPU/GPU allocation and use
separate ports. Service `gpus` settings apply to local Compose; `modal.gpu` sets
the remote allocation. Memory is in MiB, and timeout is the sandbox lifetime in
seconds. The model-cache volume mounts at `/mnt/vllm-bench-cache`; the provider
points `HF_HOME`, `VLLM_CACHE_ROOT`, and `XDG_CACHE_HOME` there. This keeps the
mount separate from cache files already baked into the service image. Pin the registry image digest for reproducible runs: Modal can cache
mutable tags after the registry image changes.

Project snapshots and client-generated dataset files are uploaded. Benchmark
scripts, metrics collection, and reports execute locally using HTTPS tunnels.
Use `bench_util` for addresses, assigned ports, and file paths. Shutdown downloads
service output and logs. If the sandbox exits early, available combined logs are
saved as `modal.log`; service files may no longer be available.

## Script library

```python
from vllm_bench import bench_util

model = bench_util.get_global_arg("model")
options = bench_util.get_global_args()
coordinator = bench_util.get_scenario_arg("coordinator")
fraction = bench_util.get_benchmark_arg("dataset_percentage", 1.0)
repeat = bench_util.current_run
benchmark = bench_util.current_benchmark
output = bench_util.get_output_dir()
```

Other helpers:

- `get_project_dir()` returns the project root in the calling environment.
- `get_service()` returns the calling startup's name, role, replica, and URL;
  it returns `None` in client benchmark scripts.
- `get_leader_url()` and `get_worker_urls()` choose the correct addresses for
  the calling environment. Disabled workers yield an empty list.
- `get_service_url("icp")` returns a named service. For a replicated group, use
  its individual name such as `worker-1`.
- `get_service_url("icp", internal=True)` returns the container address when
  constructing a benchmark payload that the leader will execute.
- `get_service_path(local_path)` translates project/trial paths in a client
  benchmark to paths readable inside the containers.
- `get_gpu_count()`, `is_gpu_available()`, and `is_cpu()` inspect the calling
  environment using `nvidia-smi` without importing PyTorch.

Benchmark scripts write `response.json` in `get_output_dir()`. Arbitrary JSON
is preserved. When present, a numeric `latency_sec` and a `results` list provide
additional summary fields; process runtime is measured independently.

## Files and infrastructure

The CLI library is mounted read-only at `/opt/vllm-bench-lib/vllm_bench`,
independently of the project mount and even with `skip_volume_mount: true`.
Keep the host installation available while containers run. Compose uses
`python3 -m vllm_bench.service` as its startup override and prepends the library
location to the configured service `PYTHONPATH`.

The project is mounted read-only at `/workspace/project`. Each trial directory
is mounted writable at `/run/vllm-bench/trial`; `get_output_dir()` selects the
client trial directory or a replica's service subdirectory. Docker logs and GPU
samples are saved under `services/<group>-<replica>/`.

The generated Compose file, resolved per-service contexts, and provider IDs are
saved under the trial's `.provider/`. The active state is recorded under the
project's `.vllm_bench/state.json`. Host ports bind to `127.0.0.1` and container addresses
use Compose DNS. A named model-cache volume persists after `down`.

By default, each run also saves a ZIP of nonignored project files. Results,
environments, native binaries, caches, `.env` files, and generated wheels are
excluded. The live local project mount reflects subsequent edits; keep files
unchanged during a run when comparing its saved snapshot.

See [RESULT_FORMAT.md](RESULT_FORMAT.md) for the saved contract and reports.

## Development checks

```bash
python -m pip install -e '.[dev]'
ruff check .
python -m unittest discover -s tests
python tests/compose_smoke.py --image python:3.12-slim
```

The opt-in Compose check launches HTTP stand-ins with Prometheus metrics. It
checks lifecycle, endpoints, mounting, scenarios, collection, and reports without
loading any model or allocating cloud resources.

## Publishing and license

See [PUBLISHING.md](PUBLISHING.md) for GitHub setup, package checks, and PyPI
Trusted Publishing. The project uses the [MIT license](LICENSE).
