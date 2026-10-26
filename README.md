# vllm_bench

Run user-defined vLLM benchmarks against existing service URLs or with services
managed through Docker Compose or Modal. Benchmark
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
vllm_bench run_scenarios --benchmark example --scenario baseline
```

## Commands and selection

A project supplies `.vllm_bench/configs/base.yaml` plus environment configs:
`local_cpu.yaml`, `local_cpu_arm.yaml`, and `modal.yaml`. The default environment
is `local_cpu`; choose another with `--config`. Environment selection is separate
from benchmark and scenario selection.

Both `up` and `run_scenarios` require `--benchmark` and `--scenario`. The plural aliases
`--benchmarks` and `--scenarios` work too. Each accepts one or more names or `'*'`
on its own. Quote the wildcard so your shell does not expand it into filenames.

```bash
# Every scenario for one benchmark:
vllm_bench run_scenarios --config local_cpu --benchmark example --scenarios '*'
# A named scenario across all benchmarks:
vllm_bench run_scenarios --benchmark '*' --scenario baseline
# Every benchmark and its own scenarios:
vllm_bench run_scenarios --benchmark '*' --scenario '*'
# Explicit subsets:
vllm_bench run_scenarios --benchmark example other --scenario baseline distributed
# Start one environment and follow its logs:
vllm_bench up --config local_cpu_arm --benchmark example --scenario baseline
vllm_bench down
vllm_bench report results/local-comparison/run001
```

A named scenario must exist in every selected benchmark. Unknown names fail
before starting containers, stopping an active environment, or saving results.
A scenario wildcard uses each benchmark's own scenarios; they need not match.
`run_scenarios` creates fresh services for every selected benchmark/scenario/repeat and
then generates reports. The former `run` command is now named `run_scenarios`.

`up` must resolve to exactly one benchmark/scenario pair. It only attaches to an
active environment when provider, image, benchmark, scenario, and arguments match.
It streams logs;
Ctrl+C stops services and saves output. Use `up -d` to leave it running, then
`down` to stop it. `down` uses saved state, and `report` uses its result directory,
so neither needs benchmark/scenario selectors or an environment config.

`run_scenarios` stops an existing environment only after validating the complete selection.
Failures retain partial results and continue; Ctrl+C stops the experiment.
Cleanup failures preserve recovery state for `down` and stop further scheduling.
Reports can be regenerated offline.

`--current-dir /path/to/project` works before or after any command and sets the
root for config lookup, scripts, snapshots, runtime state, and results. Relative
report paths also resolve from this root. `--config` works before or after `up`
or `run_scenarios`; it accepts an environment name or YAML path.

## Run against existing services

`run_only` executes exactly one benchmark/scenario once, regardless of `runs`.
It does not start, stop, reconfigure, or restart services, and leaves any active
`up` environment and its state file untouched. It uses the same benchmark executor
and report format as `run_scenarios`. Managed runs use the startup/readiness
routine shared with `up` and the provider shutdown used by `down` after each trial,
including failed trials.

```bash
vllm_bench run_only --config local_cpu \
  --benchmark example --scenario baseline \
  --service-url leader=https://example.com:8000
```

Repeat `--service-url NAME=URL` for each required service. Names match the config's
`services` keys. Supply every service selected by the scenario and benchmark
service dependencies; unknown, duplicate, and missing names fail before execution.
URLs must be HTTP(S) base URLs, including any reverse-proxy prefix. Each supplied
service group describes one external endpoint; `replicas` controls managed services
only. The client uses the supplied URLs for health checks, metrics, and benchmark
requests. `internal=True` returns the same URLs in this mode, so services that call
each other must also be able to reach those addresses.

Only benchmark scripts need to exist locally. An external-only config can omit
`base_image` and service `startup_script`; `up` and `run_scenarios` still require
those fields. Set `global_args` and `service_args` to match the models already
served. Selecting a scenario records its arguments; it does not change the remote
server's coordinator or model. The old `run` command has been renamed to
`run_scenarios`.

Results are saved under `results/external-<name>/runNNN/`, with one trial,
endpoint URLs, the client context, logs, metrics, and reports. External runs do not
upload a project snapshot, transfer inputs, or collect server logs/GPU samples.
When the client and servers share a filesystem, `get_service_path()` returns the
local path unchanged. If the project is mounted at a different path in the services,
pass `--service-project-dir /workspace/project` to translate paths beneath the
local project root. This includes sampled inputs under `results/`. This flag maps
paths only; the directory must already be shared with the services.
For cloud deployments, make inputs available to the servers
first; `--dataset-path /server/path` passes that value as the benchmark's
`dataset_path` argument. Your benchmark adapter decides how to use it.

After health checks pass, `up` prints a copyable `run_only` command before the
endpoint and log details, including the selected config, project, benchmark,
scenario, and service URLs. For local services with a mounted project, it also
includes `--service-project-dir` so sampled inputs resolve inside Docker.
You can start an environment with `up -d`, then run
that command as often as needed before stopping it with `down`. Supply dataset paths
visible to that existing environment when it cannot access the client's paths.

## Configuration

`base.yaml` defines shared services and scenarios. Benchmarks reference scenario names. For example:

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
scenarios:
  baseline:
    args:
      coordinator: single_worker
    services: [leader]
  distributed:
    args:
      coordinator: round_robin
    services: [leader, worker]
benchmarks:
  example:
    script: scripts/benchmark.py
    script_args:
      dataset_percentage: 0.1
    scenarios: [baseline, distributed]
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
Saved runs contain the resolved environment, shared scenario definitions, and
selected benchmarks with their selected scenario references. There are no separate benchmark YAML files or `benchmark_scenarios` field.

Scenarios are defined once in the top-level `scenarios` mapping. Each benchmark
selects a nonempty list of unique scenario names. Every scenario declares a
nonempty list of known service names. A benchmark may also declare `services: [icp]` to add dependencies to every selected scenario. The CLI combines both
service lists, removes overlaps, and expands each service to its replica count.
Scenario and benchmark arguments are arbitrary project data.
Names use lowercase letters, digits, underscores, or hyphens, starting with a
letter. Script paths are relative to the project root.

Optional settings:

- Benchmark `overwrite_global_args`: override global defaults for that benchmark's
  clients, services, and result metadata. Precedence is `global_args`, then
  `overwrite_global_args`, then service `service_args`. Other benchmarks keep
  their own defaults.
- Service `replicas`: number of instances; defaults to one.
- Service `gpus: all` or `gpus: ["0"]`: local NVIDIA GPU allocation. Omit for CPU.
- Service `environment`: extra string environment values.
- Service `service_args`: arguments that override `global_args` for that service.
  Startup scripts use `bench_util.get_service_args()` to read the merged values;
  clients can use `bench_util.get_service_args("cascade")` for a named service.
  Unspecified arguments inherit global values; `get_global_args()` stays unchanged.
- Service `health_path`: defaults to `/health`; omit `metrics_path` if unavailable.
- `skip_volume_mount: true`: use project code already inside the image.
- `image_workdir`: project directory for that mode; defaults to `/workspace/project`.

Startup scripts bind to `0.0.0.0` on their assigned service port and run in the
foreground. Returning stops the service. Images must support the selected
hardware; a CUDA-only vLLM build cannot automatically run on CPU.

## Modal

`run_scenarios` creates a fresh temporary Modal app for each invocation. All
selected trials run within that app; service logs are collected before the app
is stopped on completion, failure, or cancellation. The next invocation creates
a new app, even with the same `modal.app` name. Modal may retain stopped apps in
its history. `up` continues to use a persistent named app for detached services.

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
  volumes:
    /fever: kalypso-fever
```

`modal.volumes` mounts existing Volumes at absolute container paths. Use it
when a service needs data that is too large or too expensive to copy into the
project snapshot. The model cache stays at `/mnt/vllm-bench-cache`; a mount at
that path is rejected. Missing Volumes fail at startup rather than being
created silently.

```bash
vllm_bench run_scenarios --config modal --benchmark example --scenario '*'
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

Use `enviroment` for typed runtime state and `bench_util` for runtime operations.
The public import is spelled `enviroment`. Importing it does not load a context;
fields are read when a script accesses them during a CLI-managed trial.

```python
from vllm_bench import bench_util, enviroment

leader = enviroment.get_service("leader")  # Service, with attribute completion
model = leader.args["model"]
port = leader.port
benchmark = enviroment.benchmark
fraction = enviroment.get_arg("dataset_percentage", 1.0)
coordinator = enviroment.get_arg("coordinator", scope="scenario")
output = enviroment.output_dir
remote_path = bench_util.get_service_path(output / "inputs/data.csv")
```

`Environment` and `Service` are exported for type annotations:

```python
from vllm_bench import Environment, Service, enviroment


def leader_url(environment: Environment) -> str:
    leader: Service = environment.leader
    return leader.url
```

### Environment fields

| Field | Type | Meaning |
| --- | --- | --- |
| `benchmark` | `str` | Selected benchmark configuration name. |
| `scenario` | `str` | Selected scenario configuration name. |
| `repeat` | `int` | One-based repetition number. |
| `sample_interval` | `float` | Metric sampling interval in seconds. |
| `global_args` | `dict[str, JsonValue]` | Global arguments. |
| `scenario_args` | `dict[str, JsonValue]` | Selected scenario arguments. |
| `benchmark_args` | `dict[str, JsonValue]` | Selected benchmark arguments. |
| `project_dir` | `Path` | Project root visible to this script. |
| `service_project_dir` | `Path` | Project root visible to services. |
| `trial_dir` | `Path` | Trial artifact directory. |
| `output_dir` | `Path` | This script's output directory; created on access. |
| `service` | `Service \| None` | Current startup instance; `None` on the client. |
| `services` | `tuple[Service, ...]` | Instances with client-visible addresses. |
| `internal_services` | `tuple[Service, ...]` | Instances with addresses used between services. |
| `leader` | `Service` | The single leader with its client-visible address. |
| `workers` | `tuple[Service, ...]` | Workers with client-visible addresses; empty for a baseline. |
| `external_services` | `bool` | Whether services are managed outside this CLI invocation. |
| `modal_sandbox_id` | `str \| None` | Modal sandbox used for uploads, if any. |

Arguments are project-defined JSON values: strings, numbers, booleans, null,
lists, or dictionaries. Argument dictionaries are fresh copies, so script
changes do not affect the saved context or another service's arguments.

### Environment methods and Service fields

- `get_arg(name, default=None, *, scope="benchmark") -> JsonValue` reads an
  argument. `scope` is `"benchmark"`, `"scenario"`, or `"global"`.
- `get_service(name=None, *, internal=False) -> Service` returns the current
  startup instance, a named service group, or a named replica such as `worker-1`.
  Use `internal=True` for addresses reachable between services. A missing or
  ambiguous service raises `ValueError`. Calling without a name on the client
  also raises `ValueError`; use `enviroment.service` to check for a client.

`Service` has a fixed schema, accessed through attributes:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Unique instance name, such as `worker-1`. |
| `group` | `str` | Service group from YAML, such as `worker`. |
| `role` | `str` | Configured role, such as `leader`, `worker`, or `icp`. |
| `replica` | `int` | One-based replica index. |
| `url` | `str` | Base URL in the selected address space. |
| `port` | `int` | URL port, defaulting to 80 for HTTP or 443 for HTTPS. |
| `health_path` | `str` | Health endpoint relative to the base URL. |
| `metrics_path` | `str \| None` | Prometheus endpoint, or `None` when disabled. |
| `args` | `dict[str, JsonValue]` | Global arguments merged with this group's overrides. |

### Runtime utilities

- `bench_util.get_service_path(local_path)` translates project/trial paths into
  paths readable by services and uploads inputs when needed by Modal.
- `bench_util.prepare_dataset(archive, records, fraction=1.0)` samples a ZIP
  archive into the trial output `inputs/` directory and returns that directory.
  Use a CSV member path to sample rows while preserving its header, or a glob
  such as `articles/*.txt` to sample files in natural numeric order while
  preserving supporting files. It writes counts to `dataset.json`.
- `bench_util.source_environment(package)` overlays project Python sources on
  the installed wheel, retaining native libraries and missing Python files.
- `bench_util.start_engine(command, options)` launches a project-provided vLLM
  command with the overlay, GPU selection and CUDA validation, and serialized
  option flags. Pass `model` in `options`; the launcher receives `MODEL` and
  `VLLM_PYTHON`. Leaders use the last GPU; other replicas rotate across GPUs.
  It writes the resolved command, model, and GPU count to `startup.json`.
- `bench_util.get_gpu_count()`, `is_gpu_available()`, and `is_cpu()` inspect
  hardware using `nvidia-smi` without importing PyTorch.

The existing `bench_util` context getters and `current_run`/`current_benchmark`
remain available for compatibility. New scripts should use the typed
`enviroment` interface for context and service access. The package includes
`py.typed` so editors can discover the public types after installation.

Benchmark latency is measured automatically as the benchmark script's runtime,
excluding service startup and metric collection. Scripts may optionally write
`response.json` in `get_output_dir()`. Arbitrary JSON is preserved; a numeric
`latency_sec` overrides the measured latency, and a `results` list supplies the
output row count. No response file is needed for timing or engine metrics.

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
docker build -t vllm-benchmark-smoke -f tests/fixtures/compose.Dockerfile .
python tests/compose_smoke.py --image vllm-benchmark-smoke
```

The opt-in Compose check launches HTTP stand-ins with Prometheus metrics. It
checks lifecycle, endpoints, mounting, scenarios, collection, and reports without
loading any model or allocating cloud resources.

## Publishing and license

See [PUBLISHING.md](PUBLISHING.md) for GitHub setup, package checks, and PyPI
Trusted Publishing. The project uses the [MIT license](LICENSE).
