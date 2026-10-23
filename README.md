# vllm_bench

Run user-defined vLLM benchmark scenarios through Docker Compose. Benchmark
scripts and the runner execute on your machine; startup scripts execute in
containers. The same Python library supplies arguments and service addresses to
both. Modal will be a separate provider later.

## Install

```bash
python -m pip install -e /path/to/vllm_benchmark
```

Build the same package into your service image. Installing the CLI on the host
alone does not make it available in containers. All image dependencies are
installed during the image build, never by the provider at startup.

See `../multi-worker-kalypso/config.yaml` and its `vllm_benchmark_config/README.md`
for a complete integration with real service and benchmark scripts.

## Commands

From a user project containing `config.yaml`:

```bash
vllm_bench up
vllm_bench down
vllm_bench run scenarios
vllm_bench report results/local-compare_coordinators/run001
```

For a different config, put the global option before the command:

```bash
vllm_bench --config path/to/config.yaml run scenarios
```

`up` starts the first benchmark and scenario and streams container logs during
startup and after health checks pass. Ctrl+C stops the environment and saves its
logs. If an environment is already running, `up` attaches to its logs.

Use `vllm_bench up -d` to start in the background. `down` stops those containers
and saves their logs; it also works when the config has since changed.

`run scenarios` runs every benchmark × scenario × repeat sequentially, starting
fresh containers for every trial. It stops a previous `up` environment first.
Failures save partial files and continue to the next trial; Ctrl+C stops the
experiment. Cleanup failures preserve recovery state for `down` and stop further
scheduling. Reports run after cleanup and can also be regenerated offline.

## Config

```yaml
name: comparison
provider: local
base_image: my-project-benchmark:latest

global_args:
  model: my-model
  max_model_len: 8192

services:
  worker:
    role: worker
    replicas: 2
    startup_script: scripts/start_worker.py
    health_port: 8000
    metrics_path: /metrics
  leader:
    role: leader
    startup_script: scripts/start_leader.py
    health_port: 8000
    metrics_path: /metrics

benchmarks:
  example:
    script: scripts/benchmark.py
    script_args:
      dataset_percentage: 0.01
    services: [leader, worker]

benchmark_scenarios:
  - name: baseline
    args:
      coordinator: single_worker
    disabled_services: [worker]
  - name: distributed
    args:
      coordinator: round_robin

runs: 1
startup_timeout: 1800
benchmark_timeout: 3600
sample_interval: 1
```

Names must use lowercase letters, digits, underscores, or hyphens, starting
with a letter. Scenario and benchmark arguments are arbitrary project data.
The CLI does not know about coordinators or dataset formats.

Script paths and the project root are relative to the config's directory.
Each replica has its own container, published host port, and output directory.
`health_path` defaults to `/health`; omit `metrics_path` for services without
metrics. Startup scripts must bind to `0.0.0.0` at their configured health port.
Scripts run in the foreground; returning from a startup script stops its service.

Optional fields:

- `skip_volume_mount: true`: use code already installed in the image; omit the
  project mount and ZIP. Context and output are still mounted.
- `image_workdir`: image project directory when skipping the project mount;
  defaults to `/workspace/project`.
- Service `gpus: all` or `gpus: ["0"]`: NVIDIA GPU allocation. Omit for CPU.
- Service `environment`: extra string environment values for that service.

The image must support the selected hardware. Exposing GPU devices requires a
compatible Docker host; detecting no GPU does not turn a CUDA-only vLLM build
into a CPU build. Set GPU allocations per service to control sharing.

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

The project is mounted read-only at `/workspace/project`. Each trial directory
is mounted writable at `/run/vllm-bench/trial`; `get_output_dir()` selects the
client trial directory or a replica's service subdirectory. Docker logs and GPU
samples are saved under `services/<group>-<replica>/`.

The generated Compose file, resolved per-service contexts, and provider IDs are
saved under the trial's `.provider/`. The active state is recorded under the
project's `.vllm-bench/`. Host ports bind to `127.0.0.1` and container addresses
use Compose DNS. A named model-cache volume persists after `down`.

By default, each run also saves a ZIP of nonignored project files. Results,
environments, native binaries, caches, `.env` files, and generated wheels are
excluded. The live local project mount reflects subsequent edits; keep files
unchanged during a run when comparing its saved snapshot.

See [RESULT_FORMAT.md](RESULT_FORMAT.md) for the saved contract and reports.

## Development checks

```bash
python -m unittest discover -s tests
python tests/compose_smoke.py --image an-image-containing-vllm-benchmark
```

The opt-in Compose check launches HTTP stand-ins with Prometheus metrics. It
checks lifecycle, endpoints, mounting, scenarios, collection, and reports without
loading any model or allocating cloud resources.
