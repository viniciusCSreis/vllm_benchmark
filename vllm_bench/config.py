"""Validate the experiment without knowing any project's benchmark arguments."""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$")]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Service(ConfigModel):
    role: str
    startup_script: str | None = None
    replicas: int = Field(default=1, gt=0)
    health_port: int = Field(default=8000, gt=0, le=65535)
    health_path: str = "/health"
    metrics_path: str | None = None
    gpus: Literal["all"] | list[str] | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    service_args: dict[str, Any] = Field(default_factory=dict)


class Scenario(ConfigModel):
    args: dict[str, Any] = Field(default_factory=dict)
    services: list[Name] = Field(min_length=1)


class Benchmark(ConfigModel):
    script: str
    script_args: dict[str, Any] = Field(default_factory=dict)
    overwrite_global_args: dict[str, Any] = Field(default_factory=dict)
    scenarios: list[Name] = Field(min_length=1)
    services: list[Name] = Field(default_factory=list)


class ModalSettings(ConfigModel):
    app: str = "vllm-bench"
    gpu: str | None = None
    cpu: float = Field(default=8, gt=0)
    memory: int = Field(default=16384, gt=0)
    timeout: int = Field(default=14400, gt=0, le=86400)
    cache_volume: str = "vllm-bench-model-cache"


class Config(ConfigModel):
    name: Name
    provider: Literal["local", "modal", "external"] = "local"
    modal: ModalSettings = Field(default_factory=ModalSettings)
    base_image: str | None = Field(default=None, min_length=1)
    image_workdir: str = "/workspace/project"
    skip_volume_mount: bool = False
    global_args: dict[str, Any] = Field(default_factory=dict)
    services: dict[Name, Service] = Field(min_length=1)
    scenarios: dict[Name, Scenario] = Field(min_length=1)
    benchmarks: dict[Name, Benchmark] = Field(min_length=1)
    runs: int = Field(default=1, gt=0)
    startup_timeout: float = Field(default=1800, gt=0)
    benchmark_timeout: float = Field(default=3600, gt=0)
    sample_interval: float = Field(default=1, gt=0)

    @model_validator(mode="after")
    def check_references(self):
        for name, scenario in self.scenarios.items():
            self.check_services(scenario.services, f"scenario {name}")
        for benchmark, settings in self.benchmarks.items():
            names = settings.scenarios
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Scenario names must be unique in benchmark {benchmark}"
                )
            if missing := set(names) - self.scenarios.keys():
                raise ValueError(f"Unknown scenarios in {benchmark}: {sorted(missing)}")
            self.check_services(settings.services, f"benchmark {benchmark}")
        for path in [
            *(service.startup_script for service in self.services.values()),
            *(benchmark.script for benchmark in self.benchmarks.values()),
        ]:
            if path and (Path(path).is_absolute() or ".." in Path(path).parts):
                raise ValueError("Script paths must be relative to the project root")
        return self

    def check_services(self, services, label):
        if len(services) != len(set(services)):
            raise ValueError(f"Repeated services in {label}")
        if missing := set(services) - self.services.keys():
            raise ValueError(f"Unknown services in {label}: {sorted(missing)}")

    def get_scenario(self, benchmark, name):
        """Resolve shared scenario services plus the benchmark's dependencies."""
        settings = self.benchmarks[benchmark]
        if name not in settings.scenarios:
            raise ValueError(f"Unknown scenario for benchmark '{benchmark}': {name}")
        scenario = self.scenarios[name]
        return scenario.model_copy(
            update={
                "services": list(dict.fromkeys(scenario.services + settings.services))
            }
        )

    def get_global_args(self, benchmark):
        """Benchmark overrides replace global defaults; service args apply last."""
        return {**self.global_args, **self.benchmarks[benchmark].overwrite_global_args}


def resolve_config(path):
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    return path.resolve()


def merge_config(base, override):
    """Merge mappings recursively; a child list replaces the base list."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_config(path, seen=None):
    path = resolve_config(path)
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Config inheritance cycle: {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text())
    parent = data.pop("extends", None)
    if parent:
        base = read_config(path.parent / parent, seen)
    else:
        base_path = path.parent / "base.yaml"
        base = (
            read_config(base_path, seen)
            if path != base_path and base_path.exists()
            else {}
        )
    return merge_config(base, data)


def load_config(path):
    return Config.model_validate(read_config(path))


def select_benchmarks(config, benchmarks, scenarios):
    """Resolve the complete selection before starting or stopping infrastructure."""
    for label, names in [("benchmarks", benchmarks), ("scenarios", scenarios)]:
        if "*" in names and names != ["*"]:
            raise ValueError(f"Use '*' alone when selecting all {label}")
        if len(names) != len(set(names)):
            raise ValueError(f"Repeated {label} in selection")
    names = list(config.benchmarks) if benchmarks == ["*"] else benchmarks
    if missing := set(names) - config.benchmarks.keys():
        raise ValueError(f"Unknown benchmarks: {', '.join(sorted(missing))}")
    selected = {}
    for name in names:
        benchmark = config.benchmarks[name]
        available = dict.fromkeys(benchmark.scenarios)
        wanted = list(available) if scenarios == ["*"] else scenarios
        if missing := set(wanted) - available.keys():
            raise ValueError(
                f"Unknown scenarios for benchmark '{name}': "
                + ", ".join(sorted(missing))
            )
        selected[name] = benchmark.model_copy(update={"scenarios": wanted})
    return config.model_copy(update={"benchmarks": selected})
