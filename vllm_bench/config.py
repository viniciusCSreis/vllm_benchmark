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
    startup_script: str
    replicas: int = Field(default=1, gt=0)
    health_port: int = Field(default=8000, gt=0, le=65535)
    health_path: str = "/health"
    metrics_path: str | None = None
    gpus: Literal["all"] | list[str] | None = None
    environment: dict[str, str] = Field(default_factory=dict)


class Benchmark(ConfigModel):
    script: str
    script_args: dict[str, Any] = Field(default_factory=dict)


class Scenario(ConfigModel):
    name: Name
    args: dict[str, Any] = Field(default_factory=dict)
    services: list[Name] = Field(min_length=1)


class BenchmarkScenarios(ConfigModel):
    benchmark: Name
    scenarios: list[Scenario] = Field(min_length=1)


class Config(ConfigModel):
    name: Name
    provider: Literal["local"] = "local"
    base_image: str = Field(min_length=1)
    image_workdir: str = "/workspace/project"
    skip_volume_mount: bool = False
    global_args: dict[str, Any] = Field(default_factory=dict)
    services: dict[Name, Service] = Field(min_length=1)
    benchmarks: dict[Name, Benchmark] = Field(min_length=1)
    benchmark_scenarios: BenchmarkScenarios | None = None
    runs: int = Field(default=1, gt=0)
    startup_timeout: float = Field(default=1800, gt=0)
    benchmark_timeout: float = Field(default=3600, gt=0)
    sample_interval: float = Field(default=1, gt=0)

    @model_validator(mode="after")
    def check_references(self):
        if self.benchmark_scenarios is not None:
            selection = self.benchmark_scenarios
            if selection.benchmark not in self.benchmarks:
                raise ValueError(f"Unknown benchmark: {selection.benchmark}")
            names = [scenario.name for scenario in selection.scenarios]
            if len(names) != len(set(names)):
                raise ValueError("Scenario names must be unique")
            for scenario in selection.scenarios:
                if len(scenario.services) != len(set(scenario.services)):
                    raise ValueError(f"Repeated services in scenario {scenario.name}")
                if missing := set(scenario.services) - self.services.keys():
                    raise ValueError(f"Unknown scenario services: {sorted(missing)}")
        for path in [
            *(service.startup_script for service in self.services.values()),
            *(benchmark.script for benchmark in self.benchmarks.values()),
        ]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError(
                    "Script paths must be relative to the project root"
                )
        return self


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


def load_config(path):
    path = resolve_config(path)
    base_path = path.parent / "base.yaml"
    data = {}
    if path != base_path and base_path.exists():
        data = yaml.safe_load(base_path.read_text())
    data = merge_config(data, yaml.safe_load(path.read_text()))
    return Config.model_validate(data)
