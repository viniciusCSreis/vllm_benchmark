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
    services: list[str] = Field(min_length=1)


class Scenario(ConfigModel):
    name: Name
    args: dict[str, Any] = Field(default_factory=dict)
    disabled_services: list[str] = Field(default_factory=list)


class Config(ConfigModel):
    name: Name
    provider: Literal["local"] = "local"
    base_image: str = Field(min_length=1)
    image_workdir: str = "/workspace/project"
    skip_volume_mount: bool = False
    global_args: dict[str, Any] = Field(default_factory=dict)
    services: dict[Name, Service] = Field(min_length=1)
    benchmarks: dict[Name, Benchmark] = Field(min_length=1)
    benchmark_scenarios: list[Scenario] = Field(min_length=1)
    runs: int = Field(default=1, gt=0)
    startup_timeout: float = Field(default=1800, gt=0)
    benchmark_timeout: float = Field(default=3600, gt=0)
    sample_interval: float = Field(default=1, gt=0)

    @model_validator(mode="after")
    def check_references(self):
        names = [scenario.name for scenario in self.benchmark_scenarios]
        if len(names) != len(set(names)):
            raise ValueError("Scenario names must be unique")
        for benchmark in self.benchmarks.values():
            if len(benchmark.services) != len(set(benchmark.services)):
                raise ValueError("Benchmark services must not be repeated")
            if missing := set(benchmark.services) - self.services.keys():
                raise ValueError(f"Unknown benchmark services: {sorted(missing)}")
            for scenario in self.benchmark_scenarios:
                if not set(benchmark.services) - set(scenario.disabled_services):
                    raise ValueError(f"Scenario {scenario.name} disables all services")
        for scenario in self.benchmark_scenarios:
            if missing := set(scenario.disabled_services) - self.services.keys():
                raise ValueError(f"Unknown disabled services: {sorted(missing)}")
        for path in [
            *(service.startup_script for service in self.services.values()),
            *(benchmark.script for benchmark in self.benchmarks.values()),
        ]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError(
                    "Script paths must be relative to the config directory"
                )
        return self


def load_config(path):
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))
