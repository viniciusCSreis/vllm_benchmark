"""Typed runtime interface for benchmark and startup scripts.

The exported ``enviroment`` object reads the CLI context lazily, so importing
scripts does not require a running trial. Project-specific arguments are JSON
values; trial identity, paths, and service instances have explicit types.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Arguments = dict[str, JsonValue]
ArgumentScope = Literal["benchmark", "scenario", "global"]


@lru_cache(maxsize=1)
def _context() -> dict[str, JsonValue]:
    return json.loads(Path(os.environ["VLLM_BENCH_CONTEXT"]).read_text())


class Service(BaseModel):
    """One service instance at addresses visible to the caller or other services."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Unique instance name, for example worker-1."""
    group: str
    """Service group from the YAML configuration, for example worker."""
    role: str
    """Configured role, for example leader, worker, or icp."""
    replica: int
    """One-based replica index within the group."""
    url: str
    """HTTP(S) base URL in the requested address space."""
    health_path: str = "/health"
    """Health-check path relative to url."""
    metrics_path: str | None = None
    """Prometheus endpoint path, or None when metrics are disabled."""
    args: Arguments = Field(default_factory=dict)
    """Global arguments merged with this service group's overrides."""

    @property
    def port(self) -> int:
        """Explicit URL port, or the standard HTTP(S) port."""
        parsed = urlsplit(self.url)
        return parsed.port or (443 if parsed.scheme == "https" else 80)


class ScriptContext(BaseModel):
    """The complete context serialized by the CLI for one script invocation."""

    benchmark: str
    scenario: str
    repeat: int
    sample_interval: float
    global_args: Arguments
    scenario_args: Arguments
    benchmark_args: Arguments
    service_args: dict[str, Arguments]
    project_dir: Path
    service_project_dir: Path
    trial_dir: Path
    output_dir: Path
    services: list[Service]
    internal_services: list[Service]
    service: Service | None = None
    external_services: bool = False
    modal_sandbox_id: str | None = None


class Environment:
    """Read-only trial fields and typed service lookup, backed by the CLI context."""

    @property
    def _data(self) -> ScriptContext:
        return ScriptContext.model_validate(_context())

    @property
    def benchmark(self) -> str:
        """Selected benchmark's configuration name."""
        return self._data.benchmark

    @property
    def scenario(self) -> str:
        """Selected scenario's configuration name."""
        return self._data.scenario

    @property
    def repeat(self) -> int:
        """One-based repetition number."""
        return self._data.repeat

    @property
    def sample_interval(self) -> float:
        """Metric sampling interval in seconds."""
        return self._data.sample_interval

    @property
    def global_args(self) -> Arguments:
        """Project-defined global arguments, as a fresh dictionary."""
        return self._data.global_args

    @property
    def scenario_args(self) -> Arguments:
        """Arguments for the selected scenario, as a fresh dictionary."""
        return self._data.scenario_args

    @property
    def benchmark_args(self) -> Arguments:
        """Arguments for the selected benchmark, as a fresh dictionary."""
        return self._data.benchmark_args

    def get_arg(
        self,
        name: str,
        default: JsonValue = None,
        *,
        scope: ArgumentScope = "benchmark",
    ) -> JsonValue:
        """Read a JSON argument from benchmark (default), scenario, or global args."""
        data = self._data
        arguments = {
            "benchmark": data.benchmark_args,
            "scenario": data.scenario_args,
            "global": data.global_args,
        }
        return arguments[scope].get(name, default)

    @property
    def project_dir(self) -> Path:
        """Project root visible to the calling script."""
        return self._data.project_dir

    @property
    def service_project_dir(self) -> Path:
        """Project root visible to the services."""
        return self._data.service_project_dir

    @property
    def trial_dir(self) -> Path:
        """Directory containing this trial's artifacts."""
        return self._data.trial_dir

    @property
    def output_dir(self) -> Path:
        """Calling script's output directory; created on access."""
        path = self._data.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def external_services(self) -> bool:
        """Whether services are managed outside this CLI invocation."""
        return self._data.external_services

    @property
    def modal_sandbox_id(self) -> str | None:
        """Modal sandbox used for uploads, or None on other providers."""
        return self._data.modal_sandbox_id

    def _service(self, service: Service, data: ScriptContext) -> Service:
        return service.model_copy(
            update={"args": {**data.global_args, **data.service_args[service.group]}}
        )

    @property
    def service(self) -> Service | None:
        """Calling startup's instance; None in a client benchmark script."""
        data = self._data
        return self._service(data.service, data) if data.service is not None else None

    @property
    def services(self) -> tuple[Service, ...]:
        """All instances with addresses visible to the calling script."""
        data = self._data
        return tuple(self._service(service, data) for service in data.services)

    @property
    def internal_services(self) -> tuple[Service, ...]:
        """All instances with addresses reachable between services."""
        data = self._data
        return tuple(self._service(service, data) for service in data.internal_services)

    def get_service(
        self, name: str | None = None, *, internal: bool = False
    ) -> Service:
        """Return the current startup or one named group/instance.

        Raise ValueError for a missing or ambiguous name, or when a client asks
        for its current service. Use service to check for a client without raising.
        """
        if name is None:
            service = self.service
            if service is None:
                raise ValueError("Client benchmark scripts have no current service")
            return service
        services = self.internal_services if internal else self.services
        matches = [s for s in services if s.group == name or s.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one service named {name}; found {len(matches)}")
        return matches[0]

    @property
    def leader(self) -> Service:
        """The single leader at its client-visible address."""
        leaders = [service for service in self.services if service.role == "leader"]
        if len(leaders) != 1:
            raise ValueError(f"Expected one leader; found {len(leaders)}")
        return leaders[0]

    @property
    def workers(self) -> tuple[Service, ...]:
        """Worker instances at client-visible addresses; empty for a baseline."""
        return tuple(service for service in self.services if service.role == "worker")


enviroment = Environment()
