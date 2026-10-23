"""Saved trial identities and results, shared with the offline reports."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Trial(BaseModel):
    benchmark: str
    scenario: str
    repeat: int

    @property
    def directory(self):
        return Path("data") / self.benchmark / self.scenario / f"run_{self.repeat}"


class TrialResult(Trial):
    schema_version: Literal[1] = 1
    status: Literal["ok", "failed", "cancelled"] = "failed"
    runtime_sec: float | None = None
    latency_sec: float | None = None
    output_rows: int | None = None
    error: str | None = None
    global_args: dict[str, Any] = Field(default_factory=dict)
    scenario_args: dict[str, Any] = Field(default_factory=dict)
    engines: list[dict[str, Any]] = Field(default_factory=list)

    def save(self, directory):
        (directory / "result.json").write_text(self.model_dump_json(indent=2))
