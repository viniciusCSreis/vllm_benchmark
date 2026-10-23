"""Stable paths exposed to user scripts in service containers."""

from pathlib import Path

CONTAINER_TRIAL = Path("/run/vllm-bench/trial")
CONTAINER_PROJECT = Path("/workspace/project")
