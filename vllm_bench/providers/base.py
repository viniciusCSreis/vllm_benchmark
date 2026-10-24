"""Infrastructure only; scenario scheduling and metrics belong to the runner."""

from abc import ABC, abstractmethod
from contextlib import contextmanager


class Provider(ABC):
    @contextmanager
    def run_session(self):
        """Own resources shared by trials in one run_scenarios invocation."""
        yield

    @abstractmethod
    def start(self, trial, directory):
        """Start one trial and return client context and reachable service endpoints."""

    @abstractmethod
    def check_running(self):
        """Raise when a required service has exited."""

    @abstractmethod
    def follow_logs(self):
        """Return a context manager yielding a process that streams service logs."""

    @abstractmethod
    def stop(self):
        """Stop owned services and collect logs and output, including partial files."""
