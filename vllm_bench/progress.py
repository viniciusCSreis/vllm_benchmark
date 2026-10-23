"""Readable, immediately visible progress for long-running commands."""

from datetime import datetime


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)
