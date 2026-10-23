"""Read versioned result files without starting providers or contacting engines."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..contracts import Trial, TrialResult


@dataclass
class ReportData:
    root: Path
    output: Path
    records: list[dict]

    def directory(self, record):
        return (
            self.root
            / Trial.model_validate(
                {key: record[key] for key in ["benchmark", "scenario", "repeat"]}
            ).directory
        )


def load_data(result_dir):
    root = Path(result_dir)
    manifest = json.loads((root / "run.json").read_text())
    records = []
    for item in manifest["trials"]:
        trial = Trial.model_validate(item)
        path = root / trial.directory / "result.json"
        record = (
            TrialResult.model_validate_json(path.read_text())
            if path.exists()
            else TrialResult(
                **trial.model_dump(), error="Trial not completed; see logs"
            )
        )
        records.append(record.model_dump())
    output = root / "reports"
    output.mkdir(exist_ok=True)
    return ReportData(root, output, records)


def read_samples(path):
    if not path.exists():
        return []
    samples = []
    for line in path.read_text().splitlines():
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


def engine_total(record, key):
    values = [engine.get(key) for engine in record["engines"]]
    return (
        sum(values) if values and all(value is not None for value in values) else None
    )
