"""Dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records without requiring the datasets package."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object on line {line_number}")
                records.append(record)
    return records


def load_dataset(path: str | Path):
    """Load JSONL into a HuggingFace Dataset when the optional dependency is installed."""
    from datasets import Dataset

    return Dataset.from_list(read_jsonl(path))
