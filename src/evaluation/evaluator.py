"""Validation benchmark runner."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable

from .metrics import execution_accuracy


def evaluate(records: Iterable[dict], predict: Callable[[str], str], connection: sqlite3.Connection) -> float:
    """Generate SQL for records and return execution accuracy."""
    records = list(records)
    predictions = [predict(str(record["question"])) for record in records]
    references = [str(record["sql"]) for record in records]
    return execution_accuracy(predictions, references, connection)
