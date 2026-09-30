"""SQLite execution-based metrics."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def execution_accuracy(predictions: list[str], references: list[str], connection: sqlite3.Connection) -> float:
    """Compare query results, treating SQL errors as incorrect predictions."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return 0.0
    correct = 0
    for predicted, reference in zip(predictions, references):
        try:
            predicted_rows = connection.execute(predicted).fetchall()
            reference_rows = connection.execute(reference).fetchall()
        except sqlite3.Error:
            continue
        correct += predicted_rows == reference_rows
    return correct / len(predictions)


def execute_readonly_query(database_path: str | Path, sql: str, timeout_seconds: float) -> list[tuple]:
    """Execute one SELECT/WITH query against a database opened read-only."""
    statement = sql.strip()
    if not statement.lower().startswith(("select", "with")):
        raise sqlite3.OperationalError("Only SELECT and WITH queries are allowed")
    path = Path(database_path).resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    started = time.monotonic()
    connection.set_progress_handler(lambda: time.monotonic() - started > timeout_seconds, 10_000)
    try:
        try:
            return connection.execute(statement).fetchall()
        except sqlite3.OperationalError as error:
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Query exceeded {timeout_seconds} seconds") from error
            raise
    finally:
        connection.close()


def compare_query_results(predicted: list[tuple], reference: list[tuple], reference_sql: str) -> bool:
    """Compare results while ignoring row order when the reference has no ORDER BY."""
    if "order by" in reference_sql.lower():
        return predicted == reference
    return sorted(map(repr, predicted)) == sorted(map(repr, reference))
