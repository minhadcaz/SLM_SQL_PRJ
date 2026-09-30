"""Dataset adapters for reproducible SQLite Text-to-SQL benchmarks.

The benchmark data is intentionally supplied by path rather than downloaded at
runtime.  BIRD and Spider are versioned datasets with their own licences; a
local path makes the evaluated split explicit and keeps experiments repeatable.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SQLBenchmarkExample:
    """One question, its reference SQL, and the SQLite database it targets."""

    example_id: str
    database_id: str
    question: str
    reference_sql: str
    database_path: Path
    schema: str
    evidence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def database_schema(database_path: Path) -> str:
    """Return table and view DDL without copying database rows into the prompt."""
    connection = sqlite3.connect(f"file:{database_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
              AND sql IS NOT NULL
            ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
            """
        ).fetchall()
    finally:
        connection.close()
    return "\n\n".join(row[0].strip().rstrip(";") + ";" for row in rows)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark annotations were not found: {path}")
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def _find_database(database_dir: Path, database_id: str) -> Path:
    candidates = (
        database_dir / database_id / f"{database_id}.sqlite",
        database_dir / f"{database_id}.sqlite",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Database for '{database_id}' was not found; expected {expected}")


def _bird_paths(data_dir: Path, split: str, database_dir: Path | None) -> tuple[Path, Path]:
    annotations = data_dir / f"{split}.json"
    if database_dir is not None:
        return annotations, database_dir
    candidates = (data_dir / f"{split}_databases", data_dir / "dev_databases", data_dir / "train_databases")
    for candidate in candidates:
        if candidate.is_dir():
            return annotations, candidate
    raise FileNotFoundError(
        "BIRD databases were not found. Pass --database-dir, or place them under "
        f"{data_dir}/<split>_databases."
    )


def _spider_paths(data_dir: Path, split: str, database_dir: Path | None) -> tuple[Path, Path]:
    annotation_name = "train_spider.json" if split == "train" else f"{split}.json"
    annotations = data_dir / annotation_name
    return annotations, database_dir if database_dir is not None else data_dir / "database"


def load_benchmark(
    benchmark: str,
    data_dir: str | Path,
    split: str,
    database_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[SQLBenchmarkExample]:
    """Load a BIRD or Spider split from its official extracted file layout."""
    root = Path(data_dir)
    databases = Path(database_dir) if database_dir is not None else None
    benchmark = benchmark.lower()
    if benchmark == "bird":
        annotations_path, databases_path = _bird_paths(root, split, databases)
        sql_key = "SQL"
    elif benchmark == "spider":
        annotations_path, databases_path = _spider_paths(root, split, databases)
        sql_key = "query"
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    examples: list[SQLBenchmarkExample] = []
    schemas: dict[Path, str] = {}
    for index, record in enumerate(_read_records(annotations_path)):
        database_id = str(record["db_id"])
        database_path = _find_database(databases_path, database_id)
        if database_path not in schemas:
            schemas[database_path] = database_schema(database_path)
        reference_sql = str(record[sql_key]).strip()
        question = str(record["question"]).strip()
        examples.append(
            SQLBenchmarkExample(
                example_id=str(record.get("question_id", index)),
                database_id=database_id,
                question=question,
                reference_sql=reference_sql,
                database_path=database_path,
                schema=schemas[database_path],
                evidence=str(record.get("evidence", "")).strip(),
                metadata={
                    key: record[key]
                    for key in ("difficulty", "db_id")
                    if key in record
                },
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples
