"""Quality filters for SQL examples."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def normalize_sql(sql: str) -> str:
    """Normalize harmless whitespace differences for duplicate detection."""
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def filter_records(records: Iterable[dict[str, Any]], max_chars: int = 4096) -> list[dict[str, Any]]:
    """Keep bounded, non-empty SQL records and remove normalized duplicates."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        sql = str(record.get("sql", "")).strip()
        if not sql or len(sql) > max_chars or not re.match(r"^(select|with)\b", sql, re.I):
            continue
        key = normalize_sql(sql)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result
