"""Prepare and split JSONL records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import read_jsonl
from src.data.filter import filter_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/raw/examples.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    records = filter_records(read_jsonl(args.input))
    split = max(1, int(len(records) * 0.8))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", records[:split]), ("val", records[split:])):
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
