"""Benchmark one or more Text-to-SQL checkpoints on BIRD or Spider."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.benchmark_runner import evaluate_model, write_report
from src.evaluation.benchmarks import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("bird", "spider"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Extracted official benchmark directory")
    parser.add_argument("--database-dir", type=Path, help="Directory containing per-database SQLite files")
    parser.add_argument("--split", default="dev", help="Official split name, e.g. train or dev")
    parser.add_argument("--model", action="append", required=True, help="Repeat for each Hugging Face model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N records for a smoke run")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--no-evidence", action="store_true", help="Do not pass BIRD evidence to the model")
    args = parser.parse_args()
    examples = load_benchmark(
        args.benchmark,
        args.data_dir,
        args.split,
        database_dir=args.database_dir,
        limit=args.limit,
    )
    if not examples:
        raise ValueError("The selected benchmark split contains no examples")

    run_config = {
        "benchmark": args.benchmark,
        "split": args.split,
        "data_dir": str(args.data_dir),
        "database_dir": str(args.database_dir) if args.database_dir else None,
        "max_new_tokens": args.max_new_tokens,
        "timeout_seconds": args.timeout_seconds,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "use_evidence": not args.no_evidence,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for model_name in args.model:
        summary, records = evaluate_model(
            model_name,
            examples,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
            dtype=args.dtype,
            quantization=None if args.quantization == "none" else args.quantization,
            use_evidence=not args.no_evidence,
        )
        summary.update({"benchmark": args.benchmark, "split": args.split})
        write_report(args.output_dir, summary, records)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
