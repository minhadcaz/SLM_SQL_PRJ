"""Model generation and execution-accuracy evaluation for Text-to-SQL."""

from __future__ import annotations

import gc
import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmarks import SQLBenchmarkExample
from .metrics import compare_query_results, execute_readonly_query
from src.models.loader import load_model


SYSTEM_PROMPT = """You generate one valid SQLite query from a database schema and a question.
Return SQL only. Do not use Markdown or explanation."""


@dataclass(frozen=True)
class Generation:
    raw_text: str
    sql: str
    latency_seconds: float


def extract_sql(text: str) -> str:
    """Extract one read-only SQL statement from common chat-model output forms."""
    answer = text.strip()
    if "<answer>" in answer:
        answer = answer.split("<answer>", 1)[1]
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1]
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", answer, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        answer = fenced.group(1)
    match = re.search(r"\b(?:select|with)\b", answer, flags=re.IGNORECASE)
    if not match:
        return ""
    sql = answer[match.start() :].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    return sql.strip()


class TransformersSQLGenerator:
    """Deterministic chat generation from a local or Hugging Face checkpoint."""

    def __init__(self, model_name: str, dtype: str = "auto", quantization: str | None = None) -> None:
        import torch
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = load_model(model_name, quantization=quantization, **model_kwargs)
        self.model.eval()

    def generate(self, example: SQLBenchmarkExample, max_new_tokens: int, use_evidence: bool) -> Generation:
        import torch

        evidence = f"\nEvidence:\n{example.evidence}" if use_evidence and example.evidence else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Schema:\n{example.schema}\n\nQuestion:\n{example.question}{evidence}",
            },
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency_seconds = time.perf_counter() - started
        generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return Generation(raw_text=raw_text, sql=extract_sql(raw_text), latency_seconds=latency_seconds)

    def close(self) -> None:
        del self.model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def evaluate_model(
    model_name: str,
    examples: list[SQLBenchmarkExample],
    max_new_tokens: int,
    timeout_seconds: float,
    dtype: str = "auto",
    quantization: str | None = None,
    use_evidence: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate once per example and return a reproducible per-example report."""
    generator = TransformersSQLGenerator(model_name, dtype=dtype, quantization=quantization)
    records: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    matched = 0
    valid_reference = 0
    try:
        for example in examples:
            generation = generator.generate(example, max_new_tokens=max_new_tokens, use_evidence=use_evidence)
            outcome = ""
            correct = False
            try:
                reference = execute_readonly_query(
                    example.database_path, example.reference_sql, timeout_seconds
                )
                valid_reference += 1
            except TimeoutError:
                outcome = "reference_timeout"
            except sqlite3.Error as error:
                outcome = f"reference_sqlite_error:{type(error).__name__}"
            else:
                if not generation.sql:
                    outcome = "no_sql_found"
                else:
                    try:
                        prediction = execute_readonly_query(
                            example.database_path, generation.sql, timeout_seconds
                        )
                        correct = compare_query_results(prediction, reference, example.reference_sql)
                        outcome = "correct" if correct else "wrong_result"
                    except TimeoutError:
                        outcome = "prediction_timeout"
                    except sqlite3.Error as error:
                        outcome = f"prediction_sqlite_error:{type(error).__name__}"
            errors[outcome] += 1
            matched += int(correct)
            records.append(
                {
                    "model": model_name,
                    "example_id": example.example_id,
                    "database_id": example.database_id,
                    "question": example.question,
                    "reference_sql": example.reference_sql,
                    "prediction_sql": generation.sql,
                    "raw_prediction": generation.raw_text,
                    "execution_match": correct,
                    "outcome": outcome,
                    "generation_latency_seconds": generation.latency_seconds,
                    "metadata": example.metadata,
                }
            )
    finally:
        generator.close()

    total = len(examples)
    return (
        {
            "model": model_name,
            "samples": total,
            "execution_matches": matched,
            "execution_accuracy": matched / total if total else 0.0,
            "valid_reference_queries": valid_reference,
            "mean_generation_latency_seconds": (
                sum(record["generation_latency_seconds"] for record in records) / total if total else 0.0
            ),
            "outcomes": dict(sorted(errors.items())),
        },
        records,
    )


def write_report(output_dir: str | Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Write machine-readable aggregate and per-example benchmark artifacts."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_name = "_".join(
        str(summary.get(key, "")) for key in ("benchmark", "split", "model") if summary.get(key)
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", report_name)
    (directory / f"{safe_name}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / f"{safe_name}.predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
