"""Format SQL examples as ChatML conversations."""

from __future__ import annotations


def format_chatml(schema: str, question: str, sql: str, cot: str | None = None) -> str:
    """Return one training example in a compact ChatML format."""
    answer = f"{cot}\n" if cot else ""
    answer += sql.strip()
    return (
        "<|im_start|>system\n"
        "You generate valid SQLite SQL for the provided schema.\n"
        f"Schema:\n{schema.strip()}\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}\n<|im_end|>"
    )
