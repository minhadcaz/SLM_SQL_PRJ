"""LoRA adapter configuration."""

from __future__ import annotations


def create_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Create a PEFT LoRA config for causal language modeling."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
