"""Export helpers for PEFT adapters."""

from __future__ import annotations


def merge_lora(base_model_path: str, adapter_path: str, output_path: str) -> None:
    """Merge a PEFT adapter into its base model and save the result."""
    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, base_model_name_or_path=base_model_path)
    model.merge_and_unload().save_pretrained(output_path)
