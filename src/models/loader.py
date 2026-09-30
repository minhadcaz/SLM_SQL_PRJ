"""Base model loading with optional 4-bit quantization."""

from __future__ import annotations


def load_model(model_name_or_path: str, quantization: str | None = None, **kwargs):
    """Load a Transformers causal LM using the requested precision profile."""
    import torch
    from transformers import AutoModelForCausalLM

    load_kwargs = dict(kwargs)
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization is not None:
        raise ValueError(f"Unsupported quantization: {quantization}")
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
