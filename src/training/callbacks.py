"""Training callbacks."""

from __future__ import annotations


def build_callbacks():
    """Return optional Transformers callbacks available in the environment."""
    from transformers import EarlyStoppingCallback

    return [EarlyStoppingCallback(early_stopping_patience=3)]
