"""Training orchestration entrypoint."""

from __future__ import annotations


def train(model, tokenizer, train_dataset, eval_dataset, output_dir: str, **kwargs):
    """Run a minimal supervised fine-tuning job with Transformers Trainer."""
    from transformers import Trainer, TrainingArguments

    arguments = TrainingArguments(output_dir=output_dir, **kwargs)
    trainer = Trainer(
        model=model,
        args=arguments,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer.train()
