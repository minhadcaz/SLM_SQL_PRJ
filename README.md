# SLM SQL Train

Training scaffold for a small language model that generates valid SQLite SQL.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_prepare_data.py
pytest -q
```

Use `configs/lora_sft_1.5b.yaml` for a 16-bit LoRA run or `configs/qlora_4bit_t4.yaml` for a Colab T4 NF4 run. GPU-only dependencies are listed separately in `requirements-cuda.txt`.

## Base-model benchmark

`scripts/03_evaluate.py` evaluates Hugging Face checkpoints against the SQLite databases supplied with an extracted official BIRD or Spider release. It uses each checkpoint's chat template, executes the generated and reference queries read-only, then writes per-question predictions plus aggregate execution accuracy (EX).

Run a small smoke benchmark first:

```bash
python scripts/03_evaluate.py \
  --benchmark bird \
  --data-dir /path/to/bird \
  --split dev \
  --limit 20 \
  --model cycloneboy/SLM-SQL-Base-0.5B \
  --model cycloneboy/SLM-SQL-Base-1.5B
```

For BIRD, the runner expects `<data-dir>/<split>.json` and a database directory such as `<data-dir>/dev_databases`; use `--database-dir` when your extracted layout differs. For Spider, it expects `train_spider.json` for `--split train`, `dev.json` for `--split dev`, and `<data-dir>/database` by default. Use each benchmark's development split for the main generalization result; training splits are useful for diagnosing memorization and data fit.

Results are written to `outputs/benchmarks/`: `*.summary.json` has EX, latency, and outcome counts, while `*.predictions.jsonl` contains every prompt output and SQL execution outcome. Add `--quantization nf4` for 4-bit evaluation after installing `requirements-cuda.txt`.

The package is split into data preparation, model loading, training, evaluation, and export utilities under `src/`. Large datasets, checkpoints, cache files, and secrets are excluded by `.gitignore`; keep only small examples in `data/`.
# SLM_SQL_PRJ
