# Project conventions

- Keep data examples small and never commit checkpoints or secrets.
- Run `pytest -q`, `ruff check .`, and `black --check .` before submitting changes.
- Keep model and CUDA imports lazy so data utilities remain usable on CPU-only environments.
