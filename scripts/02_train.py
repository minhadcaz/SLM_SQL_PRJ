"""Launch the training pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logger import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    configure_logging()
    print(f"Training configuration selected: {args.config}")
    print("Model loading and tokenization should be wired here for the target checkpoint.")


if __name__ == "__main__":
    main()
