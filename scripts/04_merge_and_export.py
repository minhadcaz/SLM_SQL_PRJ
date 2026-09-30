"""Merge a LoRA adapter into a base model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.export import merge_lora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_lora(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
