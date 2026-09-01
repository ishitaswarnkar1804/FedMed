#!/usr/bin/env python3
"""Generate TenSEAL CKKS keys for FedMed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.privacy.tenseal_utils import create_context, save_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TenSEAL CKKS keys")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    keys_dir = config["privacy"]["keys_dir"]
    context = create_context()
    save_keys(context, keys_dir)
    print(f"Saved public and secret TenSEAL contexts to {keys_dir}")


if __name__ == "__main__":
    main()
