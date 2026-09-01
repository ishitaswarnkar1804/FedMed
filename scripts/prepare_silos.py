#!/usr/bin/env python3
"""Prepare hospital silo manifests for MSD Task01."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.silo_loader import save_silo_manifests


def _default_patient_ids(count: int = 484) -> list[str]:
    return [f"BRATS_{index:03d}" for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split MSD patients into hospital silos")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--patient-list",
        default=None,
        help="Optional text file with one patient ID per line",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    silos = config["data"]["silos"]
    output_dir = Path(config["data"]["root_dir"]) / "silos"

    if args.patient_list:
        patient_ids = [
            line.strip()
            for line in Path(args.patient_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        patient_ids = _default_patient_ids()

    split = save_silo_manifests(patient_ids, silos, output_dir)
    for silo, ids in split.items():
        print(f"{silo}: {len(ids)} patients -> {output_dir / (silo + '.json')}")


if __name__ == "__main__":
    main()
