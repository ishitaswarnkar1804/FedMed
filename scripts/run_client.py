#!/usr/bin/env python3
"""Run a single FedMed hospital client."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl

from client.client import create_client
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedMed Flower client")
    parser.add_argument("--hospital-id", required=True, help="hospital_a | hospital_b | hospital_c")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--server", default=None, help="Server address host:port")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic volumes")
    args = parser.parse_args()

    config = load_config(args.config)
    server_address = args.server or f"{config['server']['address']}:{config['server']['port']}"
    if config["server"]["address"] == "0.0.0.0":
        server_address = f"localhost:{config['server']['port']}"

    fl.client.start_numpy_client(
        server_address=server_address,
        client=create_client(
            hospital_id=args.hospital_id,
            config_path=args.config,
            use_synthetic=args.synthetic or config["data"]["synthetic"].get("enabled", True),
        ),
    )


if __name__ == "__main__":
    main()
