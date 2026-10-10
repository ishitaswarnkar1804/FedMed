#!/usr/bin/env python3
"""Run FedMed federated simulation locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl

from client.client import FedMedClient
from server.strategy import FedMedStrategy
from src.config import load_config
from src.model.unet3d import build_model, get_model_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedMed Flower simulation")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--rounds", type=int, default=None, help="Override num rounds")
    parser.add_argument(
        "--mode",
        choices=["plain", "he_ckks"],
        default=None,
        help="Override privacy mode",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.rounds is not None:
        config["federation"]["num_rounds"] = args.rounds
    if args.mode is not None:
        config["privacy"]["mode"] = args.mode

    if config["privacy"]["mode"] == "he_ckks":
        keys_dir = Path(config["privacy"]["keys_dir"])
        if not (keys_dir / "public.tenseal").exists():
            print("HE keys missing. Running generate_he_keys.py ...")
            from scripts.generate_he_keys import main as generate_keys

            generate_keys()

    silos = config["data"]["silos"]
    model = build_model(config)
    param_length = len(get_model_parameters(model))

    def client_fn(context: fl.common.Context):
        partition_id = int(context.node_config["partition-id"])
        hospital_id = silos[partition_id % len(silos)]
        return FedMedClient(
            hospital_id=hospital_id,
            config=config,
            use_synthetic=True,
        ).to_client()

    strategy = FedMedStrategy(
        fraction_fit=config["federation"].get("fraction_fit", 1.0),
        fraction_evaluate=config["federation"].get("fraction_evaluate", 1.0),
        min_fit_clients=len(silos),
        min_evaluate_clients=len(silos),
        min_available_clients=len(silos),
        privacy_mode=config["privacy"]["mode"],
        keys_dir=config["privacy"]["keys_dir"],
        param_length=param_length,
        metrics_log_path=config["metrics"]["log_path"],
    )

    log_path = Path(config["metrics"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(silos),
        config=fl.server.ServerConfig(num_rounds=config["federation"]["num_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )
    print(f"Simulation complete. Metrics written to {log_path}")


if __name__ == "__main__":
    main()
