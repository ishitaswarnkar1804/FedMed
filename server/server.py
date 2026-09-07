"""Flower server entrypoint for FedMed."""

from __future__ import annotations

import argparse

import flwr as fl
import numpy as np

from server.strategy import FedMedStrategy
from src.config import load_config
from src.model.unet3d import build_model, get_model_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedMed Flower server")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config)
    param_length = len(get_model_parameters(model))

    strategy = FedMedStrategy(
        fraction_fit=config["federation"].get("fraction_fit", 1.0),
        fraction_evaluate=config["federation"].get("fraction_evaluate", 1.0),
        min_fit_clients=config["federation"]["min_clients"],
        min_evaluate_clients=config["federation"]["min_clients"],
        min_available_clients=config["federation"]["min_clients"],
        privacy_mode=config["privacy"].get("mode", "plain"),
        keys_dir=config["privacy"]["keys_dir"],
        param_length=param_length,
        metrics_log_path=config["metrics"]["log_path"],
    )

    address = config["server"]["address"]
    port = config["server"]["port"]
    fl.server.start_server(
        server_address=f"{address}:{port}",
        config=fl.server.ServerConfig(num_rounds=config["federation"]["num_rounds"]),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
