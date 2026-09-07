"""Custom Flower strategies for FedMed."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
from flwr.common import FitRes, Metrics, NDArrays, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from server.he_aggregator import aggregate_encrypted_deltas, decode_encrypted_ndarray, is_encrypted_update
from src.metrics.logger import append_metric


class FedMedStrategy(FedAvg):
    """FedAvg with optional homomorphic aggregation and metrics logging."""

    def __init__(
        self,
        *args,
        privacy_mode: str = "plain",
        keys_dir: str = "keys/",
        param_length: int = 0,
        metrics_log_path: str = "logs/metrics.jsonl",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.privacy_mode = privacy_mode
        self.keys_dir = keys_dir
        self.param_length = param_length
        self.metrics_log_path = metrics_log_path
        self._global_parameters: NDArrays | None = None

    def initialize_parameters(self, client_manager):
        return None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        # Flower passes (ClientProxy, FitRes) tuples
        encrypted = any(
            is_encrypted_update(fit_res.metrics) for _, fit_res in results
        )
        weights = [fit_res.num_examples for _, fit_res in results]
        total_weight = float(sum(weights))

        if encrypted and self.privacy_mode == "he_ckks":
            payloads = [
                decode_encrypted_ndarray(parameters_to_ndarrays(fit_res.parameters)[0])
                for _, fit_res in results
            ]
            normalized_weights = [weight / total_weight for weight in weights]
            delta = aggregate_encrypted_deltas(
                payloads,
                normalized_weights,
                self.param_length,
                self.keys_dir,
            )
        else:
            # Plaintext FedAvg: weighted average of deltas
            deltas = [
                parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
            ]
            normalized = [weight / total_weight for weight in weights]
            # Each client sends a single flattened delta array
            delta = [
                np.sum(
                    [normalized[i] * deltas[i][j] for i in range(len(deltas))],
                    axis=0,
                )
                for j in range(len(deltas[0]))
            ]

        if self._global_parameters is None:
            self._global_parameters = [np.zeros_like(d) for d in delta]

        self._global_parameters = [
            self._global_parameters[j] + delta[j]
            for j in range(len(delta))
        ]
        aggregated_parameters = ndarrays_to_parameters(self._global_parameters)

        # Collect client-reported metrics for logging
        client_metrics = []
        for _, fit_res in results:
            client_metrics.append(fit_res.metrics)

        metrics = {
            "round": server_round,
            "privacy_mode": self.privacy_mode,
            "encrypted": encrypted,
            "num_clients": len(results),
            "client_metrics": client_metrics,
        }
        self._log_round_metrics(metrics)
        return aggregated_parameters, {"round": server_round}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple],
        failures: list[BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        if not results:
            return None, {}

        weighted_loss = 0.0
        total_examples = 0
        dice_by_hospital = {}

        # Flower passes (ClientProxy, EvaluateRes) tuples
        for _, evaluate_res in results:
            num_examples = evaluate_res.num_examples
            loss = evaluate_res.loss
            metrics = evaluate_res.metrics or {}
            weighted_loss += loss * num_examples
            total_examples += num_examples
            hospital_id = metrics.get("hospital_id", "unknown")
            dice_by_hospital[hospital_id] = metrics.get("val_dice", 0.0)

        avg_loss = weighted_loss / max(total_examples, 1)
        global_dice = float(np.mean(list(dice_by_hospital.values()))) if dice_by_hospital else 0.0

        append_metric(
            self.metrics_log_path,
            {
                "event": "evaluate",
                "round": server_round,
                "global_loss": avg_loss,
                "global_dice": global_dice,
                "hospitals": dice_by_hospital,
                "privacy_mode": self.privacy_mode,
            },
        )
        return avg_loss, {"global_dice": global_dice, "round": server_round}

    def _log_round_metrics(self, metrics: dict) -> None:
        client_metrics = metrics.get("client_metrics", [])
        hospitals = {
            item.get("hospital_id", f"client_{index}"): {
                "val_dice": item.get("val_dice", 0.0),
                "val_loss": item.get("val_loss", 0.0),
            }
            for index, item in enumerate(client_metrics)
        }
        append_metric(
            self.metrics_log_path,
            {
                "event": "fit",
                "round": metrics["round"],
                "privacy_mode": metrics["privacy_mode"],
                "encrypted": metrics["encrypted"],
                "num_clients": metrics["num_clients"],
                "hospitals": hospitals,
            },
        )


HomomorphicFedAvg = FedMedStrategy
