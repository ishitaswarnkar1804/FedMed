"""Flower client for FedMed."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from flwr.common import Metrics, NDArrays, Parameters, Scalar

from client.train_local import run_local_round
from src.config import load_config
from src.model.unet3d import build_model, get_model_parameters, set_model_parameters
from src.privacy.tenseal_utils import encrypt_vector, load_public_context


def _arrays_to_parameters(arrays: list[np.ndarray]) -> Parameters:
    return Parameters(tensors=[array.tobytes() for array in arrays], tensor_type="numpy")


def _parameters_to_arrays(parameters: Parameters) -> list[np.ndarray]:
    return [np.frombuffer(tensor, dtype=np.float32) for tensor in parameters.tensors]


class FedMedClient(NumPyClient):
    def __init__(
        self,
        hospital_id: str,
        config: dict[str, Any],
        use_synthetic: bool | None = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.config = config
        self.use_synthetic = use_synthetic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(config).to(self.device)
        self.param_count = len(get_model_parameters(self.model))
        self.privacy_mode = config["privacy"].get("mode", "plain")
        self.chunk_size = config["privacy"].get("chunk_size", 8192)
        self.public_context = None
        if self.privacy_mode == "he_ckks":
            self.public_context = load_public_context(config["privacy"]["keys_dir"])

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return [get_model_parameters(self.model)]

    def set_parameters(self, parameters: NDArrays) -> None:
        set_model_parameters(self.model, parameters[0])

    def fit(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        baseline = get_model_parameters(self.model)
        delta, num_examples, metrics = run_local_round(
            baseline,
            self.config,
            hospital_id=self.hospital_id,
            use_synthetic=self.use_synthetic,
        )

        if self.privacy_mode == "he_ckks":
            assert self.public_context is not None
            encrypted = encrypt_vector(self.public_context, delta, self.chunk_size)
            return [np.frombuffer(encrypted, dtype=np.uint8)], num_examples, {
                "hospital_id": self.hospital_id,
                "val_dice": metrics["val_dice"],
                "val_loss": metrics["val_loss"],
                "encrypted": 1.0,
            }

        return [delta], num_examples, {
            "hospital_id": self.hospital_id,
            "val_dice": metrics["val_dice"],
            "val_loss": metrics["val_loss"],
            "encrypted": 0.0,
        }

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.set_parameters(parameters)
        from src.data.silo_loader import build_loaders
        from src.metrics.segmentation import evaluate_model

        _, val_loader = build_loaders(
            self.config,
            hospital_id=self.hospital_id,
            use_synthetic=self.use_synthetic,
        )
        val_loss, val_dice = evaluate_model(self.model, val_loader, self.device)
        return val_loss, len(val_loader.dataset), {
            "hospital_id": self.hospital_id,
            "val_dice": val_dice,
        }


def create_client(
    hospital_id: str,
    config_path: str | None = None,
    use_synthetic: bool | None = None,
) -> Client:
    config = load_config(config_path)
    return FedMedClient(
        hospital_id=hospital_id,
        config=config,
        use_synthetic=use_synthetic,
    ).to_client()
