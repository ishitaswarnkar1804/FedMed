"""Homomorphic encrypted aggregation helpers."""

from __future__ import annotations

import pickle
from typing import Iterable

import numpy as np
import tenseal as ts

from src.privacy.tenseal_utils import aggregate_encrypted_payloads, decrypt_vector, load_secret_context


def is_encrypted_update(metrics: dict) -> bool:
    return float(metrics.get("encrypted", 0.0)) >= 1.0


def aggregate_encrypted_deltas(
    payloads: Iterable[bytes],
    weights: Iterable[float],
    param_length: int,
    keys_dir: str,
) -> np.ndarray:
    weighted_payloads = []
    secret_context = load_secret_context(keys_dir)
    for payload, weight in zip(payloads, weights):
        serialized_chunks: list[bytes] = pickle.loads(payload)
        scaled_chunks = [
            ts.ckks_vector_from(secret_context, serialized_chunk) * weight
            for serialized_chunk in serialized_chunks
        ]
        weighted_payloads.append(
            pickle.dumps([chunk.serialize() for chunk in scaled_chunks])
        )

    aggregated_payload = aggregate_encrypted_payloads(weighted_payloads, secret_context)
    return decrypt_vector(secret_context, aggregated_payload, param_length)


def decode_encrypted_ndarray(array: np.ndarray) -> bytes:
    return array.tobytes()
