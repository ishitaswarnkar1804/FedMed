"""TenSEAL homomorphic encryption utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import tenseal as ts
except ImportError:  # pragma: no cover - optional at import time
    ts = None


def ensure_tenseal():
    if ts is None:
        raise ImportError("TenSEAL is required for homomorphic encryption. pip install tenseal")


def create_context(poly_modulus_degree: int = 8192) -> "ts.Context":
    ensure_tenseal()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def save_keys(context: "ts.Context", keys_dir: str | Path) -> None:
    ensure_tenseal()
    keys_path = Path(keys_dir)
    keys_path.mkdir(parents=True, exist_ok=True)

    public_context = context.copy()
    public_context.make_context_public()

    with (keys_path / "public.tenseal").open("wb") as handle:
        handle.write(public_context.serialize(save_secret_key=False))

    with (keys_path / "secret.tenseal").open("wb") as handle:
        handle.write(context.serialize(save_secret_key=True))


def load_public_context(keys_dir: str | Path) -> "ts.Context":
    ensure_tenseal()
    path = Path(keys_dir) / "public.tenseal"
    with path.open("rb") as handle:
        return ts.context_from(handle.read())


def load_secret_context(keys_dir: str | Path) -> "ts.Context":
    ensure_tenseal()
    path = Path(keys_dir) / "secret.tenseal"
    with path.open("rb") as handle:
        return ts.context_from(handle.read())


def chunk_vector(vector: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    chunks = []
    for start in range(0, len(vector), chunk_size):
        chunks.append(vector[start : start + chunk_size].astype(np.float64))
    return chunks


def encrypt_vector(
    context: "ts.Context",
    vector: np.ndarray,
    chunk_size: int,
) -> bytes:
    ensure_tenseal()
    serialized_chunks = []
    for chunk in chunk_vector(vector, chunk_size):
        encrypted_chunk = ts.ckks_vector(context, chunk.tolist())
        serialized_chunks.append(encrypted_chunk.serialize())
    return pickle.dumps(serialized_chunks)


def decrypt_vector(
    secret_context: "ts.Context",
    payload: bytes,
    expected_length: int,
) -> np.ndarray:
    ensure_tenseal()
    serialized_chunks: list[bytes] = pickle.loads(payload)
    values: list[float] = []
    for serialized_chunk in serialized_chunks:
        encrypted_chunk = ts.ckks_vector_from(secret_context, serialized_chunk)
        decrypted = encrypted_chunk.decrypt()
        values.extend(decrypted)
    return np.array(values[:expected_length], dtype=np.float32)


def aggregate_encrypted_payloads(
    payloads: Iterable[bytes],
    context: "ts.Context",
) -> bytes:
    ensure_tenseal()
    payload_list = list(payloads)
    if not payload_list:
        raise ValueError("No encrypted payloads to aggregate.")

    chunk_lists = [
        [
            ts.ckks_vector_from(context, serialized_chunk)
            for serialized_chunk in pickle.loads(payload)
        ]
        for payload in payload_list
    ]
    num_chunks = len(chunk_lists[0])
    aggregated_chunks = []
    for chunk_index in range(num_chunks):
        summed = chunk_lists[0][chunk_index]
        for client_index in range(1, len(chunk_lists)):
            summed = summed + chunk_lists[client_index][chunk_index]
        aggregated_chunks.append(summed)
    return pickle.dumps([chunk.serialize() for chunk in aggregated_chunks])
