""" Dataset loaders for hospital silos and synthetic smoke tests."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from monai.data import Dataset
from torch.utils.data import DataLoader


def _stable_silo(patient_id: str, silos: list[str]) -> str:
    digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(silos)
    return silos[index]


def split_patients(patient_ids: Iterable[str], silos: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {silo: [] for silo in silos}
    for patient_id in sorted(set(patient_ids)):
        buckets[_stable_silo(patient_id, silos)].append(patient_id)
    return buckets


def save_silo_manifests(
    patient_ids: Iterable[str],
    silos: list[str],
    output_dir: str | Path,
) -> dict[str, list[str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split = split_patients(patient_ids, silos)
    for silo, ids in split.items():
        manifest = output_path / f"{silo}.json"
        manifest.write_text(json.dumps({"patients": ids}, indent=2), encoding="utf-8")
    return split


def load_silo_manifest(manifest_path: str | Path) -> list[str]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return data["patients"]


class SyntheticVolumeDataset(Dataset):
    """Lightweight synthetic 3D MRI-like volumes for local smoke tests."""

    def __init__(
        self,
        num_samples: int,
        volume_size: tuple[int, int, int],
        in_channels: int = 4,
        out_channels: int = 3,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        rng = random.Random(self.rng.randint(0, 10_000_000) + index)
        image = np.zeros((self.in_channels, *self.volume_size), dtype=np.float32)
        # Multi-class labels: whole tumor, tumor core, enhancing tumor
        label = np.zeros((self.out_channels, *self.volume_size), dtype=np.float32)

        center = [rng.randint(20, size - 20) for size in self.volume_size]
        radius_outer = rng.randint(10, 18)  # whole tumor
        radius_mid = max(4, int(radius_outer * 0.65))  # tumor core
        radius_inner = max(2, int(radius_outer * 0.35))  # enhancing tumor
        zz, yy, xx = np.ogrid[: self.volume_size[0], : self.volume_size[1], : self.volume_size[2]]
        dist_sq = (
            (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
        )
        mask_whole = dist_sq <= radius_outer ** 2
        mask_core = dist_sq <= radius_mid ** 2
        mask_enhancing = dist_sq <= radius_inner ** 2

        for channel in range(self.in_channels):
            image[channel] = rng.random() * 0.2
            image[channel][mask_whole] = 0.5 + rng.random() * 0.2
            image[channel][mask_core] = 0.7 + rng.random() * 0.2
            image[channel][mask_enhancing] = 0.85 + rng.random() * 0.15

        # Channel 0: whole tumor, Channel 1: tumor core, Channel 2: enhancing
        label[0][mask_whole] = 1.0
        if self.out_channels > 1:
            label[1][mask_core] = 1.0
        if self.out_channels > 2:
            label[2][mask_enhancing] = 1.0

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "patient_id": f"synthetic_{index:04d}",
        }


def build_synthetic_loaders(config: dict[str, Any], hospital_id: str) -> tuple[DataLoader, DataLoader]:
    synthetic_cfg = config["data"]["synthetic"]
    model_cfg = config["model"]
    seed = config["project"]["random_seed"] + hash(hospital_id) % 1000
    volume_size = tuple(synthetic_cfg.get("volume_size", model_cfg["spatial_size"]))

    train_ds = SyntheticVolumeDataset(
        num_samples=synthetic_cfg.get("num_train", 6),
        volume_size=volume_size,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        seed=seed,
    )
    val_ds = SyntheticVolumeDataset(
        num_samples=synthetic_cfg.get("num_val", 2),
        volume_size=volume_size,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def build_msd_loaders(
    config: dict[str, Any],
    hospital_id: str,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    from monai.apps import DecathlonDataset

    from src.data.transforms import get_train_transforms, get_val_transforms

    root_dir = Path(config["data"]["root_dir"])
    manifest_path = root_dir / "silos" / f"{hospital_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Silo manifest not found: {manifest_path}. Run scripts/prepare_silos.py first."
        )

    allowed_patients = set(load_silo_manifest(manifest_path))
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    train_ds = DecathlonDataset(
        root_dir=str(root_dir),
        task="Task01_BrainTumour",
        section="training",
        transform=train_transform,
        download=download,
        cache_rate=0.0,
        num_workers=config["training"].get("num_workers", 2),
    )
    val_ds = DecathlonDataset(
        root_dir=str(root_dir),
        task="Task01_BrainTumour",
        section="validation",
        transform=val_transform,
        download=False,
        cache_rate=0.0,
        num_workers=config["training"].get("num_workers", 2),
    )

    train_ds = _FilterByPatientDataset(train_ds, allowed_patients)
    val_ds = _FilterByPatientDataset(val_ds, allowed_patients)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
    )
    return train_loader, val_loader


class _FilterByPatientDataset(Dataset):
    def __init__(self, base_dataset, allowed_patients: set[str]):
        self.base_dataset = base_dataset
        self.indices = []
        for index in range(len(base_dataset)):
            sample = base_dataset[index]
            patient_id = _extract_patient_id(sample)
            if patient_id in allowed_patients:
                self.indices.append(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.indices[index]]


def _extract_patient_id(sample: dict[str, Any]) -> str:
    for key in ("patient_id", "image_meta_dict", "label_meta_dict"):
        if key in sample:
            meta = sample[key]
            if isinstance(meta, dict):
                for meta_key in ("filename_or_obj", "patient_id"):
                    if meta_key in meta:
                        value = str(meta[meta_key])
                        return Path(value).stem.split("_")[0]
            return str(meta)
    return "unknown"


def build_loaders(
    config: dict[str, Any],
    hospital_id: str,
    use_synthetic: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    if use_synthetic is None:
        use_synthetic = config["data"].get("synthetic", {}).get("enabled", True)
    if use_synthetic:
        return build_synthetic_loaders(config, hospital_id)
    return build_msd_loaders(config, hospital_id)
