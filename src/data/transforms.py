"""MONAI transforms for 3D brain tumor patches."""

from __future__ import annotations

from typing import Any

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityd,
    ToTensord,
)


def get_train_transforms(config: dict[str, Any]):
    spatial_size = tuple(config["model"]["spatial_size"])
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transforms(config: dict[str, Any]):
    spatial_size = tuple(config["model"]["spatial_size"])
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
