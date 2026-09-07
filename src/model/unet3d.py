"""3D segmentation model factory using MONAI."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, UNet


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    architecture = model_cfg.get("architecture", "unet3d").lower()

    if architecture == "segresnet":
        return SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            dropout_prob=0.2,
        )

    channels = tuple(model_cfg.get("channels", [16, 32, 64, 128]))
    strides = tuple(model_cfg.get("strides", [2, 2, 2]))
    return UNet(
        spatial_dims=3,
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=channels,
        strides=strides,
        num_res_units=model_cfg.get("num_res_units", 2),
    )


def get_model_parameters(model: nn.Module) -> np.ndarray:
    params = [value.detach().cpu().numpy().ravel() for value in model.parameters()]
    return np.concatenate(params).astype(np.float32)


def set_model_parameters(model: nn.Module, parameters: np.ndarray) -> None:
    offset = 0
    for param in model.parameters():
        size = param.numel()
        chunk = parameters[offset : offset + size]
        param.data = torch.from_numpy(chunk.reshape(param.shape)).to(param.device)
        offset += size


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
