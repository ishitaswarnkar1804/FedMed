"""Local training loop for one federated round."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.losses import DiceLoss

from src.metrics.segmentation import evaluate_model
from src.model.unet3d import build_model, get_model_parameters, set_model_parameters


def train_one_round(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    scaler = torch.cuda.amp.GradScaler(enabled=config["training"].get("amp", False))

    local_epochs = config["federation"]["local_epochs"]
    max_train_samples = config["training"].get("max_train_samples", 4)
    train_steps = 0

    for _ in range(local_epochs):
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config["training"].get("amp", False)):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_steps += 1
            if train_steps >= max_train_samples:
                break

    val_loss, val_dice = evaluate_model(model, val_loader, device)
    return {
        "train_steps": float(train_steps),
        "val_loss": val_loss,
        "val_dice": val_dice,
    }


def run_local_round(
    parameters: np.ndarray,
    config: dict[str, Any],
    hospital_id: str,
    use_synthetic: bool | None = None,
) -> tuple[np.ndarray, int, dict[str, float]]:
    from src.data.silo_loader import build_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    set_model_parameters(model, parameters)
    baseline = get_model_parameters(model)

    train_loader, val_loader = build_loaders(
        config,
        hospital_id=hospital_id,
        use_synthetic=use_synthetic,
    )
    metrics = train_one_round(model, train_loader, val_loader, config, device)
    updated = get_model_parameters(model)
    delta = updated - baseline
    num_examples = len(train_loader.dataset)
    return delta, num_examples, metrics
