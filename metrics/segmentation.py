"""Segmentation metrics for FedMed."""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def build_loss_function() -> DiceLoss:
    return DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)


def build_dice_metric() -> DiceMetric:
    return DiceMetric(include_background=True, reduction="mean")


def post_pred_transform():
    return AsDiscrete(argmax=False, threshold=0.5)


def evaluate_model(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_fn = build_loss_function()
    dice_metric = build_dice_metric()
    post_pred = post_pred_transform()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += float(loss.item())
            steps += 1

            preds = [post_pred(output) for output in outputs]
            dice_metric(y_pred=preds, y=labels)

    mean_loss = total_loss / max(steps, 1)
    dice_values = dice_metric.aggregate()
    mean_dice = float(dice_values.item()) if dice_values is not None else 0.0
    dice_metric.reset()
    return mean_loss, mean_dice
