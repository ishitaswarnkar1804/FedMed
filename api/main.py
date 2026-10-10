"""FastAPI metrics service for FedMed dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from src.metrics.logger import read_metrics

app = FastAPI(title="FedMed Metrics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_log_path() -> Path:
    config = load_config()
    return resolve_path(config["metrics"]["log_path"], config)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics/rounds")
def metrics_rounds() -> dict[str, Any]:
    records = read_metrics(_get_log_path())
    fit_records = [record for record in records if record.get("event") == "fit"]
    eval_records = [record for record in records if record.get("event") == "evaluate"]

    rounds = []
    for fit in fit_records:
        round_id = fit.get("round", 0)
        evaluation = next((item for item in eval_records if item.get("round") == round_id), {})
        hospitals = evaluation.get("hospitals") or fit.get("hospitals") or {}
        rounds.append(
            {
                "round": round_id,
                "global_dice": evaluation.get("global_dice", 0.0),
                "global_loss": evaluation.get("global_loss", 0.0),
                "privacy_mode": fit.get("privacy_mode", "plain"),
                "encrypted": fit.get("encrypted", False),
                "hospitals": hospitals,
            }
        )

    latest = rounds[-1] if rounds else {}
    return {
        "config_path": str(DEFAULT_CONFIG_PATH),
        "total_rounds": len(rounds),
        "latest": latest,
        "rounds": rounds,
    }


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    data = metrics_rounds()
    if not data["rounds"]:
        return {"message": "No metrics yet", "rounds": []}

    dice_series = [round_data["global_dice"] for round_data in data["rounds"]]
    return {
        "total_rounds": data["total_rounds"],
        "best_dice": max(dice_series),
        "latest_dice": dice_series[-1],
        "privacy_mode": data["latest"].get("privacy_mode", "plain"),
    }
