# Dashboard Guide

The FedMed dashboard visualizes federated training progress in real time.

## Access

- **Docker:** http://localhost:5173
- **Local dev:** `cd dashboard && npm install && npm run dev`

## Data Source

The Flower server strategy writes metrics to `logs/metrics.jsonl` after each fit/evaluate round. The FastAPI service at port 8000 reads this file.

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `GET /metrics/rounds` | All round metrics with per-hospital breakdown |
| `GET /metrics/summary` | Latest global Dice and best score |

## Dashboard Panels

1. **Summary cards** — total rounds, latest global Dice, privacy mode, HE status
2. **Global convergence** — line chart of Dice and loss vs federated round
3. **Per-hospital Dice** — grouped bar chart for hospital_a/b/c

## Polling

The React app polls `/metrics/rounds` every 5 seconds. Start the dashboard before launching Colab clients to observe live updates.

## Example Response

```json
{
  "total_rounds": 3,
  "latest": {
    "round": 3,
    "global_dice": 0.51,
    "privacy_mode": "he_ckks",
    "encrypted": true
  },
  "rounds": [...]
}
```
