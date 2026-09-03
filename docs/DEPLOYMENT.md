# Deployment Guide

FedMed uses a hybrid architecture: **Docker Compose locally** for the server stack, **Google Colab** for GPU hospital clients.

## Local Simulation (No Docker)

```bash
cd fedmed
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_he_keys.py
python scripts/prepare_silos.py
python scripts/run_simulation.py --rounds 3 --mode plain
```

## Docker Compose

```bash
docker compose up --build
```

### Services

| Service | Port | Purpose |
|---|---|---|
| flower-server | 8080 | Flower FL server + HE aggregation |
| metrics-api | 8000 | FastAPI metrics for dashboard |
| dashboard | 5173 | React monitoring UI |
| tunnel | — | cloudflared public URL for Colab |
| client-smoke | — | Optional CPU smoke client (`--profile smoke`) |

### Tunnel URL for Colab

After `docker compose up`, inspect tunnel logs:

```bash
docker compose logs tunnel
```

Copy the `https://*.trycloudflare.com` URL into Colab notebooks as `SERVER_URL`.

## Colab Hospital Clients

1. Open [hospital_a.ipynb](file:///c:/Users/HP/Desktop/Matplot/fedmed/notebooks/hospital_a.ipynb), [hospital_b.ipynb](file:///c:/Users/HP/Desktop/Matplot/fedmed/notebooks/hospital_b.ipynb), or [hospital_c.ipynb](file:///c:/Users/HP/Desktop/Matplot/fedmed/notebooks/hospital_c.ipynb) in Google Colab.
2. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
3. Set `HOSPITAL_ID` and `SERVER_URL`
4. Run all cells

Each notebook:

- Installs PyTorch + MONAI
- Downloads MSD data for its silo (or uses synthetic fallback)
- Connects to the Flower server via tunnel
- Runs federated rounds with checkpointing

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SERVER_URL` | — | Tunnel URL or `host:8080` |
| `HOSPITAL_ID` | notebook-specific | `hospital_a`, `hospital_b`, `hospital_c` |
| `PRIVACY_MODE` | from config | `plain` or `he_ckks` |

## Checkpointing

Colab sessions may disconnect. Notebooks save checkpoints to `/content/fedmed_checkpoints/` after each round so training can resume.

## Troubleshooting

| Issue | Fix |
|---|---|
| TenSEAL install fails on Windows | Use Docker (Linux containers) |
| Colab cannot reach server | Verify tunnel URL; server must be running |
| OOM on GPU | Reduce patch size or use `segresnet` in config |
| Round 1 HE warning | Known quirk; stabilizes from round 2 |
