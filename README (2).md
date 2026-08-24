# FedMed — Cross-Silo Federated Learning Engine

**Domain:** Privacy-Preserving Machine Learning (PPML) & Healthcare

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Flower](https://img.shields.io/badge/Flower-FL_Framework-green)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-orange)
![TenSEAL](https://img.shields.io/badge/TenSEAL-Homomorphic_Encryption-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)

> A decentralized deep learning system that trains a brain tumor
> segmentation model across three hospitals — without a single byte of
> patient data ever leaving its source.

## Why This Project Matters

Rare-disease and oncology AI models are chronically data-starved: no single
hospital has enough labeled scans, and HIPAA/GDPR make pooling that data
illegal. FedMed solves this with a full-stack **privacy-preserving ML
pipeline** — not just a federated averaging demo, but a system that layers
**homomorphic encryption**, **differential privacy**, and **secure
transport** on top of federated learning, so the central server *never sees
plaintext weights, let alone raw data*.

**This project demonstrates end-to-end competency across four hard domains
simultaneously:**
- 🧠 **Medical Computer Vision** — 3D U-Net segmentation on volumetric MRI data (MONAI/PyTorch)
- 🌐 **Distributed Systems** — multi-node orchestration, fault tolerance, gRPC+TLS
- 🔐 **Applied Cryptography** — homomorphic encryption (TenSEAL/CKKS) for ciphertext-level aggregation
- 📊 **Full-Stack Delivery** — live React/Recharts dashboard streaming real training metrics via WebSocket

## Problem Statement

Training highly accurate ML models for rare diseases requires massive patient
datasets. Strict data privacy laws (HIPAA/GDPR) prevent hospitals from sharing
raw patient data with a centralized server.

## Use Case

Researchers at three hospitals collaborate to train a brain tumor segmentation
model on MRI scans — without ever pooling raw patient data.

1. A central server sends an untrained PyTorch 3D U-Net model to each hospital node.
2. Each node trains locally on its own private data.
3. Only **encrypted weight updates** (via homomorphic encryption) are sent back.
4. The server aggregates ciphertext updates (Secure Multi-Party Computation)
   to produce a new global model — raw patient data never leaves the hospital.

## Architecture Overview

```
                 ┌─────────────────────┐
                 │   Central Server     │
                 │  (Flower Aggregator) │
                 │  FedAvg on ciphertext│
                 └──────────┬───────────┘
                gRPC + TLS  │  encrypted weights
        ┌────────────────────┼────────────────────┐
        │                    │                     │
 ┌──────▼──────┐      ┌──────▼──────┐      ┌───────▼─────┐
 │ Hospital     │      │ Hospital     │      │ Hospital     │
 │ Node 1       │      │ Node 2       │      │ Node 3       │
 │ (local MRI   │      │ (local MRI   │      │ (local MRI   │
 │  data + 3D   │      │  data + 3D   │      │  data + 3D   │
 │  U-Net train)│      │  U-Net train)│      │  U-Net train)│
 └──────────────┘      └──────────────┘      └──────────────┘
```

## What Makes This Non-Trivial

| Naive Federated Learning | FedMed |
|---|---|
| Server sees raw model weights | Server aggregates **encrypted ciphertext** — never sees plaintext weights |
| Assumes all clients stay online | Tolerates a hospital node **dropping mid-round** without crashing training |
| Plaintext gRPC | **TLS-secured** gRPC channels between server and every node |
| No formal privacy guarantee | **Differential privacy** noise gives a mathematical bound against model-inversion attacks |
| Static offline evaluation | **Live WebSocket-streamed** metrics into a real-time dashboard |

## Key Modules

| Module | Tech | Purpose |
|---|---|---|
| Federated Learning Framework | Flower / PySyft | Orchestrates decentralized training loop |
| Computer Vision Model | PyTorch / MONAI | 3D U-Net for MRI/CT tumor segmentation |
| Privacy & Encryption | TenSEAL | Homomorphic encryption for ciphertext aggregation |
| Secure Transport | gRPC + TLS | Encrypted server↔node communication |
| Training Dashboard | React / Recharts | Live convergence & accuracy monitoring |

## Results (updated as training milestones land)

| Metric | Centralized Baseline | Federated (Plaintext) | Federated (Encrypted + DP) |
|---|---|---|---|
| Dice Score | _TBD — Week 1_ | _TBD — Week 2_ | _TBD — Week 4_ |
| Training Rounds to Converge | — | _TBD_ | _TBD_ |
| Privacy Budget (ε) | N/A | N/A | _TBD_ |

> Goal: show the federated + encrypted model approaches centralized-baseline
> accuracy within an acceptable margin, proving privacy doesn't have to come
> at the cost of performance.

## Project Status

🚧 Actively in development — see the live build log in
[docs/WEEKLY_PLAN.md](docs/WEEKLY_PLAN.md) (daily commits, 4-week roadmap)
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design details.

## Repo Structure

```
fedmed/
├── server/         # Central Flower aggregator + FedAvg/HE strategies
├── clients/         # Simulated hospital nodes
├── models/unet3d/    # 3D U-Net (PyTorch + MONAI)
├── encryption/       # TenSEAL homomorphic encryption + differential privacy
├── grpc/              # Proto definitions + TLS certs
├── dashboard/react-app/ # Live metrics dashboard
├── data/              # Local datasets (gitignored)
├── scripts/            # Setup & utility scripts
└── tests/
```

## Setup

```bash
git clone <repo-url>
cd fedmed
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## License

MIT (see LICENSE)
