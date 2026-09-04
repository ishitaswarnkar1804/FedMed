# FedMed — Cross-Silo Federated Learning Engine

**Domain:** Privacy-Preserving Machine Learning (PPML) & Healthcare AI

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Flower](https://img.shields.io/badge/Flower-FL_Framework-green)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-orange)
![TenSEAL](https://img.shields.io/badge/TenSEAL-Homomorphic_Encryption-purple)
![gRPC](https://img.shields.io/badge/gRPC-TLS_Secured-lightblue)
![React](https://img.shields.io/badge/React-Dashboard-61DAFB)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)

> A decentralized deep learning system that trains a brain tumor
> segmentation model across three simulated hospitals — without a single
> byte of patient data, or even a single plaintext model weight, ever
> leaving its source.

---

## Table of Contents

- [Why This Project Matters](#why-this-project-matters)
- [The Core Idea, In Plain English](#the-core-idea-in-plain-english)
- [Problem Statement](#problem-statement)
- [Use Case Walkthrough](#use-case-walkthrough)
- [Architecture Overview](#architecture-overview)
- [What Makes This Non-Trivial](#what-makes-this-non-trivial)
- [Key Modules](#key-modules)
- [Tech Stack Rationale](#tech-stack-rationale-why-these-tools)
- [Results](#results-updated-as-training-milestones-land)
- [Project Status & Roadmap](#project-status--roadmap)
- [What We're Building Next](#what-were-building-next)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [License](#license)

---

## Why This Project Matters

Rare-disease and oncology AI models are chronically data-starved. A single
hospital might see only a handful of cases of a given rare brain tumor
subtype per year — nowhere near enough to train a reliable deep learning
model. The obvious fix, pooling data from multiple hospitals into one
central dataset, is illegal in most jurisdictions: HIPAA in the US, GDPR in
the EU, and equivalent regulations elsewhere explicitly forbid moving
identifiable patient imaging data across institutional or national
boundaries without consent that is, in practice, almost impossible to
obtain at scale.

**Federated Learning (FL)** solves half of this problem: instead of moving
data to the model, it moves the model to the data. Each hospital trains
locally and only shares *model updates*, not patient records. But naive FL
is not actually private — research has repeatedly shown that raw gradient
updates can leak information about the underlying training data through
**model inversion** and **membership inference attacks**. A curious or
compromised central server can, in principle, reconstruct approximations of
patient scans just by inspecting the weight updates it receives.

FedMed is built to close that gap. It doesn't stop at "federated" — it adds
**homomorphic encryption** so the server mathematically cannot read
individual updates even if it wanted to, and **differential privacy** so
that even the final aggregated model carries a formal, provable bound on
how much any single patient's data could have influenced it. The result is
a system that is federated **and** cryptographically private **and**
statistically private — the actual bar that real healthcare deployments
have to clear.

**This project demonstrates end-to-end competency across four genuinely
hard, independently deep domains, combined into one working system:**

- 🧠 **Medical Computer Vision** — a 3D U-Net trained to segment tumor
  volumes out of volumetric MRI scans (MONAI on top of PyTorch)
- 🌐 **Distributed Systems Engineering** — multi-node orchestration, client
  fault tolerance, and secure networking between independent machines
  (Flower, gRPC, TLS)
- 🔐 **Applied Cryptography** — real homomorphic encryption (not a toy
  XOR cipher) used to perform arithmetic aggregation directly on
  ciphertext (TenSEAL / CKKS scheme), plus differential privacy for a
  formal statistical guarantee
- 📊 **Full-Stack Product Delivery** — a live, real-time dashboard that
  streams actual training telemetry over WebSockets into a React UI, not
  just static matplotlib plots after the fact

## The Core Idea, In Plain English

Imagine three hospitals each have a locked room full of MRI scans they're
not allowed to let anyone else see — not even a peek. They all want a
shared, smarter tumor-detection model, but none of them can mail their
scans to each other or to a middleman.

So instead, a "recipe" (the untrained model) gets mailed to each hospital.
Each hospital cooks using its own private ingredients (its own scans) and
mails back only the *change in flavor* (the weight update) — not the
ingredients themselves. Before mailing that update back, each hospital also
locks it in a special box that can only be combined with other locked boxes
(homomorphic encryption) — the person combining them (the central server)
never gets to open any individual box, only look at the combined result
after all three are mixed together. And just to be extra safe, each
hospital also shakes a little statistical "fog" (differential privacy
noise) over their update before sending it, so that even the final combined
recipe can't be reverse-engineered to guess exactly what any one hospital's
ingredients were.

That's FedMed: **collaborative learning without ever exposing what's being
learned from.**

## Problem Statement

Training highly accurate ML models for rare diseases requires massive
patient datasets. However, strict data privacy laws (HIPAA/GDPR) prevent
hospitals from sharing raw patient data with a centralized server.

## Use Case Walkthrough

Researchers at three global hospitals collaborate to train a brain tumor
segmentation model using MRI scans, without ever pooling their private
data.

1. **Model distribution** — a central server sends an untrained PyTorch 3D
   U-Net model to each hospital node.
2. **Local training** — each node trains the model locally, entirely on its
   own private data; nothing leaves the hospital's infrastructure at this
   stage.
3. **Encrypted upload** — each node encrypts its resulting weight *deltas*
   using homomorphic encryption (and layers on differential-privacy noise)
   before transmitting them over a TLS-secured channel.
4. **Blind aggregation** — the central server combines the encrypted
   updates from all three hospitals using Secure Multi-Party Computation
   principles — it performs real arithmetic (summation/averaging) directly
   on the ciphertext, without ever decrypting an individual hospital's
   contribution.
5. **Global model update** — only the *final aggregate* is decrypted,
   producing a new, improved global model that every hospital benefits
   from, while patient-level data and even individual institutional
   contributions stay mathematically hidden.
6. **Repeat** — this cycle runs for many rounds until the global model's
   accuracy converges toward (ideally matches) what a centralized model
   trained on all the pooled data would have achieved.

## Architecture Overview

```
                 ┌─────────────────────┐
                 │   Central Server     │
                 │  (Flower Aggregator) │
                 │  FedAvg on ciphertext│
                 │  + WebSocket metrics │
                 └──────────┬───────────┘
                gRPC + TLS  │  encrypted, noised weight deltas
        ┌────────────────────┼────────────────────┐
        │                    │                     │
 ┌──────▼──────┐      ┌──────▼──────┐      ┌───────▼─────┐
 │ Hospital     │      │ Hospital     │      │ Hospital     │
 │ Node 1       │      │ Node 2       │      │ Node 3       │
 │ private MRI  │      │ private MRI  │      │ private MRI  │
 │ data (local) │      │ data (local) │      │ data (local) │
 │      +       │      │      +       │      │      +       │
 │ 3D U-Net     │      │ 3D U-Net     │      │ 3D U-Net     │
 │ local train  │      │ local train  │      │ local train  │
 │      +       │      │      +       │      │      +       │
 │ TenSEAL      │      │ TenSEAL      │      │ TenSEAL      │
 │ encrypt      │      │ encrypt      │      │ encrypt      │
 │      +       │      │      +       │      │      +       │
 │ DP noise     │      │ DP noise     │      │ DP noise     │
 └──────────────┘      └──────────────┘      └──────────────┘
        │                                             │
        └──────────────── live metrics ───────────────┘
                            │ WebSocket
                    ┌───────▼────────┐
                    │  React Dashboard│
                    │  (Recharts)     │
                    └─────────────────┘
```

## What Makes This Non-Trivial

| Naive Federated Learning | FedMed |
|---|---|
| Server sees raw model weights | Server aggregates **encrypted ciphertext** — never sees plaintext weights, even during aggregation |
| Assumes all clients stay online | Tolerates a hospital node **dropping mid-round** without crashing the training round |
| Plaintext gRPC | **TLS-secured** gRPC channels between server and every node |
| No formal privacy guarantee | **Differential privacy** noise gives a mathematical, provable bound against model-inversion and membership-inference attacks |
| Static offline evaluation | **Live WebSocket-streamed** metrics into a real-time dashboard, visible round-by-round as training happens |
| Single centralized dataset assumption | True **cross-silo** simulation — 3 independently partitioned, non-overlapping data stores |

## Key Modules

| Module | Tech | Purpose |
|---|---|---|
| Federated Learning Framework | Flower / PySyft | Orchestrates the decentralized training loop — client selection, round scheduling, communication |
| Computer Vision Model | PyTorch / MONAI | 3D U-Net architecture purpose-built for segmenting volumetric medical imagery (MRI/CT) |
| Privacy & Encryption | TenSEAL (CKKS scheme) | Homomorphic encryption enabling the server to aggregate weights while they remain mathematically encrypted |
| Differential Privacy | Custom (Gaussian mechanism) | Adds calibrated statistical noise to updates, bounding worst-case information leakage |
| Secure Transport | gRPC + TLS | Encrypts all network traffic between the central server and every hospital node |
| Training Dashboard | React / Recharts | Real-time monitoring UI for global model convergence, per-round accuracy, and segmentation output |

## Tech Stack Rationale (Why These Tools)

- **Flower over PySyft/TFF**: Flower is framework-agnostic (works with plain
  PyTorch, no heavy rewrite of the model code required), has an actively
  maintained simulation mode for local multi-node testing, and scales
  cleanly from "3 nodes on one laptop" to real multi-machine deployments.
- **MONAI over raw PyTorch**: MONAI provides medical-imaging-specific
  primitives (3D-aware transforms, loss functions like Dice loss, and
  pretrained medical backbones) so the modeling work focuses on the FL/PPML
  problem rather than reinventing medical image I/O.
- **TenSEAL (CKKS) over Paillier or generic MPC libraries**: CKKS supports
  approximate arithmetic on encrypted floating-point tensors, which maps
  directly onto neural network weights — this is what makes homomorphic
  aggregation of a real deep learning model computationally feasible,
  versus schemes designed only for encrypted integers.
- **gRPC + TLS over raw REST**: Flower is already gRPC-native, and gRPC's
  bidirectional streaming is a natural fit for repeated multi-round
  communication between server and nodes; TLS is the non-negotiable
  minimum for any traffic in a healthcare context.
- **React + Recharts + WebSocket over static reporting**: training a
  federated model can take a long time and can fail silently on one node;
  a live dashboard makes partial progress and node health visible
  immediately instead of only after the fact.

## Results (updated as training milestones land)

| Metric | Centralized Baseline | Federated (Plaintext) | Federated (Encrypted + DP) |
|---|---|---|---|
| Dice Score | _TBD — Week 1_ | _TBD — Week 2_ | _TBD — Week 4_ |
| Training Rounds to Converge | — | _TBD_ | _TBD_ |
| Privacy Budget (ε) | N/A | N/A | _TBD_ |
| Node-Dropout Recovery | N/A | _TBD — Week 2_ | _TBD_ |

> Goal: show the federated + encrypted model approaches centralized-baseline
> accuracy within an acceptable margin, proving privacy doesn't have to come
> at the cost of performance. This table gets filled in commit-by-commit as
> each week's milestone lands — see [docs/WEEKLY_PLAN.md](docs/WEEKLY_PLAN.md).

## Project Status & Roadmap

🚧 **Actively in development.** Built as a daily-commit project over a
4-week sprint, moving in this order: centralized baseline → federated loop
→ secure transport → node fault tolerance → homomorphic encryption → live
metrics → differential privacy → dashboard polish.

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Baseline + scaffolding | Centralized 3D U-Net accuracy number + 3 mock hospital nodes running |
| 2 | Federated loop + secure comms | Working FedAvg round over gRPC/TLS + mid-project federated audit |
| 2.5 | Resilience | Training round survives a node dropping mid-epoch |
| 3 | Homomorphic encryption + live metrics | Server aggregates ciphertext only + WebSocket metric stream |
| 4 | Differential privacy + dashboard | DP noise on updates + polished React/Recharts dashboard |

Full day-by-day breakdown: [docs/WEEKLY_PLAN.md](docs/WEEKLY_PLAN.md)
Architecture deep-dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## What We're Building Next

*(kept up to date as the project progresses — check off items as they land)*

**Immediate next steps (Week 1):**
- [x] Implement the 3D U-Net (`models/unet3d/model.py`) using MONAI building
      blocks — this same model definition gets reused unchanged inside
      every federated hospital node later. Verified: forward pass runs at
      the expected `(batch, 4, D, H, W)` shape.
- [x] Build a synthetic MRI dataset (`models/unet3d/dataset.py`) so the
      training loop is runnable and testable without needing the full
      BraTS download — plus a `BraTSDataset` loader ready for real data.
- [x] Centralized baseline training script (`models/unet3d/train_baseline.py`)
      — runs end-to-end, saves a model checkpoint and a `baseline_results.json`
      with the Dice score every later federated/encrypted result gets compared against.
- [x] Shared hospital client logic (`clients/common/client_app.py`) and thin
      per-hospital entry points (`clients/hospital_node_{1,2,3}/node.py`),
      each identifying its own hospital and local port.
- [x] Flower `ServerApp` scaffold (`server/aggregator/server_app.py`) that
      initializes and broadcasts the shared starting model.
- [x] `pyproject.toml` federation config wiring the server + client apps to
      3 SuperNodes on distinct local ports (9094/9095/9096) behind one
      local SuperLink (9092), plus `scripts/run_server.sh` /
      `scripts/run_node.sh` to launch them.
- [x] Unit tests (`tests/test_week1.py`) confirming the model, dataset,
      client, and server logic are all correctly wired — including that
      client and server agree on parameter count, a prerequisite for any
      future FedAvg aggregation to be well-defined. **All 6 tests pass.**
- [ ] Download and preprocess a real BraTS MRI subset (currently developed
      and tested against the synthetic dataset; swapping in real data is a
      one-flag change: `--data-source brats`)
- [ ] Run a live 3-node handshake (`flwr run . local-3-nodes` against the
      real SuperLink/SuperNode processes) — code and CLI commands are in
      place and each component is verified individually; the live
      multi-process network run is the last Week 1 box to check off

**Right after that (start of Week 2):**
- [x] Partition the dataset three ways with **zero overlap** between
      hospitals (`scripts/partition_data.py`), verified by assertion +
      a dedicated no-overlap test.
- [x] Implement each node's local `fit()`/`evaluate()` training routine
      (`clients/common/client_app.py`) — real training, not a stub.
- [x] Implement the server-side FedAvg aggregation strategy
      (`server/strategies/fedavg_strategy.py`) and run real end-to-end
      federated rounds, logging convergence in `docs/FEDERATED_AUDIT.md`.
- [x] TLS-secure the transport (`scripts/generate_tls_certs.sh` +
      `--secure` flag on both launch scripts), verified with `openssl verify`.
- [x] Node-dropout resilience: a round completes with only 2 of 3
      hospitals present (`scripts/node_resilience_test.py`), verified.
- [x] Mid-project federated audit written with real training numbers —
      see [`docs/FEDERATED_AUDIT.md`](docs/FEDERATED_AUDIT.md).
- [x] 7/7 automated tests passing (`tests/test_week2.py`).

Full Week 2 writeup: [`docs/WEEK2.md`](docs/WEEK2.md)

**Next (Week 3):**
- [ ] Integrate TenSEAL homomorphic encryption — encrypt weight tensors
      client-side before transmission
- [ ] Server aggregates directly on ciphertext, decrypting only the final result
- [ ] Stream loss/accuracy metrics to a WebSocket endpoint for live monitoring

**Later (Weeks 3–4):** homomorphic encryption of every weight update
(TenSEAL/CKKS), live WebSocket metric streaming, differential-privacy noise
injection with a documented epsilon/utility tradeoff, and the final
React/Recharts dashboard with segmentation-mask visualization and per-node
health indicators.

## Repo Structure

```
fedmed/
├── server/               # Central Flower aggregator + FedAvg/HE strategies
│   ├── aggregator/
│   └── strategies/
├── clients/               # Simulated hospital nodes (1 per hospital)
│   ├── hospital_node_1/
│   ├── hospital_node_2/
│   └── hospital_node_3/
├── models/unet3d/          # 3D U-Net (PyTorch + MONAI), shared by baseline + all nodes
├── encryption/
│   ├── tenseal_utils/       # Homomorphic encryption (CKKS) helpers
│   └── dp_utils/             # Differential privacy noise injection
├── grpc/
│   ├── protos/                 # Custom RPC message definitions
│   └── tls_certs/               # TLS certs (gitignored, generated locally)
├── dashboard/react-app/          # Live training metrics dashboard (React + Recharts)
├── data/                          # Local datasets (gitignored — never committed)
├── scripts/                        # Setup, data partitioning, node launch helpers
├── docs/                            # Architecture + weekly development plan
└── tests/
```

## Setup

```bash
git clone <repo-url>
cd fedmed
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Node/React dashboard setup instructions land in
`dashboard/react-app/README.md` once the frontend is scaffolded (Week 4).

## Running Week 1 Yourself

**1. Run the test suite** (verifies model, dataset, and client/server wiring):
```bash
pytest tests/test_week1.py -v
```

**2. Train the centralized baseline** (uses a synthetic dataset by default —
no external download required):
```bash
python -m models.unet3d.train_baseline --data-source synthetic --num-samples 20 --epochs 5
```
Once you have a real BraTS subset in `data/brats_raw/` (see `data/README.md`):
```bash
python -m models.unet3d.train_baseline --data-source brats --epochs 50
```

**3. Start the 3-hospital-node scaffold** (4 separate terminals):
```bash
./scripts/run_server.sh          # terminal 1 — starts the SuperLink
./scripts/run_node.sh 1 9094     # terminal 2 — Hospital 1
./scripts/run_node.sh 2 9095     # terminal 3 — Hospital 2
./scripts/run_node.sh 3 9096     # terminal 4 — Hospital 3
flwr run . local-3-nodes         # terminal 5 — triggers the handshake round
```
Success looks like each node logging its stub `fit()`/`evaluate()` call —
proof the server reached all three hospital identities individually.

## License

MIT (see [LICENSE](LICENSE))
