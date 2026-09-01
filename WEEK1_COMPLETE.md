# 🧠 FedMed — Week 1 Complete Build

<p align="center">

# FedMed

### Privacy-Preserving Federated Brain Tumor Segmentation

**Foundation Week — Centralized Baseline + Federated Node Scaffolding**

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-2C8EBB?style=for-the-badge)
![Flower](https://img.shields.io/badge/Flower-1.8%2B-FF6F00?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passed-success?style=for-the-badge)

</p>

---

## 📌 Week 1 Status

| Component | Status |
|---|:---:|
| 3D U-Net | ✅ Complete |
| Synthetic MRI Dataset | ✅ Complete |
| Centralized Training | ✅ Complete |
| Dice Evaluation | ✅ Complete |
| Model Checkpoint | ✅ Complete |
| Training Visualization | ✅ Complete |
| Flower Server | ✅ Complete |
| Hospital Node 1 | ✅ Complete |
| Hospital Node 2 | ✅ Complete |
| Hospital Node 3 | ✅ Complete |
| Client/Server Wiring | ✅ Complete |
| Automated Tests | ✅ 6/6 Passed |

---

# 🌟 Project Overview

FedMed is a cross-silo federated learning framework designed for
**privacy-preserving brain tumor segmentation from MRI scans**.

The project follows a simple principle:

> **Patient MRI data stays inside the hospital.  
> Model knowledge is exchanged between hospitals and the server.**

Instead of pooling sensitive medical data into one centralized dataset,
each hospital can eventually train locally and contribute model updates
to a shared global model.

---

# 🏗️ Week 1 Architecture

```text
                           ┌───────────────────────────┐
                           │       🧠 FEDMED           │
                           │      CENTRAL SERVER       │
                           │                           │
                           │       Global Model        │
                           │          FedAvg           │
                           └─────────────┬─────────────┘
                                         │
                                  🌐 Flower
                                  SuperLink
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼                          ▼
      ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
      │  🏥 HOSPITAL 1│          │  🏥 HOSPITAL 2│          │  🏥 HOSPITAL 3│
      │               │          │               │          │               │
      │    :9094      │          │    :9095      │          │    :9096      │
      │               │          │               │          │               │
      │ 🔒 Private MRI│          │ 🔒 Private MRI│          │ 🔒 Private MRI│
      │               │          │               │          │               │
      │ Local Client  │          │ Local Client  │          │ Local Client  │
      └───────────────┘          └───────────────┘          └───────────────┘
