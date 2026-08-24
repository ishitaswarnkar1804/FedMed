# 🏥 FedMed

### Privacy-Preserving Federated Learning for Medical AI

<p align="center">

**Train medical AI collaboratively without sharing patient data.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](#)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-2C8EBB?style=for-the-badge)](#)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FF6F00?style=for-the-badge)](#)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-Encrypted%20Computation-6C47FF?style=for-the-badge)](#)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=for-the-badge\&logo=react\&logoColor=black)](#)

</p>

---

## ✨ What is FedMed?

**FedMed** is a cross-silo **Federated Learning platform for healthcare institutions**.

It enables multiple hospitals to collaboratively train a **3D brain-tumor segmentation model** while keeping sensitive MRI scans inside each hospital.

Instead of transferring patient data to a centralized server, each hospital:

```text
Private MRI Data
      ↓
Local Training
      ↓
Protected Model Update
      ↓
Secure Aggregation
      ↓
Global Model
```

> 🔐 **Raw MRI scans never leave their respective hospital nodes.**

---

## 📑 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [Key Idea](#-key-idea)
* [Medical AI Model](#-medical-ai-model)
* [Privacy Architecture](#-privacy-architecture)
* [System Architecture](#️-system-architecture)
* [Tech Stack](#️-tech-stack)
* [Project Structure](#-project-structure)
* [Installation](#️-installation)
* [Dataset](#-dataset)
* [Running FedMed](#️-running-fedmed)
* [Federated Training](#-federated-training)
* [Training Dashboard](#-training-dashboard)
* [Simulating Three Hospitals](#-simulating-three-hospitals)
* [Security Model](#-security-model)
* [Real-World Use Case](#-real-world-use-case)
* [Project Goals](#-project-goals)
* [Future Improvements](#-future-improvements)
* [Why FedMed?](#-why-fedmed)
* [Disclaimer](#️-disclaimer)
* [Contributing](#-contributing)
* [License](#-license)

---

# 🚀 Overview

Traditional medical AI often requires collecting large datasets in a centralized location:

```text
┌─────────────┐
│  Hospital A │
└──────┬──────┘
       │
┌──────▼──────┐
│  Hospital B │ ──────► Central Dataset ──────► Train Model
└──────┬──────┘
       │
┌──────▼──────┐
│  Hospital C │
└─────────────┘
```

This creates significant:

* 🔒 Privacy challenges
* ⚖️ Compliance challenges
* 🛡️ Security concerns
* 📋 Data-sharing restrictions
* 🏥 Institutional ownership concerns

### FedMed's approach

```text
                         ┌──────────────────────────┐
                         │      🏥 FedMed Server    │
                         │                          │
                         │     Global 3D U-Net      │
                         │     Secure Aggregation   │
                         └────────────┬─────────────┘
                                      │
                              Global Model
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
       ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
       │ Hospital A  │         │ Hospital B  │         │ Hospital C  │
       │             │         │             │         │             │
       │ 🔒 Private  │         │ 🔒 Private  │         │ 🔒 Private  │
       │    MRI      │         │    MRI      │         │    MRI      │
       │      ↓      │         │      ↓      │         │      ↓      │
       │ Local Train │         │ Local Train │         │ Local Train │
       └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                             Protected Updates
                                      │
                                      ▼
                            🔐 Secure Aggregation
                                      │
                                      ▼
                              Global Model v2
```

---

# 🎯 Problem Statement

Training accurate machine-learning models for rare diseases requires **large and diverse datasets**.

However, hospitals often cannot directly share patient-level medical data because of:

| Challenge                 | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| 🔒 Patient Privacy        | Medical images contain sensitive information          |
| ⚖️ Regulations            | Healthcare data is subject to strict regulations      |
| 🏥 Institutional Policies | Hospitals may restrict data sharing                   |
| 🛡️ Security              | Centralized datasets create attractive attack targets |
| 📁 Data Ownership         | Institutions maintain control over their datasets     |

### 💡 FedMed's Solution

FedMed allows hospitals to **train collaboratively without centralizing their private datasets**.

---

# 💡 Key Idea

FedMed follows a repeated federated training cycle:

```text
       ┌──────────────────────────┐
       │ 1. Create Global Model   │
       └────────────┬─────────────┘
                    ↓
       ┌──────────────────────────┐
       │ 2. Send Model to Clients │
       └────────────┬─────────────┘
                    ↓
       ┌──────────────────────────┐
       │ 3. Train Locally         │
       └────────────┬─────────────┘
                    ↓
       ┌──────────────────────────┐
       │ 4. Protect Updates       │
       └────────────┬─────────────┘
                    ↓
       ┌──────────────────────────┐
       │ 5. Aggregate Updates     │
       └────────────┬─────────────┘
                    ↓
       ┌──────────────────────────┐
       │ 6. Improve Global Model  │
       └────────────┬─────────────┘
                    ↓
                  Repeat
```

> **The server receives model updates rather than patient MRI data.**

---

# 🧠 Medical AI Model

FedMed uses a **3D U-Net architecture** for volumetric medical-image segmentation.

### Input

```text
                 🧠 MRI Scan
                     │
                     ▼
          ┌─────────────────────┐
          │    3D MRI Volume    │
          │                     │
          │       Brain         │
          │         ↓           │
          │       Tumor         │
          └─────────────────────┘
```

### Output

```text
3D MRI Volume
      │
      ▼
  ┌─────────┐
  │ 3D U-Net│
  └────┬────┘
       │
       ▼
Tumor Segmentation Mask
```

### 📊 Evaluation Metrics

FedMed can evaluate segmentation performance using:

* **Dice Score**
* **IoU — Intersection over Union**
* **Validation Loss**
* **Precision**
* **Recall**

---

# 🔐 Privacy Architecture

FedMed combines multiple privacy mechanisms.

## 1. Federated Learning

Patient data remains inside the hospital.

```text
┌─────────────────────┐
│   Hospital Dataset  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Local Training    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Model Update      │
└──────────┬──────────┘
           ↓
      FedMed Server
```

Only the resulting model update is sent to the federation server.

---

## 2. 🔒 Homomorphic Encryption

FedMed uses **TenSEAL** to protect model updates.

The goal is to allow aggregation of protected values without requiring the server to access the underlying plaintext updates.

```text
Local Weights
     │
     ▼
 Encryption
     │
     ▼
Encrypted Update
     │
     ▼
FedMed Server
     │
     ▼
Secure Aggregation
```

---

## 3. 🛡️ Secure Aggregation

The aggregation layer is designed to prevent the server from directly inspecting an individual hospital's contribution.

> ⚠️ **Important:** FedMed is designed to improve privacy; it should not be interpreted as providing absolute or mathematically complete privacy under every possible threat model.

---

# 🏗️ System Architecture

```text
                         ┌────────────────────────┐
                         │    React Dashboard     │
                         │      + Recharts        │
                         └───────────┬────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │     FedMed Server      │
                         │                        │
                         │       Flower          │
                         │  Global Model Manager  │
                         │  Secure Aggregation    │
                         └───────────┬────────────┘
                                     │
                   ┌─────────────────┼─────────────────┐
                   │                 │                 │
                   ▼                 ▼                 ▼
             Hospital A        Hospital B        Hospital C
             ──────────        ──────────        ──────────
             Private MRI      Private MRI      Private MRI
                   │                 │                 │
                   ▼                 ▼                 ▼
              Preprocess        Preprocess        Preprocess
                   │                 │                 │
                   ▼                 ▼                 ▼
                3D U-Net          3D U-Net          3D U-Net
                   │                 │                 │
                   ▼                 ▼                 ▼
               Local Train       Local Train       Local Train
                   │                 │                 │
                   ▼                 ▼                 ▼
                Encrypt           Encrypt           Encrypt
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     │
                                     ▼
                            Secure Aggregation
                                     │
                                     ▼
                           Updated Global Model
```

---

# 🛠️ Tech Stack

## 🧠 Machine Learning

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| **Python**   | Core development        |
| **PyTorch**  | Deep learning           |
| **MONAI**    | Medical AI framework    |
| **3D U-Net** | Volumetric segmentation |

## 🌐 Federated Learning

| Technology | Purpose                          |
| ---------- | -------------------------------- |
| **Flower** | Federated learning orchestration |

## 🔐 Privacy & Security

| Technology                 | Purpose                               |
| -------------------------- | ------------------------------------- |
| **TenSEAL**                | Homomorphic encryption                |
| **Homomorphic Encryption** | Protected computation                 |
| **Secure Aggregation**     | Privacy-preserving update aggregation |

## 💻 Frontend

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| **React**    | Dashboard UI            |
| **Recharts** | Training visualizations |

## 🗃️ Data

* MRI / CT volumetric medical images
* NIfTI-compatible preprocessing
* DICOM-compatible preprocessing

---

# 📂 Project Structure

```text
FedMed/
│
├── server/
│   ├── server.py
│   ├── aggregation.py
│   └── config.py
│
├── clients/
│   ├── hospital_a/
│   │   ├── client.py
│   │   ├── dataset.py
│   │   └── config.py
│   │
│   ├── hospital_b/
│   │   ├── client.py
│   │   ├── dataset.py
│   │   └── config.py
│   │
│   └── hospital_c/
│       ├── client.py
│       ├── dataset.py
│       └── config.py
│
├── models/
│   ├── unet3d.py
│   ├── loss.py
│   └── metrics.py
│
├── privacy/
│   ├── encryption.py
│   └── secure_aggregation.py
│
├── preprocessing/
│   ├── preprocessing.py
│   └── transforms.py
│
├── dashboard/
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── data/
│   └── README.md
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/FedMed.git
cd FedMed
```

## 2. Create a Python environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### Example dependencies

```text
torch
torchvision
monai
flwr
tenseal
numpy
pandas
scikit-learn
nibabel
pydicom
```

## 4. Install frontend dependencies

```bash
cd dashboard
npm install
```

---

# 📊 Dataset

FedMed is designed for **volumetric medical-imaging datasets**.

A suitable dataset should contain:

```text
MRI Volume
     +
Ground Truth
Segmentation Mask
```

For development, the dataset can be divided into three simulated hospital datasets:

```text
                 Dataset
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Hospital A  Hospital B  Hospital C
```

Each hospital should maintain its own local copy or partition.

## ⚠️ Privacy Notice

> **Never upload real patient-identifiable medical data to this repository.**

For demonstrations and development, use:

* Publicly available de-identified datasets
* Synthetic medical data
* Artificially generated datasets

---

# ▶️ Running FedMed

## Start the federated server

```bash
python server/server.py
```

## Start Hospital A

```bash
python clients/hospital_a/client.py
```

## Start Hospital B

```bash
python clients/hospital_b/client.py
```

## Start Hospital C

```bash
python clients/hospital_c/client.py
```

The clients connect to the federated server and participate in the training rounds.

---

# 🔄 Federated Training

A typical training round looks like:

```text
                     GLOBAL MODEL
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     Hospital A      Hospital B      Hospital C
          │               │               │
     Local Epochs    Local Epochs    Local Epochs
          │               │               │
          ▼               ▼               ▼
     Model Update    Model Update    Model Update
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                  🔒 Encryption Layer
                          │
                          ▼
                  Secure Aggregation
                          │
                          ▼
                    GLOBAL MODEL
                          │
                          ▼
                       Round 2
```

This process continues until the global model reaches the desired performance.

---

# 📈 Training Dashboard

The React dashboard provides real-time visibility into federated training.

### Dashboard Metrics

| Metric                   | Description                          |
| ------------------------ | ------------------------------------ |
| **Federated Round**      | Current global training round        |
| **Global Loss**          | Loss of the global model             |
| **Dice Score**           | Segmentation quality                 |
| **IoU**                  | Intersection over Union              |
| **Hospital Status**      | Client connection/training status    |
| **Training Time**        | Time taken per round                 |
| **Client Participation** | Hospitals participating in the round |

### Example

```text
┌───────────────────────────────────────────────┐
│                 FedMed Dashboard              │
├───────────────────────────────────────────────┤
│                                               │
│  Round: 15          Dice Score: 0.87          │
│                                               │
│  Global Loss                                  │
│  1.0 ┤╲                                       │
│      │ ╲                                      │
│  0.5 │  ╲________                             │
│      │           ╲____                        │
│  0.0 └──────────────────────                  │
│                                               │
│  Hospital A     ● Training                    │
│  Hospital B     ● Training                    │
│  Hospital C     ● Training                    │
│                                               │
└───────────────────────────────────────────────┘
```

---

# 🧪 Simulating Three Hospitals

For a hackathon or local demonstration, three hospitals can be simulated as three isolated clients.

```text
                     Local Machine
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
          Client A    Client B    Client C
           MRI-A       MRI-B       MRI-C
              │           │           │
              └───────────┼───────────┘
                          ▼
                    FedMed Server
```

Each client can have:

* Different patient distributions
* Different image counts
* Different tumor characteristics
* Different local validation data

This makes the demonstration more realistic than simply splitting identical batches.

---

# 🔒 Security Model

FedMed aims to protect against unauthorized access to patient data during collaborative training.

## 🔐 Data That Stays Local

```text
❌ Patient MRI scans
❌ Patient identifiers
❌ Local medical records
❌ Raw segmentation datasets
```

## 📡 Data Exchanged

```text
✅ Model parameters / updates
✅ Training metadata
✅ Aggregated metrics
```

Model updates should be protected before transmission using the configured privacy layer.

---

# 🌍 Real-World Use Case

Imagine three hospitals located in different countries:

```text
              🇮🇳 Hospital A
                    │
                    │
                    ├──────────► 🏥 FedMed ◄──────────┐
                    │                                  │
              🇬🇧 Hospital B                          │
                    │                                  │
                    └──────────────────────────────────┤
                                                       │
              🇺🇸 Hospital C ─────────────────────────┘
```

Each institution contributes to a shared AI model while maintaining control over its own patient dataset.

### Potential Applications

* 🧠 Brain tumor segmentation
* 🫁 Lung disease detection
* 🎗️ Cancer classification
* 👁️ Retinal disease detection
* 🧠 Stroke prediction
* 🧬 Rare disease research

---

# 🎯 Project Goals

### Core Features

* [x] Federated Learning architecture
* [x] Distributed hospital clients
* [x] Local model training
* [x] 3D medical-image segmentation
* [x] Encrypted model updates
* [x] Secure aggregation architecture
* [x] Training monitoring dashboard

### Future Features

* [ ] Multi-GPU distributed training
* [ ] Differential Privacy
* [ ] Production-grade authentication
* [ ] Kubernetes deployment
* [ ] Real-world multi-institution deployment

---

# 🚧 Future Improvements

## 🔐 Differential Privacy

Add noise to updates to provide stronger protection against:

* Model-inversion attacks
* Membership-inference attacks

## 🪪 Client Authentication

Introduce secure hospital identity management using:

* Digital certificates
* Public-key infrastructure

## 🔄 Fault Tolerance

Allow training to continue when one or more hospital nodes temporarily disconnect.

## 📦 Non-IID Data Handling

Medical datasets can differ significantly between hospitals.

Potential approaches include:

* **FedProx**
* **Personalized Federated Learning**
* **Adaptive Aggregation**
* **Client Weighting**

## 🛡️ Advanced Privacy

Potential future additions:

* Differential Privacy
* Secure Multi-Party Computation
* Trusted Execution Environments
* Byzantine-robust aggregation

---

# 🏆 Why FedMed?

### Traditional Centralized ML

```text
More Data
    ↓
Centralized Dataset
    ↓
Better Model
    ↓
Higher Privacy Risk
```

### FedMed

```text
More Hospitals
      ↓
More Diverse Local Data
      ↓
Federated Training
      ↓
Better Global Model
      ↓
Patient Data Remains Local
```

> **FedMed demonstrates how privacy-preserving machine learning can make cross-institution medical AI collaboration possible.**

---

# ⚠️ Disclaimer

FedMed is an **educational/research prototype** and is **not intended for clinical diagnosis or treatment**.

The privacy and security guarantees of a real-world deployment depend on the complete system architecture, threat model, encryption configuration, authentication, infrastructure, and regulatory requirements.

> **Do not use real patient-identifiable data in development or demonstration environments without appropriate authorization and safeguards.**

---

# 🤝 Contributing

Contributions are welcome!

### 1. Create a feature branch

```bash
git checkout -b feature/your-feature
```

### 2. Stage your changes

```bash
git add .
```

### 3. Commit

```bash
git commit -m "Add your feature"
```

### 4. Push

```bash
git push origin feature/your-feature
```

### 5. Open a Pull Request

After pushing your branch, open a **Pull Request** on GitHub for review.

---

# 📜 License

This project is intended for **research and educational purposes**.

Add your preferred open-source license here, for example:

```text
MIT License
```

---

# 👨‍💻 Built With

<p align="center">

**Python** • **PyTorch** • **MONAI** • **Flower** • **TenSEAL** • **React** • **Recharts**

</p>

---

<p align="center">

## 🏥 FedMed

### *Collaborative Intelligence. Private Data. Better Healthcare.*

</p>
