# 🏥 FedMed

### Privacy-Preserving Federated Learning for Medical 

<p align="center">

**Train medical AI collaboratively without sharing patient data.**

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](#)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-2C8EBB?style=for-the-badge)](#)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-FF6F00?style=for-the-badge)](#)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-Encrypted%20Computation-6C47FF?style=for-the-badge)](#)
[![React](https://img.shields.io/badge/React-Dashboard-61DAFB?style=for-the-badge\&logo=react\&logoColor=black)](#)

</p>

---

## 📖 Overview

**FedMed** is a cross-silo Federated Learning platform designed for healthcare institutions.

It enables multiple hospitals to collaboratively train a **3D brain-tumor segmentation model** while keeping sensitive MRI scans inside each hospital.

Instead of sending patient data to a centralized server, each hospital trains the model locally and sends only **protected model updates** for global aggregation.

> 🔐 **Raw MRI scans never leave their respective hospital nodes.**

---

### Week-1 Outcome


<img width="1024" height="1536" alt="week1_PosterExplanation" src="https://github.com/user-attachments/assets/c504c27f-417c-4e94-b213-9668724d1c94" />


---

## 📑 Table of Contents

* [Overview](#-overview)
* [Problem Statement](#-problem-statement)
* [How FedMed Works](#-how-fedmed-works)
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
* [Why FedMed](#-why-fedmed)
* [Disclaimer](#️-disclaimer)
* [Contributing](#-contributing)
* [License](#-license)

---

# 🎯 Problem Statement

Training accurate machine-learning models for rare diseases requires large and diverse datasets.

However, hospitals often cannot share patient-level medical data because of:

* 🔒 Patient privacy requirements
* ⚖️ Healthcare regulations
* 🏥 Institutional data-sharing policies
* 🛡️ Security concerns
* 📁 Data ownership

FedMed addresses this problem by allowing hospitals to **train collaboratively without centralizing their private datasets**.

### Traditional Medical AI

```mermaid
flowchart LR
    A["🏥 Hospital A<br/>Private Patient Data"]
    B["🏥 Hospital B<br/>Private Patient Data"]
    C["🏥 Hospital C<br/>Private Patient Data"]

    D["🗄️ Central Dataset"]
    E["🧠 Train Medical AI"]

    A --> D
    B --> D
    C --> D
    D --> E

    classDef hospital fill:#E8F1FF,stroke:#2563EB,stroke-width:2px,color:#111827
    classDef central fill:#FEE2E2,stroke:#DC2626,stroke-width:2px,color:#111827
    classDef model fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class A,B,C hospital
    class D central
    class E model
```

### The Problem

Centralizing medical data creates significant:

**Privacy Risk → Compliance Risk → Security Risk**

---

# 🚀 How FedMed Works

FedMed replaces centralized data collection with **Federated Learning**.

```mermaid
flowchart TB

    S["🌐 FedMed Server<br/><br/>Global 3D U-Net<br/>+<br/>Secure Aggregation"]

    A["🏥 Hospital A<br/><br/>Private MRI Data"]
    B["🏥 Hospital B<br/><br/>Private MRI Data"]
    C["🏥 Hospital C<br/><br/>Private MRI Data"]

    AT["🧠 Local Training"]
    BT["🧠 Local Training"]
    CT["🧠 Local Training"]

    AU["🔐 Protected Update"]
    BU["🔐 Protected Update"]
    CU["🔐 Protected Update"]

    A --> AT --> AU --> S
    B --> BT --> BU --> S
    C --> CT --> CU --> S

    S --> M["🌍 Updated Global Model"]

    M --> A
    M --> B
    M --> C

    classDef server fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef train fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef privacy fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef model fill:#FCE7F3,stroke:#DB2777,stroke-width:2px,color:#111827

    class S server
    class A,B,C hospital
    class AT,BT,CT train
    class AU,BU,CU privacy
    class M model
```

### 🔄 Training Cycle

```mermaid
flowchart TD
    A["1️⃣ Create Global Model"]
    B["2️⃣ Send Model to Hospitals"]
    C["3️⃣ Hospitals Train Locally"]
    D["4️⃣ Protect Model Updates"]
    E["5️⃣ Secure Aggregation"]
    F["6️⃣ Update Global Model"]
    G{"7️⃣ Desired Performance?"}

    A --> B --> C --> D --> E --> F --> G

    G -- "No" --> B
    G -- "Yes" --> H["✅ Training Complete"]

    classDef process fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#111827
    classDef security fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef decision fill:#F3E8FF,stroke:#9333EA,stroke-width:2px,color:#111827
    classDef done fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class A,B,C,F process
    class D,E security
    class G decision
    class H done
```

> **The server receives model updates rather than patient MRI data.**

---

# 🧠 Medical AI Model

FedMed uses a **3D U-Net** architecture for volumetric medical-image segmentation.

### Input → Model → Output

```mermaid
flowchart LR
    A["🧠 3D MRI Volume<br/><br/>Brain + Tumor"]
    B["⚙️ 3D U-Net"]
    C["🎯 Tumor Segmentation Mask"]

    A --> B --> C

    classDef input fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef model fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef output fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class A input
    class B model
    class C output
```

### 📊 Evaluation Metrics

| Metric              | Purpose                         |
| ------------------- | ------------------------------- |
| **Dice Score**      | Measures segmentation overlap   |
| **IoU**             | Intersection over Union         |
| **Validation Loss** | Measures model error            |
| **Precision**       | Measures prediction correctness |
| **Recall**          | Measures detection coverage     |

---

# 🔐 Privacy Architecture

FedMed uses multiple privacy mechanisms.

## 1. Federated Learning

```mermaid
flowchart LR
    A["🏥 Hospital Dataset"]
    B["🧠 Local Training"]
    C["📦 Model Update"]
    D["🌐 Federation Server"]

    A --> B --> C --> D

    classDef local fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef train fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef update fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef server fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827

    class A local
    class B train
    class C update
    class D server
```

**Patient data stays inside the hospital.**

---

## 2. 🔒 Homomorphic Encryption

FedMed uses **TenSEAL** to protect model updates.

```mermaid
flowchart LR
    A["🧠 Local Weights"]
    B["🔐 Encryption"]
    C["🔒 Encrypted Update"]
    D["🌐 FedMed Server"]
    E["🛡️ Secure Aggregation"]

    A --> B --> C --> D --> E

    classDef weights fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef encryption fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef server fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef aggregation fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class A weights
    class B,C encryption
    class D server
    class E aggregation
```

The goal is to allow aggregation of protected values without requiring the server to access the underlying plaintext updates.

---

## 3. 🛡️ Secure Aggregation

The aggregation layer is designed to prevent the server from directly inspecting an individual hospital's contribution.

```mermaid
flowchart TB
    A["🔒 Hospital A Update"]
    B["🔒 Hospital B Update"]
    C["🔒 Hospital C Update"]

    D["🛡️ Secure Aggregation"]

    E["🌍 Aggregated Global Update"]

    A --> D
    B --> D
    C --> D

    D --> E

    classDef update fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef secure fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef output fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class A,B,C update
    class D secure
    class E output
```

> ⚠️ **Important:** FedMed is designed to improve privacy; it should not be interpreted as providing absolute or mathematically complete privacy under every possible threat model.

---

# 🏗️ System Architecture

```mermaid
flowchart TB

    UI["💻 React Dashboard<br/>+ Recharts"]

    SERVER["🌐 FedMed Server<br/><br/>Flower<br/>Global Model Manager<br/>Secure Aggregation"]

    A["🏥 Hospital A<br/>Private MRI"]
    B["🏥 Hospital B<br/>Private MRI"]
    C["🏥 Hospital C<br/>Private MRI"]

    AP["Preprocessing"]
    BP["Preprocessing"]
    CP["Preprocessing"]

    AM["3D U-Net"]
    BM["3D U-Net"]
    CM["3D U-Net"]

    AT["Local Training"]
    BT["Local Training"]
    CT["Local Training"]

    AE["🔐 Encrypt"]
    BE["🔐 Encrypt"]
    CE["🔐 Encrypt"]

    AGG["🛡️ Secure Aggregation"]

    GLOBAL["🌍 Updated Global Model"]

    UI --> SERVER

    SERVER --> A
    SERVER --> B
    SERVER --> C

    A --> AP --> AM --> AT --> AE
    B --> BP --> BM --> BT --> BE
    C --> CP --> CM --> CT --> CE

    AE --> AGG
    BE --> AGG
    CE --> AGG

    AGG --> GLOBAL
    GLOBAL --> SERVER

    classDef dashboard fill:#DBEAFE,stroke:#2563EB,stroke-width:3px,color:#111827
    classDef server fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef process fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef security fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef aggregation fill:#FCE7F3,stroke:#DB2777,stroke-width:3px,color:#111827
    classDef global fill:#D1FAE5,stroke:#059669,stroke-width:3px,color:#111827

    class UI dashboard
    class SERVER server
    class A,B,C hospital
    class AP,BP,CP,AM,BM,CM,AT,BT,CT process
    class AE,BE,CE security
    class AGG aggregation
    class GLOBAL global
```

---

# 🛠️ Tech Stack

## Machine Learning

| Technology      | Role               |
| --------------- | ------------------ |
| 🐍 **Python**   | Core development   |
| 🔥 **PyTorch**  | Deep learning      |
| 🏥 **MONAI**    | Medical AI         |
| 🧠 **3D U-Net** | Image segmentation |

## Federated Learning

| Technology    | Role                             |
| ------------- | -------------------------------- |
| 🌸 **Flower** | Federated learning orchestration |

## Privacy & Security

| Technology                    | Role                           |
| ----------------------------- | ------------------------------ |
| 🔐 **TenSEAL**                | Homomorphic encryption         |
| 🔒 **Homomorphic Encryption** | Protected computation          |
| 🛡️ **Secure Aggregation**    | Privacy-preserving aggregation |

## Frontend

| Technology      | Role               |
| --------------- | ------------------ |
| ⚛️ **React**    | Dashboard          |
| 📊 **Recharts** | Data visualization |

## Data

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

## 3. Install dependencies

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

FedMed is designed for volumetric medical-imaging datasets.

A suitable dataset should contain:

```mermaid
flowchart LR
    A["🧠 MRI Volume"]
    B["🎯 Ground Truth<br/>Segmentation Mask"]

    A --> C["📊 Training Dataset"]
    B --> C

    classDef data fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef dataset fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class A,B data
    class C dataset
```

For development, the dataset can be divided into three simulated hospital datasets:

```mermaid
flowchart TB
    D["📊 Development Dataset"]

    D --> A["🏥 Hospital A"]
    D --> B["🏥 Hospital B"]
    D --> C["🏥 Hospital C"]

    A --> A1["Private Local Partition"]
    B --> B1["Private Local Partition"]
    C --> C1["Private Local Partition"]

    classDef dataset fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef local fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class D dataset
    class A,B,C hospital
    class A1,B1,C1 local
```

> ⚠️ **Never upload real patient-identifiable medical data to this repository.**

For demonstrations and development, use:

* Publicly available de-identified datasets
* Synthetic medical data
* Artificially generated datasets

---

# ▶️ Running FedMed

### Start the federated server

```bash
python server/server.py
```

### Start Hospital A

```bash
python clients/hospital_a/client.py
```

### Start Hospital B

```bash
python clients/hospital_b/client.py
```

### Start Hospital C

```bash
python clients/hospital_c/client.py
```

---

# 🔄 Federated Training

A complete training round:

```mermaid
flowchart TB

    G["🌍 GLOBAL MODEL"]

    G --> A["🏥 Hospital A"]
    G --> B["🏥 Hospital B"]
    G --> C["🏥 Hospital C"]

    A --> AT["Local Epochs"]
    B --> BT["Local Epochs"]
    C --> CT["Local Epochs"]

    AT --> AU["Model Update"]
    BT --> BU["Model Update"]
    CT --> CU["Model Update"]

    AU --> E["🔐 Encryption Layer"]
    BU --> E
    CU --> E

    E --> S["🛡️ Secure Aggregation"]

    S --> N["🌍 UPDATED GLOBAL MODEL"]

    N --> R{"Continue Training?"}

    R -- "Yes" --> G
    R -- "No" --> F["✅ Final Model"]

    classDef global fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef train fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef update fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef secure fill:#FCE7F3,stroke:#DB2777,stroke-width:3px,color:#111827
    classDef decision fill:#F3E8FF,stroke:#9333EA,stroke-width:2px,color:#111827
    classDef final fill:#D1FAE5,stroke:#059669,stroke-width:3px,color:#111827

    class G,N global
    class A,B,C hospital
    class AT,BT,CT train
    class AU,BU,CU update
    class E,S secure
    class R decision
    class F final
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

### Dashboard Flow

```mermaid
flowchart LR
    A["🌐 Federated Server"]
    B["📡 Training Metrics"]
    C["💻 React Dashboard"]
    D["📊 Recharts"]
    E["👨‍💻 Developer / Researcher"]

    A --> B --> C --> D --> E

    classDef server fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef metrics fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#111827
    classDef dashboard fill:#DBEAFE,stroke:#2563EB,stroke-width:3px,color:#111827
    classDef chart fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef user fill:#F3F4F6,stroke:#6B7280,stroke-width:2px,color:#111827

    class A server
    class B metrics
    class C dashboard
    class D chart
    class E user
```

---

# 🧪 Simulating Three Hospitals

For a hackathon or local demonstration, three hospitals can be simulated as three isolated clients.

```mermaid
flowchart TB

    M["💻 Local Machine"]

    M --> A["🏥 Client A<br/>MRI-A"]
    M --> B["🏥 Client B<br/>MRI-B"]
    M --> C["🏥 Client C<br/>MRI-C"]

    A --> S["🌐 FedMed Server"]
    B --> S
    C --> S

    classDef machine fill:#F3F4F6,stroke:#6B7280,stroke-width:2px,color:#111827
    classDef client fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef server fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827

    class M machine
    class A,B,C client
    class S server
```

Each client can have:

* Different patient distributions
* Different image counts
* Different tumor characteristics
* Different local validation data

This makes the demonstration more realistic than simply splitting identical batches.

---

# 🔒 Security Model

## Data That Stays Local

```mermaid
flowchart TB
    A["🏥 Hospital"]

    A --> B["❌ Patient MRI Scans"]
    A --> C["❌ Patient Identifiers"]
    A --> D["❌ Local Medical Records"]
    A --> E["❌ Raw Segmentation Dataset"]

    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:3px,color:#111827
    classDef private fill:#FEE2E2,stroke:#DC2626,stroke-width:2px,color:#111827

    class A hospital
    class B,C,D,E private
```

## Data Exchanged

```mermaid
flowchart TB
    H["🏥 Hospital"]

    H --> A["✅ Model Parameters / Updates"]
    H --> B["✅ Training Metadata"]
    H --> C["✅ Aggregated Metrics"]

    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:3px,color:#111827
    classDef exchange fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827

    class H hospital
    class A,B,C exchange
```

Model updates should be protected before transmission using the configured privacy layer.

---

# 🌍 Real-World Use Case

Imagine three hospitals located in different countries.

```mermaid
flowchart LR

    A["🇮🇳 Hospital A"]
    B["🇬🇧 Hospital B"]
    C["🇺🇸 Hospital C"]

    F["🏥 FedMed<br/>Federated AI Platform"]

    A --> F
    B --> F
    C --> F

    F --> G["🌍 Shared Global Model"]

    classDef hospital fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef platform fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef model fill:#DCFCE7,stroke:#16A34A,stroke-width:3px,color:#111827

    class A,B,C hospital
    class F platform
    class G model
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

### ✅ Completed

* [x] Federated Learning architecture
* [x] Distributed hospital clients
* [x] Local model training
* [x] 3D medical-image segmentation
* [x] Encrypted model updates
* [x] Secure aggregation architecture
* [x] Training monitoring dashboard

### 🚧 Planned

* [ ] Multi-GPU distributed training
* [ ] Differential Privacy
* [ ] Production-grade authentication
* [ ] Kubernetes deployment
* [ ] Real-world multi-institution deployment

---

# 🚧 Future Improvements

### 🔐 Differential Privacy

Add noise to updates to provide stronger protection against model-inversion and membership-inference attacks.

### 🪪 Client Authentication

Introduce secure hospital identity management using certificates or public-key infrastructure.

### 🔄 Fault Tolerance

Allow training to continue when one or more hospital nodes temporarily disconnect.

### 📦 Non-IID Data Handling

Medical datasets can differ significantly between hospitals.

Future versions can implement:

* FedProx
* Personalized Federated Learning
* Adaptive aggregation
* Client weighting

### 🛡️ Advanced Privacy

Potential future additions include:

* Differential Privacy
* Secure Multi-Party Computation
* Trusted Execution Environments
* Byzantine-robust aggregation

---

# 🏆 Why FedMed?

## Traditional Centralized ML

```mermaid
flowchart TD
    A["📊 More Data"]
    B["🗄️ Centralized Dataset"]
    C["🧠 Better Model"]
    D["⚠️ Higher Privacy Risk"]

    A --> B --> C --> D

    classDef data fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef central fill:#FEE2E2,stroke:#DC2626,stroke-width:2px,color:#111827
    classDef model fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef risk fill:#FECACA,stroke:#B91C1C,stroke-width:3px,color:#111827

    class A data
    class B central
    class C model
    class D risk
```

## FedMed

```mermaid
flowchart TD
    A["🏥 More Hospitals"]
    B["🌍 More Diverse Local Data"]
    C["🔄 Federated Training"]
    D["🧠 Better Global Model"]
    E["🔐 Patient Data Remains Local"]

    A --> B --> C --> D --> E

    classDef hospitals fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#111827
    classDef data fill:#DBEAFE,stroke:#2563EB,stroke-width:2px,color:#111827
    classDef federated fill:#EDE9FE,stroke:#7C3AED,stroke-width:3px,color:#111827
    classDef model fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#111827
    classDef privacy fill:#FEF3C7,stroke:#D97706,stroke-width:3px,color:#111827

    class A hospitals
    class B data
    class C federated
    class D model
    class E privacy
```

> 💡 **FedMed demonstrates how privacy-preserving machine learning can make cross-institution medical AI collaboration possible.**

---






# ⚠️ Disclaimer

FedMed is an **educational/research prototype** and is **not intended for clinical diagnosis or treatment**.

The privacy and security guarantees of a real-world deployment depend on the complete system architecture, threat model, encryption configuration, authentication, infrastructure, and regulatory requirements.

> 🚨 **Do not use real patient-identifiable data in development or demonstration environments without appropriate authorization and safeguards.**

---

# 🤝 Contributing

Contributions are welcome!

### Create a feature branch

```bash
git checkout -b feature/your-feature
```

### Stage changes

```bash
git add .
```

### Commit

```bash
git commit -m "Add your feature"
```

### Push

```bash
git push origin feature/your-feature
```

Then open a **Pull Request** on GitHub.

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

🐍 **Python**
🔥 **PyTorch**
🏥 **MONAI**
🌸 **Flower**
🔐 **TenSEAL**
⚛️ **React**
📊 **Recharts**

</p>

---

<p align="center">

# 🏥 FedMed

### **Collaborative Intelligence. Private Data. Better Healthcare.**

</p>
