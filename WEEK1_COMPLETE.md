# 🧠 FedMed

<p align="center">
  <img src="https://img.shields.io/badge/FedMed-Privacy--Preserving%20Medical%20AI-6C63FF?style=for-the-badge" alt="FedMed">
</p>

<p align="center">
  <strong>Federated Learning for Privacy-Preserving Brain Tumor Segmentation</strong>
</p>

<p align="center">
  A cross-silo federated learning framework that enables multiple hospitals
  to collaboratively train a 3D U-Net without centralizing sensitive MRI data.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-2C8EBB?style=flat-square)
![Flower](https://img.shields.io/badge/Flower-1.8%2B-FF6F00?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-6%20Passed-success?style=flat-square)
![Week](https://img.shields.io/badge/Week%201-Complete-6C63FF?style=flat-square)

</p>

---

## 🌟 Overview

Medical institutions often cannot freely share patient MRI scans because of
privacy, security, and regulatory constraints.

**FedMed solves this problem using Federated Learning.**

Instead of moving MRI data to a central server:

> 🏥 **Data stays inside the hospital.**  
> 🧠 **The model travels between hospitals.**  
> 🔄 **Knowledge is aggregated centrally.**

The project uses a **3D U-Net** for brain tumor segmentation and
**Flower** for the federated learning infrastructure.

---

# 🏗️ System Architecture

```mermaid
flowchart TB

    SERVER["🧠 FEDMED CENTRAL SERVER
    <br/>
    Flower ServerApp
    <br/>
    Global 3D U-Net"]

    LINK["🌐 FLOWER SUPERLINK
    <br/>
    Central Communication Layer"]

    H1["🏥 HOSPITAL 1
    <br/>
    SuperNode :9094"]

    H2["🏥 HOSPITAL 2
    <br/>
    SuperNode :9095"]

    H3["🏥 HOSPITAL 3
    <br/>
    SuperNode :9096"]

    D1[("🔒 Private MRI Data")]
    D2[("🔒 Private MRI Data")]
    D3[("🔒 Private MRI Data")]

    C1["⚙️ Local Client"]
    C2["⚙️ Local Client"]
    C3["⚙️ Local Client"]

    SERVER <--> LINK

    LINK <--> H1
    LINK <--> H2
    LINK <--> H3

    H1 --> C1
    H2 --> C2
    H3 --> C3

    D1 --> C1
    D2 --> C2
    D3 --> C3

    C1 -. "Model Updates" .-> SERVER
    C2 -. "Model Updates" .-> SERVER
    C3 -. "Model Updates" .-> SERVER

    style SERVER fill:#6C63FF,color:#fff,stroke:#4B43B6,stroke-width:3px
    style LINK fill:#24292F,color:#fff,stroke:#000,stroke-width:2px

    style H1 fill:#E8F1FF,stroke:#3776AB,stroke-width:2px
    style H2 fill:#E8F1FF,stroke:#3776AB,stroke-width:2px
    style H3 fill:#E8F1FF,stroke:#3776AB,stroke-width:2px

    style D1 fill:#FFF4E5,stroke:#FF9800
    style D2 fill:#FFF4E5,stroke:#FF9800
    style D3 fill:#FFF4E5,stroke:#FF9800
