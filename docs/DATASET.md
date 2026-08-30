# Dataset Guide

FedMed uses the **Medical Segmentation Decathlon Task01_BrainTumour** dataset for brain tumor segmentation.

## Overview

| Property | Value |
|---|---|
| Task | Multi-modal brain tumor segmentation |
| Modalities | FLAIR, T1, T1ce, T2 |
| Format | NIfTI (.nii.gz) |
| Training cases | ~484 |
| Source | [Medical Decathlon](http://medicaldecathlon.com/) |

## Hospital Silos

Patients are split into three non-overlapping silos using a stable hash of patient ID:

- `hospital_a` — ~161 patients
- `hospital_b` — ~161 patients
- `hospital_c` — ~162 patients

Generate manifests:

```bash
python scripts/prepare_silos.py
```

Manifests are written to `data/msd/silos/*.json`.

## Download (MONAI)

In Google Colab or locally with sufficient disk (~2 GB):

```python
from monai.apps import DecathlonDataset

DecathlonDataset(
    root_dir="data/msd",
    task="Task01_BrainTumour",
    section="training",
    download=True,
)
```

## Colab GPU Tips

- Mount Google Drive to cache downloads between sessions
- Use patch size `(96, 96, 96)` for T4 GPU
- Set `num_workers=2` in DataLoader
- Enable mixed precision in config (`training.amp: true`)

## Synthetic Smoke Tests

For local/Docker smoke tests without downloading MSD, enable synthetic data:

```yaml
data:
  synthetic:
    enabled: true
```

Synthetic volumes mimic multi-channel MRI patches with a spherical tumor region.
