# Privacy and Homomorphic Encryption

FedMed implements a privacy-preserving federated learning workflow aligned with HIPAA/GDPR principles: **raw patient MRI never leaves the hospital environment**.

## Threat Model (Demo)

| Asset | Location | Protected from |
|---|---|---|
| Raw MRI volumes | Hospital Colab/runtime | Central server, other hospitals |
| Local weight deltas | Encrypted in transit | Passive eavesdroppers |
| Aggregated global model | Flower server | Individual hospital updates (HE mode) |

## Workflow

1. Each hospital trains a MONAI 3D U-Net on its private silo.
2. The client computes **weight deltas** (post-training minus pre-training).
3. In `he_ckks` mode, deltas are encrypted with TenSEAL CKKS before upload.
4. The server homomorphically aggregates encrypted deltas without decrypting individual updates.
5. The aggregated result is decrypted to produce the next global model.

## TenSEAL Configuration

Generate keys once:

```bash
python scripts/generate_he_keys.py
```

Files created:

- `keys/public.tenseal` — used by clients for encryption; server for homomorphic addition
- `keys/secret.tenseal` — used by server to decrypt the aggregated update (demo setup)

## Performance Mitigations

Full U-Net encryption is expensive. FedMed v1 mitigates this by:

- Encrypting **deltas only**, not full weights
- Chunking parameter vectors (`privacy.chunk_size: 8192`)
- Starting with `privacy.mode: plain` to validate the FL loop

## Mode Toggle

```yaml
privacy:
  mode: plain      # development / baseline
  mode: he_ckks    # homomorphic encryption enabled
```

## Limitations (v1)

- Single-key CKKS (not multi-key xMK-CKKS)
- Server holds decryption key in demo deployment — production would use threshold decryption or client-side-only decryption
- PySyft is documented as an alternative research stack but not implemented here

## Compliance Narrative

FedMed demonstrates **data minimization**: only mathematically transformed model updates cross the network, not identifiable patient records. This supports research collaboration under strict healthcare privacy regimes.
