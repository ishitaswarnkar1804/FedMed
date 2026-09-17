import pandas as pd
import hashlib


def anonymize_id(patient_id):
    """Convert patient ID into a hashed identifier."""
    return hashlib.sha256(str(patient_id).encode()).hexdigest()[:12]


def anonymize_dataset(input_file, output_file):
    """Remove direct identifiers and anonymize patient IDs."""

    df = pd.read_csv(input_file)

    # Remove common direct identifiers if present
    identifiers = ["name", "email", "phone", "address"]

    for column in identifiers:
        if column in df.columns:
            df.drop(columns=[column], inplace=True)

    # Hash patient ID if available
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].apply(anonymize_id)

    df.to_csv(output_file, index=False)

    print("Privacy preprocessing completed.")
    print(f"Anonymized dataset saved to: {output_file}")


if __name__ == "__main__":
    anonymize_dataset(
        "data/raw/medical_data.csv",
        "data/raw/anonymized_medical_data.csv"
    )