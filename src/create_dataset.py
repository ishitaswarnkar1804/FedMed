from sklearn.datasets import load_breast_cancer
import pandas as pd
import os


def create_dataset():
    dataset = load_breast_cancer()

    df = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    df["target"] = dataset.target

    os.makedirs("data/raw", exist_ok=True)

    output_path = "data/raw/medical_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset created successfully: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    create_dataset()