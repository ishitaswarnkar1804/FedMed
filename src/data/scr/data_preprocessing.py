import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


DATA_PATH = "data/raw/medical_data.csv"
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


def preprocess_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Original shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save processed datasets
    pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    ).to_csv(
        f"{PROCESSED_DIR}/X_train.csv",
        index=False
    )

    pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    ).to_csv(
        f"{PROCESSED_DIR}/X_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{PROCESSED_DIR}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{PROCESSED_DIR}/y_test.csv",
        index=False
    )

    # Save scaler
    joblib.dump(
        scaler,
        f"{MODEL_DIR}/scaler.pkl"
    )

    print("Preprocessing completed successfully.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


if __name__ == "__main__":
    preprocess_data()