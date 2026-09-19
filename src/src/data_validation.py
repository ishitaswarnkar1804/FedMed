import pandas as pd
import os

DATA_PATH = "data/raw/medical_data.csv"


def validate_data(file_path):
    if not os.path.exists(file_path):
        print("❌ Dataset not found.")
        return False

    df = pd.read_csv(file_path)

    print("===== FedMed Data Validation =====")

    # Check empty dataset
    if df.empty:
        print("❌ Dataset is empty.")
        return False

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Check missing values
    missing = df.isnull().sum().sum()

    if missing > 0:
        print(f"⚠️ Missing values found: {missing}")
    else:
        print("✅ No missing values.")

    # Check duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        print(f"⚠️ Duplicate rows found: {duplicates}")
    else:
        print("✅ No duplicate rows.")

    # Check numeric columns
    numeric_columns = df.select_dtypes(include="number").columns

    if len(numeric_columns) == 0:
        print("❌ No numeric features found.")
        return False

    print(f"✅ Numeric features: {len(numeric_columns)}")

    # Check infinite values
    infinite_values = df[numeric_columns].isin(
        [float("inf"), float("-inf")]
    ).sum().sum()

    if infinite_values > 0:
        print(f"❌ Infinite values found: {infinite_values}")
        return False
    else:
        print("✅ No infinite values.")

    print("\n✅ Data validation completed successfully.")
    return True


if __name__ == "__main__":
    validate_data(DATA_PATH)