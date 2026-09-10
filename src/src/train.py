import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier


X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).values.ravel()


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

print("Training model...")

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/fedmed_model.pkl"
)

print("Model trained successfully.")
print("Model saved to models/fedmed_model.pkl")
