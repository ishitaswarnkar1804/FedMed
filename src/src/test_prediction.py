import pandas as pd
import joblib


model = joblib.load(
    "models/fedmed_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# Take one test record
X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

sample = X_test.iloc[0:1]


prediction = model.predict(sample)

probability = model.predict_proba(sample).max()


print("========== FedMed Prediction ==========")

print(
    "Prediction:",
    prediction[0]
)

print(
    "Confidence:",
    round(probability * 100, 2),
    "%"
)