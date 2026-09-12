import joblib
import pandas as pd


MODEL_PATH = "models/fedmed_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def predict(input_data):

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Convert input to DataFrame
    data = pd.DataFrame([input_data])

    # Scale input
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)[0]

    # Probability
    probability = model.predict_proba(data_scaled).max()

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }