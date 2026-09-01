
from fastapi import FastAPI, Request
import joblib
import numpy as np

app = FastAPI(
    title="Iris ML Prediction API",
    description="Containerized Machine Learning API using FastAPI",
    version="1.0"
)

# Load trained model
model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {
        "message": "ML Model API is running",
        "model": "Random Forest",
        "dataset": "Iris"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    features = body["features"]

    data = np.array(features).reshape(1, -1)

    prediction = model.predict(data)
    probabilities = model.predict_proba(data)

    return {
        "prediction": int(prediction[0]),
        "probability": float(max(probabilities[0]))
    }
