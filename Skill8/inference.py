import json
import joblib
import numpy as np

def model_fn(model_dir):
    model_path = f"{model_dir}/iris_model.joblib"
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError("Only application/json is supported")

    data = json.loads(request_body)

    if isinstance(data, dict):
        features = data["features"]
    else:
        features = data

    return np.array(features).reshape(1, -1)

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    return {
        "prediction": int(prediction[0]),
        "probabilities": probabilities[0].tolist()
    }

def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError("Only application/json is supported")

    return json.dumps(prediction), accept
