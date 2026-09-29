
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("iris_model.joblib")

@app.route("/")
def home():
    return jsonify({"message": "Iris ML API is running"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]])

    prediction = model.predict(features)[0]

    species = ["setosa", "versicolor", "virginica"][prediction]

    return jsonify({
        "prediction": int(prediction),
        "species": species
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
