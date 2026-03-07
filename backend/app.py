from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# =========================
# Load Models
# =========================

regression_model = joblib.load("model.pkl")
classification_model = joblib.load("classification_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

print("Models loaded successfully 🚀")

# =========================
# Home Route
# =========================

@app.route("/")
def home():
    return "EcoVision Hybrid AQI System Running 🚀"

# =========================
# Prediction API
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    try:

        data = request.get_json()

        features = np.array([[
            float(data["PM2.5"]),
            float(data["PM10"]),
            float(data["NO2"]),
            float(data["SO2"]),
            float(data["CO"]),
            float(data["O3"])
        ]])

        # Apply scaling
        scaled_features = scaler.transform(features)

        # Regression prediction
        predicted_aqi = regression_model.predict(scaled_features)[0]

        # Classification prediction
        predicted_class = classification_model.predict(scaled_features)[0]
        predicted_category = label_encoder.inverse_transform([predicted_class])[0]

        return jsonify({
            "Predicted_AQI": round(float(predicted_aqi), 2),
            "Predicted_Category": predicted_category
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# Model Metrics API
# =========================

@app.route("/model-metrics", methods=["GET"])
def model_metrics():
    try:

        with open("model_metrics.json", "r") as f:
            metrics = json.load(f)

        # Convert dictionary to list
        metrics_list = []

        for model_name, values in metrics.items():
            metrics_list.append({
                "model": model_name,
                "accuracy": values["accuracy"],
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1_score"]
            })

        return jsonify(metrics_list)

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# Run Server
# =========================

if __name__ == "__main__":
    app.run(debug=True)