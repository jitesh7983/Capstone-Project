import os
import json
import logging
import numpy as np
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App Initialization
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
CLF_PATH     = os.path.join(BASE_DIR, "classification_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.pkl")

METRICS_PATH      = os.path.join(BASE_DIR, "model_metrics.json")
FEAT_IMP_PATH     = os.path.join(BASE_DIR, "feature_importance.json")
REG_METRICS_PATH  = os.path.join(BASE_DIR, "regression_metrics.json")

REQUIRED_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

VALID_RANGES = {
    "PM2.5": (0, 500),
    "PM10":  (0, 600),
    "NO2":   (0, 400),
    "SO2":   (0, 500),
    "CO":    (0, 50),
    "O3":    (0, 400)
}

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities freely.",
    "Satisfactory": "Acceptable air quality. Sensitive individuals limit exertion.",
    "Moderate": "Sensitive groups should reduce outdoor activity.",
    "Poor": "Everyone may experience health effects. Wear a mask.",
    "Very Poor": "Avoid outdoor activity. Keep windows closed.",
    "Severe": "Health emergency. Stay indoors."
}

# --------------------------------------------------
# Load Main Models
# --------------------------------------------------

def load_artifacts():
    try:
        reg_model = joblib.load(MODEL_PATH)
        clf_model = joblib.load(CLF_PATH)
        encoder   = joblib.load(ENCODER_PATH)
        scaler    = joblib.load(SCALER_PATH)

        logger.info("Main models loaded successfully.")
        return reg_model, clf_model, encoder, scaler

    except Exception as e:
        logger.error(f"Failed to load main models: {e}")
        return None, None, None, None

regression_model, classification_model, label_encoder, scaler = load_artifacts()

# --------------------------------------------------
# Load Forecast Models (NEW)
# --------------------------------------------------

HORIZONS = [1, 2, 3, 4, 5, 6, 7]

forecast_models = {}
forecast_scaler = None

try:
    forecast_scaler = joblib.load(
        os.path.join(BASE_DIR, "forecast_scaler.pkl")
    )

    for h in HORIZONS:
        forecast_models[h] = joblib.load(
            os.path.join(BASE_DIR, f"forecast_h{h}.pkl")
        )

    logger.info("Forecast models loaded successfully.")

except Exception as e:
    logger.warning(f"Forecast models not loaded: {e}")

# --------------------------------------------------
# Input Validation
# --------------------------------------------------

def parse_and_validate(data):
    values = []

    for f in REQUIRED_FEATURES:

        if f not in data or str(data[f]).strip() == "":
            return None, f"Missing value for '{f}'"

        try:
            val = float(data[f])
        except:
            return None, f"{f} must be numeric"

        low, high = VALID_RANGES[f]
        if not (low <= val <= high):
            return None, f"{f} out of range [{low}, {high}]"

        values.append(val)

    return values, None

# --------------------------------------------------
# AQI Category
# --------------------------------------------------

def get_category(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "system": "EcoVision AI API",
        "version": "2.0"
    })

# ---------------- Prediction ----------------

@app.route("/predict", methods=["POST"])
def predict():

    if regression_model is None:
        return jsonify({"error": "Models not loaded"}), 500

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON required"}), 400

    input_values, err = parse_and_validate(body)
    if err:
        return jsonify({"error": err}), 400

    try:
        features = np.array([input_values])
        scaled   = scaler.transform(features)

        predicted_aqi = float(regression_model.predict(scaled)[0])
        predicted_aqi = round(max(0, predicted_aqi), 2)

        class_idx = classification_model.predict(scaled)[0]
        category  = label_encoder.inverse_transform([class_idx])[0]

        return jsonify({
            "success": True,
            "predicted_aqi": predicted_aqi,
            "predicted_category": category,
            "health_advice": HEALTH_ADVICE.get(category, "")
        })

    except Exception as e:
        logger.error(e)
        return jsonify({"error": "Prediction failed"}), 500

# ---------------- Forecast (NEW) ----------------

@app.route("/forecast", methods=["POST"])
def forecast():

    if not forecast_models:
        return jsonify({"error": "Forecast models not loaded"}), 500

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "JSON required"}), 400

    input_values, err = parse_and_validate(body)
    if err:
        return jsonify({"error": err}), 400

    recent_aqi = body.get("recent_aqi", [])

    if len(recent_aqi) < 3:
        return jsonify({"error": "Provide at least 3 AQI values"}), 400

    try:
        aqi_lag1 = float(recent_aqi[-1])
        aqi_lag2 = float(recent_aqi[-2])
        aqi_lag3 = float(recent_aqi[-3])

        roll3 = np.mean(recent_aqi[-3:])
        roll7 = np.mean(recent_aqi[-7:]) if len(recent_aqi) >= 7 else np.mean(recent_aqi)

        features = np.array([[
            *input_values,
            aqi_lag1, aqi_lag2, aqi_lag3,
            roll3, roll7
        ]])

        scaled = forecast_scaler.transform(features)

        predictions = []

        for h in HORIZONS:
            pred = float(forecast_models[h].predict(scaled)[0])
            pred = round(max(0, pred), 1)

            cat = get_category(pred)

            predictions.append({
                "day": h,
                "label": f"Day +{h}",
                "aqi": pred,
                "category": cat,
                "advice": HEALTH_ADVICE.get(cat, "")
            })

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:
        logger.error(e)
        return jsonify({"error": "Forecast failed"}), 500

# ---------------- Other APIs ----------------

@app.route("/feature-importance")
def feature_importance():
    if not os.path.exists(FEAT_IMP_PATH):
        return jsonify({"error": "Not found"}), 404

    return jsonify(json.load(open(FEAT_IMP_PATH)))

@app.route("/model-metrics", methods=["GET"])
def model_metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics file not found"}), 404

    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        
        metrics_list = []

        for name, vals in metrics.items():
            metrics_list.append({
                "model": name,
                "accuracy": float(vals.get("accuracy", 0)),
                "precision": float(vals.get("precision", 0)),
                "recall": float(vals.get("recall", 0)),
                "f1_score": float(
                    vals.get("f1_score", vals.get("f1", 0))
                ),
                "cv_f1_mean": float(vals.get("cv_f1_mean", 0)),
                "cv_f1_std": float(vals.get("cv_f1_std", 0))
            })

        return jsonify(metrics_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/regression-metrics")
def regression_metrics():
    if not os.path.exists(REG_METRICS_PATH):
        return jsonify({"error": "Not found"}), 404

    return jsonify(json.load(open(REG_METRICS_PATH)))

@app.route("/health-info")
def health_info():
    return jsonify(HEALTH_ADVICE)

# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)