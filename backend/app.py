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

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
CLF_PATH       = os.path.join(BASE_DIR, "classification_model.pkl")
ENCODER_PATH   = os.path.join(BASE_DIR, "label_encoder.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "scaler.pkl")
METRICS_PATH   = os.path.join(BASE_DIR, "model_metrics.json")
FEAT_IMP_PATH  = os.path.join(BASE_DIR, "feature_importance.json")
REG_METRICS_PATH = os.path.join(BASE_DIR, "regression_metrics.json")

REQUIRED_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Valid physical ranges for each pollutant (µg/m³ or ppm)
VALID_RANGES = {
    "PM2.5": (0, 500),
    "PM10":  (0, 600),
    "NO2":   (0, 400),
    "SO2":   (0, 500),
    "CO":    (0, 50),
    "O3":    (0, 400)
}

# Health recommendations per AQI category
HEALTH_ADVICE = {
    "Good":        "Air quality is satisfactory. Enjoy outdoor activities freely.",
    "Satisfactory":"Acceptable air quality. Unusually sensitive individuals should limit prolonged exertion.",
    "Moderate":    "Sensitive groups (children, elderly, respiratory patients) should reduce outdoor activity.",
    "Poor":        "Everyone may experience health effects. Wear a mask outdoors and limit exertion.",
    "Very Poor":   "Health alert — avoid prolonged outdoor activity. Keep windows closed.",
    "Severe":      "Health emergency. Stay indoors with air purification. Seek medical advice if symptomatic."
}

# --------------------------------------------------
# Load ML Artifacts
# --------------------------------------------------

def load_artifacts():
    """Load all ML models and preprocessing objects at startup."""
    try:
        reg_model  = joblib.load(MODEL_PATH)
        clf_model  = joblib.load(CLF_PATH)
        encoder    = joblib.load(ENCODER_PATH)
        scaler     = joblib.load(SCALER_PATH)
        logger.info("All ML artifacts loaded successfully.")
        return reg_model, clf_model, encoder, scaler
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return None, None, None, None

regression_model, classification_model, label_encoder, scaler = load_artifacts()

# --------------------------------------------------
# Helper — validate & parse inputs
# --------------------------------------------------

def parse_and_validate(data: dict):
    """
    Returns (input_values: list[float], error_msg: str | None).
    Checks presence, numeric type, and physical range for every feature.
    """
    input_values = []

    for feature in REQUIRED_FEATURES:

        # --- Presence check ---
        if feature not in data or str(data[feature]).strip() == "":
            return None, f"Missing value for '{feature}'"

        # --- Numeric check ---
        try:
            value = float(data[feature])
        except (ValueError, TypeError):
            return None, f"'{feature}' must be a numeric value, got: {data[feature]!r}"

        # --- Range check ---
        low, high = VALID_RANGES[feature]
        if not (low <= value <= high):
            return None, (
                f"'{feature}' value {value} is outside the valid range "
                f"[{low}, {high}]"
            )

        input_values.append(value)

    return input_values, None

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    """Health-check endpoint."""
    return jsonify({
        "status":  "online",
        "system":  "EcoVision AI — AQI Prediction API",
        "version": "2.0.0",
        "models_loaded": regression_model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with pollutant readings and returns:
      - predicted_aqi       (float)
      - predicted_category  (str)
      - health_advice       (str)
      - confidence_scores   (dict)  — per-class probabilities
    """
    if regression_model is None:
        logger.error("Prediction attempted but models are not loaded.")
        return jsonify({"error": "Models not loaded. Check server logs."}), 500

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    # --- Validate inputs ---
    input_values, err = parse_and_validate(body)
    if err:
        logger.warning(f"Invalid input: {err}")
        return jsonify({"error": err}), 400

    try:
        features_array  = np.array([input_values])
        scaled_features = scaler.transform(features_array)

        # Regression — numeric AQI
        predicted_aqi = float(regression_model.predict(scaled_features)[0])
        predicted_aqi = max(0.0, round(predicted_aqi, 2))   # AQI can't be negative

        # Classification — category label
        class_index        = classification_model.predict(scaled_features)[0]
        predicted_category = label_encoder.inverse_transform([class_index])[0]

        # Per-class confidence scores (if model supports predict_proba)
        confidence_scores = {}
        if hasattr(classification_model, "predict_proba"):
            proba = classification_model.predict_proba(scaled_features)[0]
            confidence_scores = {
                label: round(float(prob), 4)
                for label, prob in zip(label_encoder.classes_, proba)
            }

        advice = HEALTH_ADVICE.get(predicted_category, "No advice available.")

        logger.info(
            f"Prediction — AQI: {predicted_aqi}, "
            f"Category: {predicted_category}"
        )

        return jsonify({
            "success":           True,
            "predicted_aqi":     predicted_aqi,
            "predicted_category": predicted_category,
            "health_advice":     advice,
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "Internal prediction error. Check server logs."}), 500


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Returns pollutant feature importances from the regression model."""
    if not os.path.exists(FEAT_IMP_PATH):
        return jsonify({"error": "Feature importance file not found. Run training first."}), 404

    try:
        with open(FEAT_IMP_PATH, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Feature importance read error: {e}")
        return jsonify({"error": "Could not read feature importance file."}), 500


@app.route("/model-metrics", methods=["GET"])
def model_metrics():
    """Returns classification metrics for all trained models."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics file not found. Run training first."}), 404

    try:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

        metrics_list = [
            {
                "model":      model_name,
                "accuracy":   round(v.get("accuracy",   0), 4),
                "precision":  round(v.get("precision",  0), 4),
                "recall":     round(v.get("recall",     0), 4),
                "f1_score":   round(v.get("f1_score",   0), 4),
                "cv_f1_mean": round(v.get("cv_f1_mean", 0), 4),
                "cv_f1_std":  round(v.get("cv_f1_std",  0), 4)
            }
            for model_name, v in metrics.items()
        ]

        return jsonify(metrics_list)

    except Exception as e:
        logger.error(f"Model metrics read error: {e}")
        return jsonify({"error": "Could not read metrics file."}), 500


@app.route("/regression-metrics", methods=["GET"])
def regression_metrics():
    """Returns regression performance metrics (R², MAE, RMSE)."""
    if not os.path.exists(REG_METRICS_PATH):
        return jsonify({"error": "Regression metrics file not found. Run training first."}), 404

    try:
        with open(REG_METRICS_PATH, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Regression metrics read error: {e}")
        return jsonify({"error": "Could not read regression metrics file."}), 500


@app.route("/health-info", methods=["GET"])
def health_info():
    """Returns the health advice mapping for all AQI categories."""
    return jsonify(HEALTH_ADVICE)


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))

    # FIX: debug mode driven by env variable — never True in production
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting EcoVision API on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)