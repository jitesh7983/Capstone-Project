import os
import json
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import logging

# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("--- EcoVision Classification Training Pipeline Started ---")

# --------------------------------------------------
# 1. Load Dataset  (path relative to this script)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "air_quality.csv")

try:
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna()
    logger.info(f"Dataset loaded successfully. Rows: {len(data)}")
except FileNotFoundError:
    logger.error(f"Dataset not found at: {DATASET_PATH}")
    raise SystemExit(1)

# --------------------------------------------------
# 2. Convert AQI Value → Category
# --------------------------------------------------

def categorize_aqi(aqi: float) -> str:
    """Map a numeric AQI value to its WHO/CPCB category label."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

data["AQI_Category"] = data["AQI"].apply(categorize_aqi)

logger.info("AQI categories assigned.")
logger.info(f"Category distribution:\n{data['AQI_Category'].value_counts()}")

# --------------------------------------------------
# 3. Feature Selection
# --------------------------------------------------

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

X = data[FEATURES]
y = data["AQI_Category"]

# --------------------------------------------------
# 4. Encode Labels
# --------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

logger.info(f"Classes: {label_encoder.classes_.tolist()}")

# --------------------------------------------------
# 5. Train-Test Split  (stratified to handle imbalance)
# --------------------------------------------------

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded          # ← keeps class proportions in both splits
)

# --------------------------------------------------
# 6. Feature Scaling  (fit ONLY on training data)
# --------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# --------------------------------------------------
# 7. Model Definitions
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM":                 SVC(probability=True),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    # FIX: removed deprecated use_label_encoder parameter
    "XGBoost":             XGBClassifier(eval_metric="mlogloss", random_state=42)
}

results        = []
metrics_summary = {}

best_model  = None
best_f1     = 0.0          # FIX: use F1 (weighted) instead of raw accuracy

logger.info("\nStarting Model Comparison...\n")

# --------------------------------------------------
# 8. Model Training, Cross-Validation & Evaluation
# --------------------------------------------------

for name, model in models.items():

    # --- Train ---
    model.fit(X_train, y_train)

    # --- Hold-out metrics ---
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # --- 5-fold Cross-Validation on training data (F1 weighted) ---
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="f1_weighted"
    )

    results.append([name, acc, prec, rec, f1,
                    cv_scores.mean(), cv_scores.std()])

    metrics_summary[name] = {
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1_score":  float(f1),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std":  float(cv_scores.std())
    }

    logger.info(
        f"{name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} "
        f"| CV-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

    # FIX: select best model by weighted F1, not accuracy
    if f1 > best_f1:
        best_f1    = f1
        best_model = model

best_name = best_model.__class__.__name__
logger.info(f"\nBest Model (by F1): {best_name}  —  F1={best_f1:.4f}")

# --------------------------------------------------
# 9. Save Best Model & Artifacts
# --------------------------------------------------

OUTPUT_DIR = BASE_DIR          # saves alongside the script inside /backend

joblib.dump(best_model,    os.path.join(OUTPUT_DIR, "classification_model.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
joblib.dump(scaler,        os.path.join(OUTPUT_DIR, "scaler.pkl"))

logger.info("Model artifacts saved.")

# Save metrics JSON for React dashboard
metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_summary, f, indent=4)

logger.info(f"Metrics saved → {metrics_path}")

# --------------------------------------------------
# 10. Model Comparison Table
# --------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall",
             "F1 Score", "CV F1 Mean", "CV F1 Std"]
)

logger.info(f"\nModel Comparison Table:\n{results_df.to_string(index=False)}")

# --------------------------------------------------
# 11. Confusion Matrix
# --------------------------------------------------

y_best_pred = best_model.predict(X_test)
cm          = confusion_matrix(y_test, y_best_pred)

# FIX: save in correct labelled format for React dashboard
matrix_output = {
    "labels": label_encoder.classes_.tolist(),
    "values": cm.tolist()
}
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.json")
with open(cm_path, "w") as f:
    json.dump(matrix_output, f)

# Visualization for report
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

logger.info("Confusion matrix exported successfully.")
logger.info("--- Classification Training Complete ---")