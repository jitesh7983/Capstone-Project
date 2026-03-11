import os
import json
import logging

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import xgboost as xgb

# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("--- EcoVision: Model Training Pipeline Started ---")

# --------------------------------------------------
# 1. Load Dataset  (path relative to this script)
# --------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "air_quality.csv")

try:
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna()
    logger.info(f"Dataset loaded. Rows: {len(data)}")
except FileNotFoundError:
    logger.error(f"Dataset not found at: {DATASET_PATH}")
    raise SystemExit(1)

# --------------------------------------------------
# 2. AQI Category Logic
# --------------------------------------------------

def get_aqi_category(aqi: float) -> str:
    """Map a numeric AQI value to its CPCB category label."""
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

data["AQI_Category"] = data["AQI"].apply(get_aqi_category)

logger.info("AQI categories assigned.")
logger.info(f"Category distribution:\n{data['AQI_Category'].value_counts()}")

# --------------------------------------------------
# 3. Features and Targets
# --------------------------------------------------

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

X       = data[FEATURES]
y_reg   = data["AQI"]
y_class = data["AQI_Category"]

# Encode categories
encoder        = LabelEncoder()
y_class_encoded = encoder.fit_transform(y_class)

logger.info(f"Classes: {encoder.classes_.tolist()}")

# --------------------------------------------------
# 4. Train-Test Split  (stratified for classification)
# --------------------------------------------------

X_train_raw, X_test_raw, \
y_train_reg, y_test_reg, \
y_train_c,   y_test_c = train_test_split(
    X,
    y_reg,
    y_class_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_class_encoded    # keeps class balance in both splits
)

# --------------------------------------------------
# 5. Feature Scaling  (fit ONLY on training data)
# --------------------------------------------------

scaler   = StandardScaler()
X_train  = scaler.fit_transform(X_train_raw)
X_test   = scaler.transform(X_test_raw)

# --------------------------------------------------
# 6. Regression Training  (XGBoost + GridSearchCV)
# --------------------------------------------------

logger.info("\n[Phase 1] Regression Training — XGBoost with GridSearchCV")

xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

param_grid = {
    "n_estimators":  [100, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth":     [4, 6]
}

grid_reg = GridSearchCV(
    xgb_reg,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_reg.fit(X_train, y_train_reg)

final_reg_model = grid_reg.best_estimator_

logger.info(f"Best Regressor Params: {grid_reg.best_params_}")

reg_pred = final_reg_model.predict(X_test)

r2   = r2_score(y_test_reg, reg_pred)
mae  = mean_absolute_error(y_test_reg, reg_pred)
rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))

logger.info(f"Regression  R²  : {r2:.4f}")
logger.info(f"Regression  MAE : {mae:.2f}")
logger.info(f"Regression  RMSE: {rmse:.2f}")

# Save regression metrics separately
reg_metrics = {
    "r2_score": float(r2),
    "mae":      float(mae),
    "rmse":     float(rmse),
    "best_params": grid_reg.best_params_
}
with open(os.path.join(BASE_DIR, "regression_metrics.json"), "w") as f:
    json.dump(reg_metrics, f, indent=4)

# --------------------------------------------------
# 7. Classification Model Comparison
# --------------------------------------------------

logger.info("\n[Phase 2] Classification Training & Comparison")

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM":                 SVC(probability=True),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    # FIX: removed deprecated use_label_encoder parameter
    "XGBoost":             xgb.XGBClassifier(
                               eval_metric="mlogloss",
                               random_state=42
                           )
}

best_clf_model  = None
best_f1         = 0.0          # FIX: rank by weighted F1, not raw accuracy
metrics_results = {}
comparison_rows = []

for name, model in clf_models.items():

    # --- Train ---
    model.fit(X_train, y_train_c)
    pred = model.predict(X_test)

    # --- Hold-out metrics ---
    acc  = accuracy_score(y_test_c, pred)
    prec = precision_score(y_test_c, pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test_c, pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_test_c, pred,        average="weighted", zero_division=0)

    # --- 5-fold Cross-Validation (F1 weighted) ---
    cv_scores = cross_val_score(
        model, X_train, y_train_c,
        cv=5, scoring="f1_weighted"
    )

    metrics_results[name] = {
        "accuracy":   float(acc),
        "precision":  float(prec),
        "recall":     float(rec),
        "f1_score":   float(f1),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std":  float(cv_scores.std())
    }

    comparison_rows.append([name, acc, prec, rec, f1,
                             cv_scores.mean(), cv_scores.std()])

    logger.info(
        f"{name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} "
        f"| CV-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )

    # FIX: select best model by weighted F1, not accuracy
    if f1 > best_f1:
        best_f1        = f1
        best_clf_model = model

best_name = type(best_clf_model).__name__
logger.info(f"\nBest Classifier (by F1): {best_name}  —  F1={best_f1:.4f}")

# Model comparison table
comp_df = pd.DataFrame(
    comparison_rows,
    columns=["Model", "Accuracy", "Precision", "Recall",
             "F1 Score", "CV F1 Mean", "CV F1 Std"]
)
logger.info(f"\nComparison Table:\n{comp_df.to_string(index=False)}")

# --------------------------------------------------
# 8. Confusion Matrix Export  (fixed format for React)
# --------------------------------------------------

cm = confusion_matrix(y_test_c, best_clf_model.predict(X_test))

# FIX: export as labelled dict, not raw DataFrame JSON
matrix_output = {
    "labels": encoder.classes_.tolist(),
    "values": cm.tolist()
}
cm_path = os.path.join(BASE_DIR, "confusion_matrix.json")
with open(cm_path, "w") as f:
    json.dump(matrix_output, f, indent=4)

logger.info(f"Confusion matrix saved → {cm_path}")

# Confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True, fmt="d", cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))

# --------------------------------------------------
# 9. Feature Importance  (from regression model)
# --------------------------------------------------

importances = final_reg_model.feature_importances_

feat_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feat_path = os.path.join(BASE_DIR, "feature_importance.json")
feat_df.to_json(feat_path, orient="records")

logger.info(f"Feature importance saved → {feat_path}")

# Feature importance plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Feature Importance — AQI Prediction (XGBoost Regressor)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "feature_importance.png"))

# --------------------------------------------------
# 10. Save Models & All Artifacts
# --------------------------------------------------

logger.info("\nSaving models and artifacts...")

joblib.dump(final_reg_model,  os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(best_clf_model,   os.path.join(BASE_DIR, "classification_model.pkl"))
joblib.dump(encoder,          os.path.join(BASE_DIR, "label_encoder.pkl"))
joblib.dump(scaler,           os.path.join(BASE_DIR, "scaler.pkl"))

with open(os.path.join(BASE_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics_results, f, indent=4)

with open(os.path.join(BASE_DIR, "best_model.txt"), "w") as f:
    f.write(best_name)

logger.info("--- EcoVision Training Pipeline Complete ---")