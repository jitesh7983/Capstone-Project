import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Regression models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Loading dataset...")

# Load dataset
data = pd.read_csv("D:/ECOVISION/dataset/air_quality.csv")

# Remove missing values
data = data.dropna()

# =========================
# Create AQI Category
# =========================

def get_aqi_category(aqi):

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

# =========================
# Features and Targets
# =========================

features = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

X = data[features]

y_reg = data["AQI"]
y_class = data["AQI_Category"]

# Encode AQI categories
encoder = LabelEncoder()
y_class_encoded = encoder.fit_transform(y_class)

# =========================
# Train Test Split
# =========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_class_encoded, test_size=0.2, random_state=42
)

print("\n==============================")
print("Regression Model Training")
print("==============================")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_reg_train)

lr_pred = lr.predict(X_test)

print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_reg_test, lr_pred))
print("R2:", r2_score(y_reg_test, lr_pred))

# Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)

rf_reg.fit(X_train, y_reg_train)

rf_pred = rf_reg.predict(X_test)

print("\nRandom Forest Regressor")
print("MAE:", mean_absolute_error(y_reg_test, rf_pred))
print("R2:", r2_score(y_reg_test, rf_pred))

xgb_reg = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_reg.fit(X_train, y_reg_train)

xgb_pred = xgb_reg.predict(X_test)

print("\nXGBoost Regressor")
print("MAE:", mean_absolute_error(y_reg_test, xgb_pred))
print("R2:", r2_score(y_reg_test, xgb_pred))
print("\n==============================")
print("Classification Model Training")
print("==============================")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

best_model = None
best_accuracy = 0

metrics_results = {}

for name, model in models.items():

    model.fit(X_train_c, y_train_c)

    pred = model.predict(X_test_c)

    acc = accuracy_score(y_test_c, pred)
    prec = precision_score(y_test_c, pred, average="weighted")
    rec = recall_score(y_test_c, pred, average="weighted")
    f1 = f1_score(y_test_c, pred, average="weighted")

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    # Save metrics
    metrics_results[name] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }

    # Select best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Classification Model:", best_model.__class__.__name__)
print("Best Accuracy:", best_accuracy)

print("\nSaving models...")

# Save models
joblib.dump(xgb_reg, "model.pkl")
joblib.dump(best_model, "classification_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Models saved successfully!")

# =========================
# Save Model Metrics
# =========================

with open("model_metrics.json", "w") as f:
    json.dump(metrics_results, f, indent=4)

print("Model metrics saved successfully!")