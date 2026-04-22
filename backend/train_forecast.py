import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

print("--- EcoVision: Forecast Model Training ---")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "air_quality.csv")

# ─── Load & Sort by Date ─────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
df = df.dropna()

# Sort by date so lag features make sense
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

# ─── Features & Lag Engineering ──────────────────────────────────────────────
POLL_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Add lag features (previous 1, 2, 3 days AQI)
df["AQI_lag1"] = df["AQI"].shift(1)
df["AQI_lag2"] = df["AQI"].shift(2)
df["AQI_lag3"] = df["AQI"].shift(3)

# Add rolling average (smooths noise)
df["AQI_roll3"] = df["AQI"].rolling(3).mean()
df["AQI_roll7"] = df["AQI"].rolling(7).mean()

# Features for forecasting
FORECAST_FEATURES = POLL_FEATURES + [
    "AQI_lag1", "AQI_lag2", "AQI_lag3",
    "AQI_roll3", "AQI_roll7"
]

# Forecast horizons (in days — since dataset is daily)
HORIZONS = [1, 2, 3, 4, 5, 6, 7]   # 1 day ahead ... 7 days ahead

# Create target columns for each horizon
for h in HORIZONS:
    df[f"AQI_next{h}"] = df["AQI"].shift(-h)

# Drop rows with NaN (from lag/shift operations)
df = df.dropna()

X = df[FORECAST_FEATURES]

# ─── Scale Features ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(BASE_DIR, "forecast_scaler.pkl"))

# ─── Train One Model Per Horizon ─────────────────────────────────────────────
forecast_metrics = {}

for h in HORIZONS:
    y = df[f"AQI_next{h}"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Horizon +{h}d → R²: {r2:.4f} | MAE: {mae:.2f}")

    forecast_metrics[f"horizon_{h}"] = {
        "r2":  round(float(r2),  4),
        "mae": round(float(mae), 2)
    }

    # Save model for this horizon
    joblib.dump(
        model,
        os.path.join(BASE_DIR, f"forecast_h{h}.pkl")
    )

# Save metrics
with open(os.path.join(BASE_DIR, "forecast_metrics.json"), "w") as f:
    json.dump(forecast_metrics, f, indent=4)

print("\n--- Forecast Training Complete ---")
print(f"Trained {len(HORIZONS)} models for horizons: {HORIZONS}")