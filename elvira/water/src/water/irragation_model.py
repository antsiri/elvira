import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from water_mk1.feature_engineering import load_data, build_features, get_feature_columns

OPTIMAL_MIN = -13.8  # bar  (equivale a -1.4 MPa)
OPTIMAL_MAX = -11.8  # bar  (equivale a -1.2 MPa)

HORIZON   = 6
LAG_HOURS = 48   # aumentato da 24 a 48

def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in range(1, LAG_HOURS + 1):
        df[f"wp_lag_{lag}h"] = df["water_potential_avg"].shift(lag)

    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"moist_lag_{lag}h"] = df["moist_soil_avg"].shift(lag)

    # Feature temporali con ciclicità (sin/cos cattura meglio periodicità)
    df["hour_sin"]  = np.sin(2 * np.pi * df["ts"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["ts"].dt.hour / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["ts"].dt.dayofyear / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["ts"].dt.dayofyear / 365)
    df["month"]     = df["ts"].dt.month

    # Differenze (trend del water potential)
    df["wp_diff_1h"]  = df["water_potential_avg"].diff(1)
    df["wp_diff_6h"]  = df["water_potential_avg"].diff(6)
    df["wp_diff_24h"] = df["water_potential_avg"].diff(24)

    df["target_wp"] = df["water_potential_avg"].shift(-HORIZON)

    return df.dropna()


def train(df: pd.DataFrame):
    df = build_lag_features(df)

    feature_cols = (
        get_feature_columns() +
        [f"wp_lag_{lag}h" for lag in range(1, LAG_HOURS + 1)] +
        [f"moist_lag_{lag}h" for lag in [1, 3, 6, 12, 24, 48]] +
        ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"] +
        ["wp_diff_1h", "wp_diff_6h", "wp_diff_24h"]
    )

    X = df[feature_cols]
    y = df["target_wp"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    early_stopping_rounds=50,  # ← qui
    random_state=42,
)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\nMAE:  {mae:.4f} bar")
    print(f"R²:   {r2:.4f}")

    model_path = Path(__file__).resolve().parents[3] / "data" / "irrigation_model.pkl"
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    print(f"Modello salvato in: {model_path}")

    return model, feature_cols


def predict(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    df = build_lag_features(df)
    X  = df[feature_cols]

    df["predicted_wp"] = model.predict(X)
    df["irrigate"]     = df["predicted_wp"] < OPTIMAL_MIN
    df["status"]       = df["predicted_wp"].apply(
        lambda v: "🔴 Irrigare" if v < OPTIMAL_MIN
        else ("✅ Ottimale" if OPTIMAL_MIN <= v <= OPTIMAL_MAX else "💧 Troppo umido")
    )

    return df[["ts", "water_potential_avg", "predicted_wp", "status", "irrigate"]]

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[3] / "data" / "two_years_history.csv"
    df = load_data(str(data_path))
    df = build_features(df)

    print("Addestrando il modello...")
    model, feature_cols = train(df)

    print("\nUltime predizioni:")
    results = predict(df.tail(100), model, feature_cols)
    print(results.tail(10).to_string(index=False))

