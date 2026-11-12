import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "pm25"

# Same features for every model. No station id, no coordinates (spatial CV rule).
NUMERIC_FEATURES = [
    "AOD", "NO2", "CO", "O3",
    "temperature_celsius", "pressure_mb", "wind_u", "wind_v", "wind_speed",
    "doy_sin", "doy_cos", "month", "dayofweek",
    "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll3_mean",
    "AOD_lag1", "AOD_lag2", "AOD_diff1", "temp_lag1",
]
INDICATOR_FEATURES = ["NO2_missing", "CO_missing", "O3_missing"]
ALL_FEATURES = NUMERIC_FEATURES + INDICATOR_FEATURES


def load_data(path="data/features.csv"):
    """Full dataset indexed by row_id (the same ids are used in the fold files)."""
    return pd.read_csv(path, parse_dates=["date"]).set_index("row_id")


def load_folds(name):
    """name: 'temporal', 'spatial' or 'standard'."""
    with open(f"data/folds_{name}.json") as f:
        return json.load(f)["folds"]


def load_test_ids():
    with open("data/test_row_ids.json") as f:
        return json.load(f)


def build_pipeline(model, scale=True):
    """Median imputation (+ optional scaling) fitted on the training fold only."""
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(steps), NUMERIC_FEATURES),
        ("ind", "passthrough", INDICATOR_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", model)])