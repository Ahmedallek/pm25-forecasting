import numpy as np
import pandas as pd

df = pd.read_csv("data/dataset_with_rowid.csv", parse_dates=["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# Calendar features (cyclical day-of-year)
doy = df["date"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

# Wind speed from components
df["wind_speed"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)

# Lags and rolling stats: per station, past values only, never across a gap.
# A lag of k days is valid only if the row k days earlier exists for the same station.
def add_lags(g):
    g = g.set_index("date")
    full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    out = pd.DataFrame(index=g.index)
    for k in (1, 2, 3):
        out[f"pm25_lag{k}"] = full["pm25"].shift(k).reindex(g.index)
    past = full["pm25"].shift(1)  # excludes today's value
    out["pm25_roll3_mean"] = past.rolling(3, min_periods=3).mean().reindex(g.index)
    out["pm25_roll7_mean"] = past.rolling(7, min_periods=7).mean().reindex(g.index)
    out["pm25_roll7_std"] = past.rolling(7, min_periods=7).std().reindex(g.index)
    out["pm25_roll7_max"] = past.rolling(7, min_periods=7).max().reindex(g.index)
    for k in (1, 2):
        out[f"AOD_lag{k}"] = full["AOD"].shift(k).reindex(g.index)
    out["AOD_diff1"] = (full["AOD"] - full["AOD"].shift(1)).reindex(g.index)
    out["temp_lag1"] = full["temperature_celsius"].shift(1).reindex(g.index)
    return out.reset_index(drop=True)

feats = []
for sid, g in df.groupby("id", sort=False):
    f = add_lags(g)
    f.index = g.index
    feats.append(f)
df = df.join(pd.concat(feats))

# Missingness indicators for Sentinel-5P columns (no forward-fill)
for c in ["NO2", "CO", "O3"]:
    df[f"{c}_missing"] = df[c].isna().astype(int)

df.to_csv("data/features.csv", index=False)

lag_cols = [c for c in df.columns if "lag" in c or "roll" in c or c == "AOD_diff1"]
print("Rows:", len(df), "| Columns:", df.shape[1])
print("\nShare of rows with a valid value (%):")
print((100 * df[lag_cols].notna().mean()).round(1).to_string())
print("\nRows with valid pm25_lag1:", int(df["pm25_lag1"].notna().sum()))
print("Rows with valid pm25_roll7_mean:", int(df["pm25_roll7_mean"].notna().sum()))