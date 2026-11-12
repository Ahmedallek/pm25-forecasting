from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

MIN_RECORDS = 100  # stations with fewer records are dropped (decision made before any split)

df = pd.read_csv("data/clean.csv", parse_dates=["date"])
rep = pd.read_csv("results/cleaning_report.csv").set_index("item")["value"]
n_low = int(rep["pm25_sentinel_low (<= 0)"])
n_high = int(rep["pm25_sentinel_high (>= 500)"])
n_removed = int(rep["rows_removed_invalid_pm25"])

# 1) Station selection
counts = df.groupby("id").size()
keep_ids = counts[counts >= MIN_RECORDS].index
kept = df[df["id"].isin(keep_ids)].copy()
print("Stations kept:", len(keep_ids), "of", len(counts))
print("Rows kept:", len(kept), "of", len(df))
kept.to_csv("data/clean_kept.csv", index=False)

# 2) Variable metadata: family, unit
meta = {
    "pm25": ("Target", "ug/m3"),
    "AOD": ("AOD", "dimensionless"),
    "NO2": ("NO2", "mol/m2 (to verify)"),
    "CO": ("CO", "mol/m2 (to verify)"),
    "O3": ("O3", "mol/m2 (to verify)"),
    "temperature_celsius": ("Meteorological", "degC"),
    "pressure_mb": ("Meteorological", "mb"),
    "wind_u": ("Meteorological", "m/s (to verify)"),
    "wind_v": ("Meteorological", "m/s (to verify)"),
    "latitude": ("Spatial", "degrees"),
    "longitude": ("Spatial", "degrees"),
}
num_cols = list(meta.keys())
corr = kept[num_cols].corr()  # Pearson

key = ["date", "latitude", "longitude", "id"]
n_dup_key = int(kept.duplicated(subset=key).sum())
target = kept["pm25"]

rows = []
for col in num_cols:
    x = kept[col]
    xv = x.dropna()
    q1, med, q3 = xv.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    n_iqr = int(((xv < q1 - 1.5 * iqr) | (xv > q3 + 1.5 * iqr)).sum())
    z = (xv - xv.mean()) / xv.std()
    n_z = int((z.abs() > 3).sum())

    others = corr[col].drop(col).abs()
    partner = others.idxmax()

    if col == "pm25":
        assoc, pval = "", np.nan
    else:
        mask = x.notna() & target.notna()
        rho, pval = stats.spearmanr(x[mask], target[mask])
        assoc = f"spearman={rho:.3f}"

    rows.append({
        "model_id": "eda",
        "model_group": "all",
        "variable": col,
        "family": meta[col][0],
        "type": "numeric",
        "unit": meta[col][1],
        "n_total": len(x),
        "n_missing": int(x.isna().sum()),
        "pct_missing": round(100 * x.isna().mean(), 2),
        "n_unique": int(x.nunique()),
        "n_duplicates_key": n_dup_key,
        "min": xv.min(), "q1": q1, "median": med, "q3": q3, "max": xv.max(),
        "mean": xv.mean(), "std": xv.std(),
        "skewness": stats.skew(xv), "kurtosis": stats.kurtosis(xv),
        "iqr": iqr,
        "n_out_iqr": n_iqr, "pct_out_iqr": round(100 * n_iqr / len(xv), 2),
        "n_out_z": n_z, "pct_out_z": round(100 * n_z / len(xv), 2),
        "n_sentinel_low": n_low if col == "pm25" else "",
        "n_sentinel_high": n_high if col == "pm25" else "",
        "n_removed_total": n_removed if col == "pm25" else "",
        "max_abs_corr": round(others.max(), 3),
        "corr_partner": partner,
        "target_assoc": assoc,
        "target_pval": pval,
        "is_leakage_suspect": False,
        "notes": "",
    })

out = pd.DataFrame(rows)
out.to_csv("results/Exploratory_Data.csv", index=False)
print("\nSaved results/Exploratory_Data.csv")
cols = ["variable", "pct_missing", "min", "median", "max", "skewness",
        "pct_out_iqr", "max_abs_corr", "corr_partner", "target_assoc"]
print(out[cols].to_string(index=False))
print("\nPM2.5 share > 25:", round(100 * (target > 25).mean(), 1), "%")
print("PM2.5 share > 50:", round(100 * (target > 50).mean(), 1), "%")