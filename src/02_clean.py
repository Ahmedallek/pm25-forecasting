from pathlib import Path
import pandas as pd

RAW_FILE = Path("data/raw/combined_gee_pm25.csv")
CLEAN_FILE = Path("data/clean.csv")
REPORT_FILE = Path("results/cleaning_report.csv")

report = []


def log(item, value):
    """Print a cleaning statistic and store it for the report."""
    print(f"{item}: {value}")
    report.append({"item": item, "value": value})


df = pd.read_csv(RAW_FILE)
log("rows_initial", len(df))

# 1) Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
log("invalid_dates", int(df["date"].isna().sum()))
log("date_min", df["date"].min().date())
log("date_max", df["date"].max().date())

# 2) Invalid PM2.5 values (protocol section 5.1)
mask_missing = df["pm25"].isna()
mask_low = df["pm25"] <= 0
mask_high = df["pm25"] >= 500
log("pm25_missing", int(mask_missing.sum()))
log("pm25_sentinel_low (<= 0)", int(mask_low.sum()))
log("pm25_sentinel_high (>= 500)", int(mask_high.sum()))

invalid = mask_missing | mask_low | mask_high
log("rows_removed_invalid_pm25", int(invalid.sum()))
df = df.loc[~invalid].copy()

# 3) Duplicates: all columns, then key fields
n_full = int(df.duplicated().sum())
log("duplicates_all_columns", n_full)
df = df.drop_duplicates()

key = ["date", "latitude", "longitude", "id"]
n_key = int(df.duplicated(subset=key).sum())
log("duplicates_key (date, lat, lon, id)", n_key)
df = df.drop_duplicates(subset=key, keep="first")

# 4) Station consistency checks
log("n_station_ids", df["id"].nunique())
log("n_sensor_names", df["sensor_name"].nunique())
log("ids_with_several_names", int((df.groupby("id")["sensor_name"].nunique() > 1).sum()))
log("ids_with_several_coords",
    int((df.groupby("id")[["latitude", "longitude"]].nunique().max(axis=1) > 1).sum()))

per_station = df.groupby("id").size()
log("records_per_station_min", int(per_station.min()))
log("records_per_station_median", float(per_station.median()))
log("records_per_station_max", int(per_station.max()))

# 5) Save
log("rows_final", len(df))
df = df.sort_values(["id", "date"]).reset_index(drop=True)
df.to_csv(CLEAN_FILE, index=False)
pd.DataFrame(report).to_csv(REPORT_FILE, index=False)
print("\nSaved:", CLEAN_FILE, "and", REPORT_FILE)