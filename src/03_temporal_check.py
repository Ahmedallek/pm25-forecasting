from pathlib import Path
import pandas as pd

df = pd.read_csv("data/clean.csv", parse_dates=["date"])
df = df.sort_values(["id", "date"])

rows = []
for station_id, g in df.groupby("id"):
    dates = g["date"].drop_duplicates()
    gaps = dates.diff().dt.days.dropna()
    full_range_days = (dates.max() - dates.min()).days + 1
    rows.append({
        "id": station_id,
        "sensor_name": g["sensor_name"].iloc[0],
        "n_records": len(g),
        "date_min": dates.min().date(),
        "date_max": dates.max().date(),
        "span_days": full_range_days,
        "coverage_pct": round(100 * len(dates) / full_range_days, 1),
        "median_gap_days": gaps.median() if len(gaps) else None,
        "max_gap_days": gaps.max() if len(gaps) else None,
    })

summary = pd.DataFrame(rows).sort_values("n_records", ascending=False)
summary.to_csv("results/station_summary.csv", index=False)

print("Stations:", len(summary))
print("Stations with >= 100 records:", int((summary["n_records"] >= 100).sum()))
print("Stations with >= 300 records:", int((summary["n_records"] >= 300).sum()))
print("Stations with < 30 records:", int((summary["n_records"] < 30).sum()))
print("\nMedian gap between records (all stations):")
print(summary["median_gap_days"].describe())
print("\nTop 10 stations by records:")
print(summary.head(10).to_string(index=False))
print("\nBottom 10 stations by records:")
print(summary.tail(10).to_string(index=False))
print("\nRecords per year:")
print(df["date"].dt.year.value_counts().sort_index())