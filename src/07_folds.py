import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

SEED = 42
TEST_YEAR = 2025  # future test period, never used in any development fold

df = pd.read_csv("data/clean_kept.csv", parse_dates=["date"])
df = df.sort_values(["date", "id"]).reset_index(drop=True)
df["row_id"] = np.arange(len(df))

# Future test set: all of 2025 and later. Development set: before 2025.
test = df[df["date"].dt.year >= TEST_YEAR]
dev = df[df["date"].dt.year < TEST_YEAR].reset_index(drop=True)
print("Development rows:", len(dev), "| Future test rows:", len(test))
print("Dev period:", dev["date"].min().date(), "to", dev["date"].max().date())
print("Test period:", test["date"].min().date(), "to", test["date"].max().date())

# Save the dataset with a stable row id so every model uses the same rows
df.to_csv("data/dataset_with_rowid.csv", index=False)
test["row_id"].to_json("data/test_row_ids.json", orient="values")


def to_list(a):
    return [int(x) for x in a]


# 1) Temporal CV: 5 folds, 1-day gap (by unique dates, so no leakage across stations)
unique_dates = np.sort(dev["date"].unique())
tscv = TimeSeriesSplit(n_splits=5, gap=1)
temporal = []
for k, (tr_d, va_d) in enumerate(tscv.split(unique_dates)):
    tr = dev[dev["date"].isin(unique_dates[tr_d])]["row_id"]
    va = dev[dev["date"].isin(unique_dates[va_d])]["row_id"]
    temporal.append({"fold": k, "train": to_list(tr), "val": to_list(va)})
    print(f"Temporal fold {k}: train={len(tr)} val={len(va)} "
          f"({pd.Timestamp(unique_dates[va_d][0]).date()} to {pd.Timestamp(unique_dates[va_d][-1]).date()})")

# 2) Spatial CV: GroupKFold by station id (station id is a group only, never a feature)
gkf = GroupKFold(n_splits=5)
spatial = []
for k, (tr, va) in enumerate(gkf.split(dev, groups=dev["id"])):
    spatial.append({"fold": k,
                    "train": to_list(dev.iloc[tr]["row_id"]),
                    "val": to_list(dev.iloc[va]["row_id"])})
    print(f"Spatial fold {k}: train={len(tr)} val={len(va)} "
          f"stations_val={dev.iloc[va]['id'].nunique()}")

# 3) Random CV (diagnostic only)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
standard = []
for k, (tr, va) in enumerate(kf.split(dev)):
    standard.append({"fold": k,
                     "train": to_list(dev.iloc[tr]["row_id"]),
                     "val": to_list(dev.iloc[va]["row_id"])})
    print(f"Random fold {k}: train={len(tr)} val={len(va)}")

for name, folds in [("folds_temporal", temporal), ("folds_spatial", spatial), ("folds_standard", standard)]:
    with open(f"data/{name}.json", "w") as f:
        json.dump({"seed": SEED, "test_year_from": TEST_YEAR, "folds": folds}, f)

print("\nSaved folds_temporal.json, folds_spatial.json, folds_standard.json in data/")