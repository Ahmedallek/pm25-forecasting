import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from pipeline_utils import (ALL_FEATURES, TARGET, build_pipeline,
                            load_data, load_folds, load_test_ids)

data = load_data()
test_ids = set(load_test_ids())


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


for scheme in ["temporal", "spatial", "standard"]:
    print(f"\n=== {scheme} CV ===")
    for fold in load_folds(scheme):
        tr, va = fold["train"], fold["val"]
        # Leakage checks
        assert not set(tr) & set(va), "train/validation overlap"
        assert not (set(tr) | set(va)) & test_ids, "future test rows in a fold"
        if scheme == "temporal":
            gap = (data.loc[va, "date"].min() - data.loc[tr, "date"].max()).days
            assert gap >= 1, "validation not after training"
        if scheme == "spatial":
            assert not set(data.loc[tr, "id"]) & set(data.loc[va, "id"]), "station in both sets"

        Xtr, ytr = data.loc[tr, ALL_FEATURES], data.loc[tr, TARGET]
        Xva, yva = data.loc[va, ALL_FEATURES], data.loc[va, TARGET]

        base = build_pipeline(DummyRegressor(strategy="mean")).fit(Xtr, ytr)
        ridge = build_pipeline(Ridge(alpha=1.0)).fit(Xtr, ytr)
        print(f"fold {fold['fold']}: train={len(tr)} val={len(va)} | "
              f"RMSE mean-baseline={rmse(yva, base.predict(Xva)):.2f} | "
              f"RMSE ridge={rmse(yva, ridge.predict(Xva)):.2f}")

print("\nAll leakage checks passed.")