import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
SEED = 42

df = pd.read_csv("data/clean_kept.csv", parse_dates=["date"]).sort_values(["id", "date"])
eda = pd.read_csv("results/Exploratory_Data.csv")
cols = list(eda["variable"])
static_cols = ["latitude", "longitude"]  # constant per station: no temporal stats


def daily_series(g, c):
    s = g.drop_duplicates("date").set_index("date")[c]
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))


def longest_segment(s):
    s = s.dropna()
    if s.empty:
        return s
    run = (s.index.to_series().diff().dt.days.fillna(1) != 1).cumsum()
    return s[run == run.value_counts().idxmax()]


# 1) Isolation Forest and LOF (univariate, descriptive only, not used to remove rows)
for c in cols:
    x = df[c].dropna().to_numpy().reshape(-1, 1)
    n_if = int((IsolationForest(random_state=SEED).fit_predict(x) == -1).sum())
    n_lof = int((LocalOutlierFactor(n_neighbors=20).fit_predict(x) == -1).sum())
    i = eda["variable"] == c
    eda.loc[i, "n_out_if"] = n_if
    eda.loc[i, "pct_out_if"] = round(100 * n_if / len(x), 2)
    eda.loc[i, "n_out_lof"] = n_lof
    eda.loc[i, "pct_out_lof"] = round(100 * n_lof / len(x), 2)
    print("outliers done:", c)

# 2) VIF on predictors (median-filled and standardized, descriptive only)
preds = [c for c in cols if c != "pm25"]
X = df[preds]
X = (X.fillna(X.median()) - X.mean()) / X.std()
Xc = sm.add_constant(X)
for j, c in enumerate(preds, start=1):
    eda.loc[eda["variable"] == c, "vif"] = variance_inflation_factor(Xc.values, j)

# 3) ACF lag-1 and ADF p-value: median over stations, never bridging gaps
for c in cols:
    if c in static_cols:
        continue
    acfs, adfs = [], []
    for sid, g in df.groupby("id"):
        s = daily_series(g, c)
        a = s.autocorr(1)
        if pd.notna(a):
            acfs.append(a)
        seg = longest_segment(s)
        if len(seg) >= 30 and seg.nunique() > 1:
            try:
                adfs.append(adfuller(seg, autolag="AIC")[1])
            except Exception:
                pass
    i = eda["variable"] == c
    eda.loc[i, "acf_lag1"] = np.median(acfs) if acfs else np.nan
    eda.loc[i, "pacf_lag1"] = np.median(acfs) if acfs else np.nan  # PACF(1) = ACF(1)
    eda.loc[i, "adf_p"] = np.median(adfs) if adfs else np.nan
    print("temporal stats done:", c)

eda["notes"] = ("IF/LOF univariate (descriptive); ACF and ADF = median over stations "
                "on non-bridged daily series; ADF on stations with a >=30-day segment")
eda.to_csv("results/Exploratory_Data.csv", index=False)
print("\nUpdated results/Exploratory_Data.csv")
print(eda[["variable", "n_out_if", "n_out_lof", "vif", "acf_lag1", "adf_p"]].round(3).to_string(index=False))

# 4) Lead-lag: pooled Spearman between PM2.5(t) and X(t-k), per station, no gap bridging
lead_vars = ["AOD", "NO2", "CO", "O3", "temperature_celsius", "wind_u", "wind_v", "pressure_mb"]
lags = range(0, 8)
xs = {(v, k): [] for v in lead_vars for k in lags}
ys = {(v, k): [] for v in lead_vars for k in lags}
for sid, g in df.groupby("id"):
    full = g.drop_duplicates("date").set_index("date")
    full = full.reindex(pd.date_range(full.index.min(), full.index.max(), freq="D"))
    y = full["pm25"]
    for v in lead_vars:
        for k in lags:
            x = full[v].shift(k)
            m = x.notna() & y.notna()
            xs[(v, k)].append(x[m].to_numpy())
            ys[(v, k)].append(y[m].to_numpy())

rows = []
for (v, k) in xs:
    x = np.concatenate(xs[(v, k)])
    y = np.concatenate(ys[(v, k)])
    rho = stats.spearmanr(x, y)[0] if len(x) > 30 else np.nan
    rows.append({"variable": v, "lag_days": k, "spearman": rho, "n_pairs": len(x)})
ll = pd.DataFrame(rows)
ll.to_csv("results/lead_lag.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
for v in lead_vars:
    d = ll[ll["variable"] == v]
    ax.plot(d["lag_days"], d["spearman"], marker="o", label=v)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("lag k (days): predictor at t-k vs PM2.5 at t")
ax.set_ylabel("Spearman correlation")
ax.set_title("Lead-lag relationships with PM2.5")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig("figures/09_lead_lag.png", dpi=150)
print("\nSaved results/lead_lag.csv and figures/09_lead_lag.png")
print(ll.pivot(index="lag_days", columns="variable", values="spearman").round(3))