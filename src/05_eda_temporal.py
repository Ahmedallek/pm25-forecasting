import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

df = pd.read_csv("data/clean_kept.csv", parse_dates=["date"])
df = df.sort_values(["id", "date"])


def longest_segment(series):
    """Longest run of consecutive daily values without gaps."""
    s = series.dropna()
    if s.empty:
        return s
    new_run = s.index.to_series().diff().dt.days.fillna(1) != 1
    run_id = new_run.cumsum()
    return s[run_id == run_id.value_counts().idxmax()]


def monthly_missing(g, col):
    m = g.groupby(g["date"].dt.strftime("%Y-%m"))[col].apply(
        lambda x: round(100 * x.isna().mean(), 1))
    return json.dumps(m.to_dict())


rows = []
for sid, g in df.groupby("id"):
    g = g.drop_duplicates("date").set_index("date", drop=False)
    y = g["pm25"]
    daily = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq="D"))
    seg = longest_segment(daily)

    gaps = y.index.to_series().diff().dt.total_seconds().dropna()
    n_days_span = len(daily)

    # Autocorrelation (pairwise, so gaps are never bridged)
    acf1 = daily.autocorr(1)
    acf7 = daily.autocorr(7)

    # Tests on the longest continuous segment
    lb_p = adf_p = kpss_p = np.nan
    if len(seg) >= 30:
        lb_p = float(acorr_ljungbox(seg, lags=[7])["lb_pvalue"].iloc[0])
        adf_p = float(adfuller(seg, autolag="AIC")[1])
        try:
            kpss_p = float(kpss(seg, regression="c", nlags="auto")[1])
        except Exception:
            pass

    # Linear trend
    t_days = (y.index - y.index.min()).days.values
    lr = stats.linregress(t_days, y.values) if len(y) > 2 else None

    # Seasonal amplitude: max - min of monthly medians
    med_month = y.groupby(y.index.month).median()
    seasonal_amp = float(med_month.max() - med_month.min())

    # Drift: KS test, first half vs second half of the records
    half = len(y) // 2
    ks = stats.ks_2samp(y.values[:half], y.values[half:])

    n25 = int((y > 25).sum())
    n50 = int((y > 50).sum())
    strategy = "temporal" if (len(y) >= 100 and len(seg) >= 30) else "spatial"

    rows.append({
        "model_id": "eda", "model_group": "all", "variable": "pm25",
        "node_id": sid,
        "n_records": len(y),
        "t_min": y.index.min().date(), "t_max": y.index.max().date(),
        "sampling_interval_median_sec": gaps.median(),
        "sampling_interval_std_sec": gaps.std(),
        "pct_missing_intervals": round(100 * (1 - len(y) / n_days_span), 1),
        "acf_lag1": acf1, "acf_lag7": acf7, "acf_lag24": np.nan,
        "pacf_lag1": acf1,  # PACF at lag 1 equals ACF at lag 1
        "ljung_box_p": lb_p, "adf_p": adf_p, "kpss_p": kpss_p,
        "trend_slope_per_day": lr.slope if lr else np.nan,
        "trend_pval": lr.pvalue if lr else np.nan,
        "diurnal_amplitude": np.nan,  # daily data: not applicable
        "seasonal_amplitude": seasonal_amp,
        "target_drift_stat": ks.statistic, "target_drift_pval": ks.pvalue,
        "n_days_above_25": n25, "n_days_above_50": n50,
        "pct_days_above_25": round(100 * n25 / len(y), 2),
        "pct_days_above_50": round(100 * n50 / len(y), 2),
        "regime_breaks": np.nan,
        "cv_strategy_recommended": strategy,
        "gap_between_train_test_sec": 86400.0,
        "n_cycles_observed": round(n_days_span / 365.25, 1),
        "no2_pct_missing_by_month": monthly_missing(g, "NO2"),
        "co_pct_missing_by_month": monthly_missing(g, "CO"),
        "o3_pct_missing_by_month": monthly_missing(g, "O3"),
        "notes": f"longest_continuous_segment={len(seg)}d; max_gap_days={gaps.max() / 86400:.0f}",
    })

out = pd.DataFrame(rows)
out.to_csv("results/Exploratory_temporal.csv", index=False)
print("Saved results/Exploratory_temporal.csv with", len(out), "stations")
print("\nMedian ACF lag-1:", round(out["acf_lag1"].median(), 3))
print("Median ACF lag-7:", round(out["acf_lag7"].median(), 3))
print("Stations with ADF p < 0.05:", int((out["adf_p"] < 0.05).sum()), "of", int(out["adf_p"].notna().sum()))
print("Stations with Ljung-Box p < 0.05:", int((out["ljung_box_p"] < 0.05).sum()))
print("Stations with significant drift (p < 0.05):", int((out["target_drift_pval"] < 0.05).sum()))
print("CV strategy counts:")
print(out["cv_strategy_recommended"].value_counts())