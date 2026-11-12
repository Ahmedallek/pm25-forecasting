import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("data/clean_kept.csv", parse_dates=["date"])
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
sns.set_theme(style="whitegrid")

# 1) Histograms of key variables
cols = ["pm25", "AOD", "NO2", "CO", "O3", "temperature_celsius", "pressure_mb", "wind_u", "wind_v"]
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, c in zip(axes.ravel(), cols):
    ax.hist(df[c].dropna(), bins=60, color="steelblue")
    ax.set_title(c)
fig.suptitle("Distributions of key variables")
fig.tight_layout()
fig.savefig("figures/01_histograms.png", dpi=150)
plt.close(fig)

# 2) PM2.5 distribution: raw and log scale, with thresholds
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["pm25"], bins=100, color="steelblue")
axes[0].axvline(25, color="orange", ls="--", label="25 ug/m3")
axes[0].axvline(50, color="red", ls="--", label="50 ug/m3")
axes[0].set_title("PM2.5 (raw)")
axes[0].legend()
axes[1].hist(np.log1p(df["pm25"]), bins=100, color="steelblue")
axes[1].set_title("PM2.5 (log1p)")
fig.tight_layout()
fig.savefig("figures/02_pm25_distribution.png", dpi=150)
plt.close(fig)

# 3) Boxplots
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, c in zip(axes.ravel(), cols):
    sns.boxplot(y=df[c], ax=ax, color="lightsteelblue")
    ax.set_title(c)
fig.suptitle("Boxplots of key variables")
fig.tight_layout()
fig.savefig("figures/03_boxplots.png", dpi=150)
plt.close(fig)

# 4) Correlation heatmaps (Pearson and Spearman)
num = ["pm25", "AOD", "NO2", "CO", "O3", "temperature_celsius", "pressure_mb", "wind_u", "wind_v"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, method in zip(axes, ["pearson", "spearman"]):
    sns.heatmap(df[num].corr(method=method), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(method.capitalize())
fig.tight_layout()
fig.savefig("figures/04_correlations.png", dpi=150)
plt.close(fig)

# 5) Seasonality: PM2.5 by month
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="month", y="pm25", ax=ax, showfliers=False, color="lightsteelblue")
ax.set_title("PM2.5 by month (outliers hidden)")
fig.tight_layout()
fig.savefig("figures/05_pm25_by_month.png", dpi=150)
plt.close(fig)

# 6) Drift: PM2.5 by year (median, IQR, share above 25)
g = df.groupby("year")["pm25"]
drift = pd.DataFrame({
    "median": g.median(),
    "q1": g.quantile(0.25),
    "q3": g.quantile(0.75),
    "pct_above_25": g.apply(lambda s: 100 * (s > 25).mean()),
    "n": g.size(),
})
drift.to_csv("results/pm25_drift_by_year.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(drift.index, drift["median"], marker="o")
axes[0].fill_between(drift.index, drift["q1"], drift["q3"], alpha=0.3)
axes[0].set_title("PM2.5 median and IQR by year")
axes[1].bar(drift.index, drift["pct_above_25"], color="orange")
axes[1].set_title("% of records above 25 ug/m3")
fig.tight_layout()
fig.savefig("figures/06_drift_by_year.png", dpi=150)
plt.close(fig)

# 7) Missingness of NO2/CO/O3 by month
miss = df.groupby("month")[["NO2", "CO", "O3"]].apply(lambda x: 100 * x.isna().mean())
fig, ax = plt.subplots(figsize=(9, 4))
miss.plot(marker="o", ax=ax)
ax.set_title("Missing values (%) by month")
ax.set_ylabel("%")
fig.tight_layout()
fig.savefig("figures/07_missing_by_month.png", dpi=150)
plt.close(fig)

# 8) Records per station (top 30)
cnt = df.groupby("id").size().sort_values(ascending=False).head(30)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(cnt.index.astype(str), cnt.values, color="steelblue")
ax.set_title("Records per station (top 30)")
ax.set_xlabel("station id")
fig.tight_layout()
fig.savefig("figures/08_records_per_station.png", dpi=150)
plt.close(fig)

print("Saved 8 figures in figures/ and results/pm25_drift_by_year.csv")
print(drift.round(2))