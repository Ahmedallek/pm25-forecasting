from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

raw_dir = Path("data/raw")
csv_files = sorted(raw_dir.glob("*.csv"))
print("CSV files found:")
for f in csv_files:
    print("  -", f.name)

if not csv_files:
    raise SystemExit("No CSV file in data/raw/.")

for f in csv_files:
    print("\n" + "=" * 70)
    print("FILE:", f.name)
    print("=" * 70)

    df = pd.read_csv(f)

    print("\n--- Shape (rows, columns) ---")
    print(df.shape)

    print("\n--- Columns and dtypes ---")
    print(df.dtypes)

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Missing values per column ---")
    miss = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
    })
    print(miss)

    print("\n--- Fully duplicated rows ---")
    print(df.duplicated().sum())

    print("\n--- Quick statistics ---")
    print(df.describe().T)