"""Trim a full DKASC download to 2020-2023 and save as CSV into dkasc_raw/.
Usage: set INFILE below, then:  python trim_dkasc.py
"""
import pandas as pd

INFILE = r"dkasc_raw/full_download.csv"   # <-- change to your file (.csv or .xlsx)

if INFILE.lower().endswith(".xlsx"):
    df = pd.read_excel(INFILE)
else:
    df = pd.read_csv(INFILE, low_memory=False)

ts_col = next(c for c in df.columns if "time" in c.lower() or "date" in c.lower())
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", format="mixed")

mask = (df[ts_col] >= "2020-01-01") & (df[ts_col] < "2024-01-01")
trimmed = df[mask]
print(f"Original: {len(df):,} rows | Trimmed: {len(trimmed):,} rows")
print(f"Range: {trimmed[ts_col].min()} -> {trimmed[ts_col].max()}")
trimmed.to_csv("dkasc_raw/dkasc_2020_2023.csv", index=False)
print("Saved -> dkasc_raw/dkasc_2020_2023.csv")
