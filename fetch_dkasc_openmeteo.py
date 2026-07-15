"""
Build a training-ready solar forecasting dataset (v2):
  DKASC Alice Springs generation data  +  Open-Meteo historical FORECAST weather

STEP 1 -- MANUAL DOWNLOAD (once, ~2 minutes)
--------------------------------------------
DKASC's download requires agreeing to their terms in a web form, so it can't
be scripted cleanly. Do this:

  1. Open  https://dkasolarcentre.com.au/download?location=alice-springs
  2. Scroll to "Before downloading this file":
       - tick "I agree with the terms and conditions"
       - tick the notes-on-the-data box
       - select your role (e.g. Developer / Student)
  3. Under "Single Technologies (full data set with weather data)" click
     CSV Download next to a FIXED-tilt system. Good first choices:
       - 13  Trina, 5.3 kW, mono-Si, Fixed, 2009      (long history)
       - 38  Q CELLS, 5.9 kW, mono-Si, Fixed, 2017
       - 24  Q CELLS, 6.1 kW, poly-Si, Fixed, 2016
     (Avoid 1A/1B/2/4/6/22 -- those are trackers; fine later, harder first.)
  4. In the popup, set Data Resolution = 5 Minutes and a date range
     (2020-01-01 to 2023-12-31 is plenty; each year ~7 MB, be patient).
  5. Save the file into the folder  ./dkasc_raw/  next to this script.

STEP 2 -- RUN
-------------
    pip install pandas numpy pvlib requests pyarrow
    python fetch_dkasc_openmeteo_v2.py

Output: ./output/training_table.parquet (+ a CSV sample)
"""

import os
import sys
import glob
import requests
import numpy as np
import pandas as pd

try:
    import pvlib
except ImportError:
    sys.exit("pip install pvlib  (required)")

# ----------------------------------------------------------------------------
LAT, LON = -23.7624, 133.8754           # DKASC, Alice Springs
TZ = "Australia/Darwin"                  # ACST, no DST
START, END = "2020-01-01", "2023-12-31"  # used for the weather pull
RAW_DIR, OUT_DIR = "dkasc_raw", "output"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. GENERATION DATA (manually downloaded DKASC CSVs in ./dkasc_raw/)
# ----------------------------------------------------------------------------
def _is_html(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(200).lstrip().lower()
    return head.startswith(b"<") or b"<!doctype" in head or b"<html" in head

def load_dkasc() -> pd.DataFrame:
    files = [f for f in glob.glob(os.path.join(RAW_DIR, "*.csv"))]
    good = []
    for f in files:
        if _is_html(f):
            print(f"  SKIPPING {f} -- this is a saved webpage, not data. "
                  "Delete it and follow the manual-download steps at the top "
                  "of this script.")
        else:
            good.append(f)
    if not good:
        sys.exit("\nNo valid DKASC CSVs found in ./dkasc_raw/ -- see STEP 1 "
                 "instructions at the top of this script.")

    frames = []
    for f in good:
        df = pd.read_csv(f, low_memory=False)
        cols = {c.lower().strip(): c for c in df.columns}

        # timestamp column: named like 'timestamp' / 'time', else first col
        ts_key = next((cols[k] for k in cols if "time" in k or "date" in k),
                      df.columns[0])

        # power column: DKASC exports name it 'Active_Power',
        # 'Active Power', or '<Array name> - Active Power (kW)'
        pw_candidates = [c for c in df.columns
                         if "active" in c.lower() and "power" in c.lower()]
        if not pw_candidates:
            pw_candidates = [c for c in df.columns
                             if "power" in c.lower() and "reactive"
                             not in c.lower()]
        if not pw_candidates:
            print(f"  SKIPPING {f} -- no power column found. "
                  f"Columns: {list(df.columns)[:8]}...")
            continue
        pw_key = pw_candidates[0]
        print(f"  {os.path.basename(f)}: using '{ts_key}' + '{pw_key}'")

        sub = df[[ts_key, pw_key]].rename(
            columns={ts_key: "ts", pw_key: "gen_kw"})
        sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce",
                                   format="mixed", dayfirst=False)
        sub["gen_kw"] = pd.to_numeric(sub["gen_kw"], errors="coerce")
        frames.append(sub.dropna(subset=["ts"]))

    if not frames:
        sys.exit("No parseable DKASC files.")

    g = (pd.concat(frames).sort_values("ts")
         .drop_duplicates("ts").set_index("ts"))
    g["gen_kw"] = g["gen_kw"].clip(lower=0)
    if g.index.tz is None:                       # DKASC exports local time
        g.index = g.index.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
        g = g[g.index.notna()]

    g = g.resample("15min").mean()               # 5-min -> 15-min mean power
    print(f"DKASC: {len(g):,} 15-min rows, {g.index.min()} -> "
          f"{g.index.max()}")
    return g

# ----------------------------------------------------------------------------
# 2. WEATHER: Open-Meteo *historical forecast* archive
# ----------------------------------------------------------------------------
def load_weather(start: str, end: str) -> pd.DataFrame:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": ",".join([
            "shortwave_radiation", "direct_normal_irradiance",
            "diffuse_radiation", "temperature_2m", "wind_speed_10m",
            "cloud_cover", "relative_humidity_2m", "surface_pressure",
            "precipitation"]),
        "timezone": TZ,
    }
    print("Downloading Open-Meteo historical forecasts...")
    r = requests.get(url, params=params, timeout=600)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame(h).rename(columns={
        "time": "ts", "shortwave_radiation": "fc_ghi",
        "direct_normal_irradiance": "fc_dni", "diffuse_radiation": "fc_dhi",
        "temperature_2m": "fc_temp", "wind_speed_10m": "fc_wind",
        "cloud_cover": "fc_cloud", "relative_humidity_2m": "fc_rh",
        "surface_pressure": "fc_pressure", "precipitation": "fc_rain"})
    w["ts"] = pd.to_datetime(w["ts"]).dt.tz_localize(
        TZ, ambiguous="NaT", nonexistent="NaT")
    w = w.dropna(subset=["ts"]).set_index("ts")
    w = w.resample("15min").interpolate("linear")
    print(f"Weather: {len(w):,} rows")
    return w

# ----------------------------------------------------------------------------
# 3. SOLAR GEOMETRY + CLEAR-SKY (pvlib)
# ----------------------------------------------------------------------------
def add_solar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    loc = pvlib.location.Location(LAT, LON, tz=TZ)
    sp = loc.get_solarposition(idx)
    cs = loc.get_clearsky(idx)
    out = pd.DataFrame(index=idx)
    out["sun_elev"] = sp["apparent_elevation"].values
    out["sun_az"] = sp["azimuth"].values
    out["cs_ghi"] = cs["ghi"].values
    return out

# ----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def build_features(g: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    df = g.join(w, how="inner")
    if df.empty:
        sys.exit("Generation and weather date ranges don't overlap. "
                 "Adjust START/END to match your downloaded DKASC range.")
    df = df.join(add_solar_features(df.index))

    df.loc[df["sun_elev"] <= 0, "gen_kw"] = 0.0
    df["csi"] = (df["fc_ghi"] / df["cs_ghi"].clip(lower=1)).clip(0, 1.5)

    h = df.index.hour + df.index.minute / 60
    doy = df.index.dayofyear
    df["hour_sin"], df["hour_cos"] = (np.sin(2*np.pi*h/24),
                                      np.cos(2*np.pi*h/24))
    df["doy_sin"], df["doy_cos"] = (np.sin(2*np.pi*doy/365),
                                    np.cos(2*np.pi*doy/365))

    for lag in (1, 2, 4, 96, 192):
        df[f"gen_lag_{lag}"] = df["gen_kw"].shift(lag)
    df["gen_roll_mean_4"] = df["gen_kw"].shift(1).rolling(4).mean()
    df["gen_roll_max_96"] = df["gen_kw"].shift(1).rolling(96).max()

    df["capacity_kw"] = (df["gen_kw"].rolling(96*30, min_periods=96*7)
                         .quantile(0.99).bfill())

    day = df[df["sun_elev"] > 0]
    bad = day["gen_kw"].isna().groupby(day.index.date).mean() > 0.10
    bad_days = set(bad[bad].index)
    df = df[~pd.Series(df.index.date, index=df.index).isin(bad_days)]

    df = df.dropna(subset=["gen_kw"])
    print(f"Training table: {len(df):,} rows x {df.shape[1]} cols "
          f"(dropped {len(bad_days)} bad days)")
    return df

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    gen = load_dkasc()
    # match weather pull to the generation data actually present
    start = max(pd.Timestamp(START, tz=TZ), gen.index.min()).strftime("%Y-%m-%d")
    end = min(pd.Timestamp(END, tz=TZ), gen.index.max()).strftime("%Y-%m-%d")
    wx = load_weather(start, end)
    table = build_features(gen, wx)
    table.to_parquet(os.path.join(OUT_DIR, "training_table.parquet"))
    table.head(500).to_csv(os.path.join(OUT_DIR,
                                        "training_table_sample.csv"))
    print("\nSaved -> output/training_table.parquet")
    print(table.filter(["gen_kw", "fc_ghi", "cs_ghi", "csi", "sun_elev",
                        "gen_lag_96"]).describe().round(2))
