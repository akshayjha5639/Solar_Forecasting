"""
Phase 2: Baselines + frozen evaluation harness.

Run next to the Phase 1 output:
    pip install pandas numpy pvlib pyarrow matplotlib
    python phase2_baselines.py

Baselines built:
  1. persistence        : P(t) = P(t - 24h)                       [24h horizon]
  2. smart_persistence  : persistence scaled by clear-sky ratio   [24h horizon]
  3. physics            : PVLib chain, forecast irradiance -> POA -> PVWatts
  4. persistence_15min  : P(t) = P(t - 15min)                     [15m horizon]

Evaluation rules (FROZEN -- every future model uses exactly these):
  - test period : calendar year 2023 (train/val = 2020-2022)
  - daytime only: sun_elev > 5 degrees
  - metrics     : nMAE %, nRMSE % (normalised by capacity), bias %,
                  skill % vs smart persistence
Outputs: output/baseline_results.csv, output/baseline_sample_week.png
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import pvlib
except ImportError:
    sys.exit("pip install pvlib")

# ----------------------------------------------------------------------------
LAT, LON, TZ = -23.7624, 133.8754, "Australia/Darwin"
TILT, AZIMUTH = 23.8, 0.0        # fixed array: tilt ~ latitude, facing NORTH
                                 # (southern hemisphere -> azimuth 0)
GAMMA_PDC = -0.004               # power temp coefficient (1/degC), mono-Si
TEST_START = "2023-01-01"
DAY_ELEV = 5.0                   # daytime threshold (degrees)

TABLE = "output/training_table.parquet"
if not os.path.exists(TABLE):
    sys.exit(f"{TABLE} not found -- run fetch_dkasc_openmeteo_v2.py first")

df = pd.read_parquet(TABLE).sort_index()
cap = float(df["capacity_kw"].median())
print(f"Loaded {len(df):,} rows | effective capacity ~{cap:.2f} kW")

# ----------------------------------------------------------------------------
# BASELINE 1 & 2: persistence and smart persistence (24h horizon)
# ----------------------------------------------------------------------------
df["pred_persistence"] = df["gen_lag_96"]

cs_ratio = (df["cs_ghi"] / df["cs_ghi"].shift(96).clip(lower=1)).clip(0, 2)
df["pred_smart_pers"] = (df["gen_lag_96"] * cs_ratio).clip(0, cap * 1.1)

# short-horizon persistence (15 min)
df["pred_pers_15m"] = df["gen_lag_1"]

# ----------------------------------------------------------------------------
# BASELINE 3: physics chain (works day one for any new site -- no history)
#   forecast GHI/DNI/DHI -> plane-of-array -> cell temp -> PVWatts DC -> AC
# ----------------------------------------------------------------------------
loc = pvlib.location.Location(LAT, LON, tz=TZ)
sp = loc.get_solarposition(df.index)

poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=TILT, surface_azimuth=AZIMUTH,
    solar_zenith=sp["apparent_zenith"], solar_azimuth=sp["azimuth"],
    dni=df["fc_dni"].fillna(0), ghi=df["fc_ghi"].fillna(0),
    dhi=df["fc_dhi"].fillna(0))["poa_global"].clip(lower=0)

t_cell = pvlib.temperature.faiman(poa, df["fc_temp"].fillna(25),
                                  df["fc_wind"].fillna(1))

pdc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=cap * 1.1,
                                gamma_pdc=GAMMA_PDC)
df["pred_physics"] = (pdc * 0.96).clip(0, cap * 1.05)   # inverter eff ~96%
df.loc[df["sun_elev"] <= 0, "pred_physics"] = 0.0

# ----------------------------------------------------------------------------
# FROZEN EVALUATION HARNESS
# ----------------------------------------------------------------------------
def evaluate(frame: pd.DataFrame, pred_col: str, cap_kw: float) -> dict:
    """Daytime-only metrics on the test year. FROZEN -- do not change."""
    t = frame[(frame.index >= TEST_START) & (frame["sun_elev"] > DAY_ELEV)]
    t = t.dropna(subset=[pred_col, "gen_kw"])
    err = t[pred_col] - t["gen_kw"]
    return {
        "model": pred_col.replace("pred_", ""),
        "n": len(t),
        "nMAE_%": 100 * err.abs().mean() / cap_kw,
        "nRMSE_%": 100 * np.sqrt((err**2).mean()) / cap_kw,
        "bias_%": 100 * err.mean() / cap_kw,
    }

rows = [evaluate(df, c, cap) for c in
        ["pred_persistence", "pred_smart_pers", "pred_physics",
         "pred_pers_15m"]]
res = pd.DataFrame(rows).set_index("model")

# skill vs smart persistence (the official benchmark)
ref = res.loc["smart_pers", "nMAE_%"]
res["skill_vs_smartpers_%"] = 100 * (1 - res["nMAE_%"] / ref)

res = res.round(2)
print("\n=== BASELINE RESULTS (test year 2023, daytime only) ===")
print(res.to_string())
res.to_csv("output/baseline_results.csv")

# condition breakdown: clear vs cloudy days (by daily mean clear-sky index)
test = df[(df.index >= TEST_START) & (df["sun_elev"] > DAY_ELEV)].copy()
daily_csi = test.groupby(test.index.date)["csi"].mean()
clear_days = set(daily_csi[daily_csi >= 0.8].index)
cloudy_days = set(daily_csi[daily_csi < 0.5].index)
for label, days in [("CLEAR", clear_days), ("CLOUDY", cloudy_days)]:
    sub = test[pd.Series(test.index.date, index=test.index).isin(days)]
    if len(sub):
        e = (sub["pred_smart_pers"] - sub["gen_kw"]).abs().mean() / cap * 100
        p = (sub["pred_physics"] - sub["gen_kw"]).abs().mean() / cap * 100
        print(f"{label:6s} days ({len(days):3d}): smart_pers nMAE "
              f"{e:5.2f}% | physics nMAE {p:5.2f}%")

# plot one test week
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wk = df.loc["2023-03-06":"2023-03-12"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(wk.index, wk["gen_kw"], "k-", lw=1.2, label="actual")
    ax.plot(wk.index, wk["pred_smart_pers"], "--", lw=1,
            label="smart persistence")
    ax.plot(wk.index, wk["pred_physics"], ":", lw=1.2, label="physics")
    ax.set_ylabel("kW"); ax.legend(); ax.set_title("Baselines, sample week")
    fig.tight_layout()
    fig.savefig("output/baseline_sample_week.png", dpi=120)
    print("\nPlot -> output/baseline_sample_week.png")
except ImportError:
    print("(matplotlib not installed -- skipped plot)")

print("\nPhase 2 done. These numbers are the bar every ML model must beat.")
