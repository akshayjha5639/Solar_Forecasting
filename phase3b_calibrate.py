"""
Phase 3b: Conformalized Quantile Regression (CQR) calibration.

Fixes the under-covering P10-P90 bands (71-74% observed vs 80% target)
by widening them with a correction learned on the validation set.

Method (Romano et al., 2019):
  1. Predict P10/P90 on the calibration set (2022-07 .. 2022-12).
  2. Nonconformity score per point: s = max(P10 - y, y - P90)
     (how far outside the band the truth fell; negative if inside).
  3. q_hat = the 80th percentile of scores (finite-sample corrected).
  4. Adjusted band on test: [P10 - q_hat, P90 + q_hat].

Run after phase3_xgboost.py:
    python phase3b_calibrate.py

Overwrites output/xgb_test_predictions_{horizon}.parquet with calibrated
bands and writes output/xgb_calibration_report.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

TABLE = "output/training_table.parquet"
TEST_START, VAL_START = "2023-01-01", "2022-07-01"
DAY_ELEV, TARGET_COVER = 5.0, 0.80

df = pd.read_parquet(TABLE).sort_index()
cap = float(df["capacity_kw"].median())

BASE_FEATS = ["fc_ghi", "fc_dni", "fc_dhi", "fc_temp", "fc_wind", "fc_cloud",
              "fc_rh", "fc_pressure", "fc_rain", "cs_ghi", "csi",
              "sun_elev", "sun_az", "hour_sin", "hour_cos",
              "doy_sin", "doy_cos"]
BASE_FEATS = [f for f in BASE_FEATS if f in df.columns]

HORIZONS = {
    "15min": ["gen_lag_1", "gen_lag_2", "gen_lag_4", "gen_lag_96",
              "gen_roll_mean_4"],
    "1h":    ["gen_lag_4", "gen_lag_96"],
    "24h":   ["gen_lag_96", "gen_lag_192"],
}

def load_model(name, col):
    p = f"output/xgb_models/xgb_{name}_{col}.json"
    if not os.path.exists(p):
        sys.exit(f"{p} missing -- run phase3_xgboost.py first")
    m = xgb.XGBRegressor()
    m.load_model(p)
    return m

report = []
for name, lag_cols in HORIZONS.items():
    feats = BASE_FEATS + [c for c in lag_cols if c in df.columns]
    data = df.dropna(subset=feats + ["gen_kw"])

    cal = data[(data.index >= VAL_START) & (data.index < TEST_START)]
    te = data[data.index >= TEST_START].copy()

    models = {c: load_model(name, c) for c in ("p10", "p50", "p90")}

    # --- nonconformity scores on calibration set (daytime only) ---
    cal_day = cal[cal["sun_elev"] > DAY_ELEV]
    p10c = models["p10"].predict(cal_day[feats])
    p90c = models["p90"].predict(cal_day[feats])
    y = cal_day["gen_kw"].values
    scores = np.maximum(p10c - y, y - p90c)
    n = len(scores)
    q_hat = np.quantile(scores, min(1.0, TARGET_COVER * (n + 1) / n))

    # --- apply to test set ---
    for c in ("p10", "p50", "p90"):
        te[c] = models[c].predict(te[feats])
    te["p10"] -= q_hat
    te["p90"] += q_hat
    for c in ("p10", "p50", "p90"):
        te[c] = te[c].clip(0, cap * 1.05)
        te.loc[te["sun_elev"] <= 0, c] = 0.0
    srt = np.sort(te[["p10", "p50", "p90"]].values, axis=1)
    te["p10"], te["p50"], te["p90"] = srt[:, 0], srt[:, 1], srt[:, 2]

    day = te[te["sun_elev"] > DAY_ELEV]
    cover_after = 100 * ((day["gen_kw"] >= day["p10"]) &
                         (day["gen_kw"] <= day["p90"])).mean()
    width = 100 * (day["p90"] - day["p10"]).mean() / cap

    report.append({"horizon": name,
                   "q_hat_kW": round(float(q_hat), 3),
                   "coverage_after_%": round(cover_after, 2),
                   "mean_band_width_%cap": round(width, 1)})

    te[["gen_kw", "sun_elev", "csi", "p10", "p50", "p90"]].to_parquet(
        f"output/xgb_test_predictions_{name}.parquet")
    print(f"{name}: q_hat={q_hat:.3f} kW -> coverage {cover_after:.1f}% "
          f"(band width {width:.1f}% of capacity)")

rep = pd.DataFrame(report).set_index("horizon")
rep.to_csv("output/xgb_calibration_report.csv")
print("\nCalibrated predictions saved -- refresh the dashboard to see "
      "the widened bands.")
print("Coverage should now sit near 80%. Slightly above is fine "
      "(conformal guarantees are >= target).")
