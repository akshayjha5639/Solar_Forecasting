"""
Phase 3c: Retrain with a CLEAN split + Mondrian conformal calibration.

Why: in Phase 3 the same 2022-H2 slice was used both for early stopping and
for conformal calibration, making the correction optimistically small
(coverage stalled at ~77% instead of 80%).

New split:
  train        2020-01 .. 2021-12
  early-stop   2022-01 .. 2022-06
  calibration  2022-07 .. 2022-12   (model NEVER sees this)
  test         2023                 (frozen harness, unchanged)

Calibration is Mondrian (per condition group): separate q_hat for
clear (csi >= 0.7) and cloudy (csi < 0.7) points, so cloudy bands widen
more than clear ones.

Run:  python phase3c_recalibrate.py       (~10 min, retrains 9 models)
Overwrites output/xgb_models/*.json and xgb_test_predictions_*.parquet.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

TABLE = "output/training_table.parquet"
TRAIN_END, ES_END, CAL_END = "2022-01-01", "2022-07-01", "2023-01-01"
DAY_ELEV, TARGET = 5.0, 0.80
CSI_SPLIT = 0.7                      # clear/cloudy group boundary

os.makedirs("output/xgb_models", exist_ok=True)
df = pd.read_parquet(TABLE).sort_index()
cap = float(df["capacity_kw"].median())
print(f"Loaded {len(df):,} rows | capacity ~{cap:.2f} kW")

BASE_FEATS = [f for f in
              ["fc_ghi", "fc_dni", "fc_dhi", "fc_temp", "fc_wind", "fc_cloud",
               "fc_rh", "fc_pressure", "fc_rain", "cs_ghi", "csi",
               "sun_elev", "sun_az", "hour_sin", "hour_cos",
               "doy_sin", "doy_cos"] if f in df.columns]

HORIZONS = {
    "15min": ["gen_lag_1", "gen_lag_2", "gen_lag_4", "gen_lag_96",
              "gen_roll_mean_4"],
    "1h":    ["gen_lag_4", "gen_lag_96"],
    "24h":   ["gen_lag_96", "gen_lag_192"],
}

PARAMS = dict(n_estimators=1500, max_depth=7, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              early_stopping_rounds=50, n_jobs=-1, random_state=42)

def q_hat_grouped(cal_day: pd.DataFrame, feats, models) -> dict:
    """Mondrian CQR: one correction per csi group."""
    p10 = models["p10"].predict(cal_day[feats])
    p90 = models["p90"].predict(cal_day[feats])
    y = cal_day["gen_kw"].values
    scores = np.maximum(p10 - y, y - p90)
    out = {}
    for label, mask in [("clear", cal_day["csi"].values >= CSI_SPLIT),
                        ("cloudy", cal_day["csi"].values < CSI_SPLIT)]:
        s = scores[mask]
        n = len(s)
        out[label] = float(np.quantile(s, min(1.0, TARGET*(n+1)/n))) \
            if n >= 50 else float(np.quantile(scores, TARGET))
    return out

results = []
for name, lag_cols in HORIZONS.items():
    feats = BASE_FEATS + [c for c in lag_cols if c in df.columns]
    data = df.dropna(subset=feats + ["gen_kw"])

    tr  = data[data.index < TRAIN_END]
    es  = data[(data.index >= TRAIN_END) & (data.index < ES_END)]
    cal = data[(data.index >= ES_END) & (data.index < CAL_END)]
    te  = data[data.index >= CAL_END].copy()
    print(f"\n--- {name}: train {len(tr):,} | early-stop {len(es):,} | "
          f"cal {len(cal):,} | test {len(te):,} ---")

    models = {}
    for q in (0.1, 0.5, 0.9):
        m = xgb.XGBRegressor(objective="reg:quantileerror",
                             quantile_alpha=q, **PARAMS)
        m.fit(tr[feats], tr["gen_kw"],
              eval_set=[(es[feats], es["gen_kw"])], verbose=False)
        col = f"p{int(q*100)}"
        models[col] = m
        m.save_model(f"output/xgb_models/xgb_{name}_{col}.json")

    # Mondrian conformal on the untouched calibration slice
    cal_day = cal[cal["sun_elev"] > DAY_ELEV]
    qh = q_hat_grouped(cal_day, feats, models)
    print(f"q_hat: clear={qh['clear']:.3f} kW, cloudy={qh['cloudy']:.3f} kW")

    for c in ("p10", "p50", "p90"):
        te[c] = models[c].predict(te[feats])
    adj = np.where(te["csi"] >= CSI_SPLIT, qh["clear"], qh["cloudy"])
    te["p10"] -= adj
    te["p90"] += adj
    for c in ("p10", "p50", "p90"):
        te[c] = te[c].clip(0, cap * 1.05)
        te.loc[te["sun_elev"] <= 0, c] = 0.0
    srt = np.sort(te[["p10", "p50", "p90"]].values, axis=1)
    te["p10"], te["p50"], te["p90"] = srt[:, 0], srt[:, 1], srt[:, 2]

    day = te[te["sun_elev"] > DAY_ELEV]
    e = day["p50"] - day["gen_kw"]
    row = {"horizon": name,
           "nMAE_%": 100*e.abs().mean()/cap,
           "nRMSE_%": 100*np.sqrt((e**2).mean())/cap,
           "bias_%": 100*e.mean()/cap,
           "coverage_%": 100*((day["gen_kw"] >= day["p10"]) &
                              (day["gen_kw"] <= day["p90"])).mean(),
           "band_width_%cap": 100*(day["p90"]-day["p10"]).mean()/cap}
    for grp, mask in [("clear", day["csi"] >= CSI_SPLIT),
                      ("cloudy", day["csi"] < CSI_SPLIT)]:
        g = day[mask]
        row[f"cover_{grp}_%"] = 100*((g["gen_kw"] >= g["p10"]) &
                                     (g["gen_kw"] <= g["p90"])).mean()
    results.append(row)

    te[["gen_kw", "sun_elev", "csi", "p10", "p50", "p90"]].to_parquet(
        f"output/xgb_test_predictions_{name}.parquet")

res = pd.DataFrame(results).set_index("horizon").round(2)
print("\n=== PHASE 3c RESULTS (test 2023, daytime, frozen harness) ===")
print(res.to_string())
res.to_csv("output/xgb_results_calibrated.csv")
print("\nNote: nMAE may tick up slightly vs Phase 3 -- the model now trains "
      "on 2 years instead of 2.5. That is the honest price of clean "
      "calibration. Coverage (overall AND per group) should now be ~80%.")
