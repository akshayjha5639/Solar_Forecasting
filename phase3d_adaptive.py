"""
Phase 3d: Adaptive Conformal Inference (ACI) -- online band calibration.

Why: static calibration (Phase 3c) stalled at ~75-77% coverage because the
calibration slice (Jul-Dec 2022) cannot represent all 2023 seasons.

Method (Gibbs & Candes, 2021, simplified per-group variant):
  Keep a per-group additive correction q_t. Each evening, look at today's
  daytime points: if fewer than 80% fell inside [P10-q, P90+q], increase q;
  if more, decrease it. Only past data is ever used -> no leakage. This is
  the same mechanism a production system runs continuously.

Run after phase3c_recalibrate.py (uses its saved models):
    python phase3d_adaptive.py
Overwrites xgb_test_predictions_*.parquet with ACI bands.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

TABLE = "output/training_table.parquet"
ES_END, CAL_END = "2022-07-01", "2023-01-01"
DAY_ELEV, TARGET, CSI_SPLIT = 5.0, 0.80, 0.7

df = pd.read_parquet(TABLE).sort_index()
cap = float(df["capacity_kw"].median())
ETA = 0.02 * cap          # daily adaptation step (kW); ~2% of capacity

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

def load(name, col):
    p = f"output/xgb_models/xgb_{name}_{col}.json"
    if not os.path.exists(p):
        sys.exit(f"{p} missing -- run phase3c_recalibrate.py first")
    m = xgb.XGBRegressor(); m.load_model(p); return m

results = []
for name, lag_cols in HORIZONS.items():
    feats = BASE_FEATS + [c for c in lag_cols if c in df.columns]
    data = df.dropna(subset=feats + ["gen_kw"])
    cal = data[(data.index >= ES_END) & (data.index < CAL_END)]
    te = data[data.index >= CAL_END].copy()
    models = {c: load(name, c) for c in ("p10", "p50", "p90")}

    # raw quantile predictions
    for c in ("p10", "p50", "p90"):
        te[c] = models[c].predict(te[feats])

    # initialise q per group from static calibration (warm start)
    cd = cal[cal["sun_elev"] > DAY_ELEV]
    s = np.maximum(models["p10"].predict(cd[feats]) - cd["gen_kw"].values,
                   cd["gen_kw"].values - models["p90"].predict(cd[feats]))
    grp_mask = cd["csi"].values >= CSI_SPLIT
    q = {"clear": float(np.quantile(s[grp_mask], TARGET)),
         "cloudy": float(np.quantile(s[~grp_mask], TARGET))}

    # ---- online day-by-day adaptation over the test year ----
    te["grp"] = np.where(te["csi"] >= CSI_SPLIT, "clear", "cloudy")
    te["adj"] = 0.0
    q_hist = []
    for d, day_rows in te.groupby(te.index.date):
        # 1. apply CURRENT q to today's forecasts (issued before today)
        adj = day_rows["grp"].map(q).values
        te.loc[day_rows.index, "adj"] = adj

        # 2. after the day ends, update q from today's daytime outcomes
        obs = day_rows[day_rows["sun_elev"] > DAY_ELEV]
        for g in ("clear", "cloudy"):
            og = obs[obs["grp"] == g]
            if len(og) < 4:
                continue
            inside = ((og["gen_kw"] >= og["p10"] - q[g]) &
                      (og["gen_kw"] <= og["p90"] + q[g])).mean()
            q[g] += ETA * (TARGET - inside)      # widen if under-covering
            q[g] = max(q[g], 0.0)
        q_hist.append({"date": d, **q})

    te["p10"] = (te["p10"] - te["adj"])
    te["p90"] = (te["p90"] + te["adj"])
    for c in ("p10", "p50", "p90"):
        te[c] = te[c].clip(0, cap * 1.05)
        te.loc[te["sun_elev"] <= 0, c] = 0.0
    srt = np.sort(te[["p10", "p50", "p90"]].values, axis=1)
    te["p10"], te["p50"], te["p90"] = srt[:, 0], srt[:, 1], srt[:, 2]

    day = te[te["sun_elev"] > DAY_ELEV]
    cov = 100 * ((day["gen_kw"] >= day["p10"]) &
                 (day["gen_kw"] <= day["p90"])).mean()
    h1 = day[day.index < "2023-07-01"]
    h2 = day[day.index >= "2023-07-01"]
    cov1 = 100 * ((h1["gen_kw"] >= h1["p10"]) & (h1["gen_kw"] <= h1["p90"])).mean()
    cov2 = 100 * ((h2["gen_kw"] >= h2["p10"]) & (h2["gen_kw"] <= h2["p90"])).mean()
    e = day["p50"] - day["gen_kw"]
    results.append({"horizon": name,
                    "nMAE_%": 100*e.abs().mean()/cap,
                    "coverage_%": cov,
                    "cov_H1_2023_%": cov1, "cov_H2_2023_%": cov2,
                    "band_width_%cap": 100*(day["p90"]-day["p10"]).mean()/cap,
                    "final_q_clear_kW": round(q["clear"], 3),
                    "final_q_cloudy_kW": round(q["cloudy"], 3)})

    te[["gen_kw", "sun_elev", "csi", "p10", "p50", "p90"]].to_parquet(
        f"output/xgb_test_predictions_{name}.parquet")
    pd.DataFrame(q_hist).to_csv(f"output/aci_q_history_{name}.csv",
                                index=False)

res = pd.DataFrame(results).set_index("horizon").round(2)
print("\n=== PHASE 3d: ADAPTIVE CONFORMAL (test 2023, daytime) ===")
print(res.to_string())
res.to_csv("output/xgb_results_adaptive.csv")
print("\nCoverage should now track ~80%, with H2 closer than H1 "
      "(the adapter needs a few weeks to converge). q history saved to "
      "output/aci_q_history_*.csv -- plot it to watch the seasonal drift.")
