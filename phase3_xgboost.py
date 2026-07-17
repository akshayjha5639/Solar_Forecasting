"""
Phase 3: XGBoost quantile models (P10/P50/P90) per horizon.

Run after Phase 2:
    python phase3_xgboost.py

Horizons and leakage rule:
  a model issued at time (t - h) predicting time t may only use lag features
  with lag >= h. Weather forecast + solar geometry + time encodings at time t
  are always allowed (they are known in advance).

    15min  (h = 1 block ) : lags 1, 2, 4, 96 + rolling
    1h     (h = 4 blocks) : lags 4, 96
    24h    (h = 96 blocks): lags 96, 192

Split: train 2020 - 2022-06, early-stop validation 2022-07 - 2022-12,
       TEST 2023 (frozen harness, identical to Phase 2).
Outputs: output/xgb_results.csv, output/xgb_models/*.json,
         output/xgb_sample_week.png
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    sys.exit("pip install xgboost>=2.0")

TABLE = "output/training_table.parquet"
if not os.path.exists(TABLE):
    sys.exit("run fetch_dkasc_openmeteo.py first")
os.makedirs("output/xgb_models", exist_ok=True)

TEST_START, VAL_START = "2023-01-01", "2022-07-01"
DAY_ELEV = 5.0
QUANTILES = (0.1, 0.5, 0.9)

df = pd.read_parquet(TABLE).sort_index()
cap = float(df["capacity_kw"].median())
print(f"Loaded {len(df):,} rows | capacity ~{cap:.2f} kW")

BASE_FEATS = ["fc_ghi", "fc_dni", "fc_dhi", "fc_temp", "fc_wind", "fc_cloud",
              "fc_rh", "fc_pressure", "fc_rain", "cs_ghi", "csi",
              "sun_elev", "sun_az", "hour_sin", "hour_cos",
              "doy_sin", "doy_cos"]
BASE_FEATS = [f for f in BASE_FEATS if f in df.columns]

HORIZONS = {                      # horizon name -> (blocks, allowed lag cols)
    "15min": (1,  ["gen_lag_1", "gen_lag_2", "gen_lag_4", "gen_lag_96",
                   "gen_roll_mean_4"]),
    "1h":    (4,  ["gen_lag_4", "gen_lag_96"]),
    "24h":   (96, ["gen_lag_96", "gen_lag_192"]),
}

PARAMS = dict(n_estimators=1500, max_depth=7, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              early_stopping_rounds=50, n_jobs=-1, random_state=42)

def frozen_eval(t: pd.DataFrame, pred: str) -> dict:
    """Identical rules to Phase 2: test 2023, daytime only, nMAE/nRMSE/bias."""
    m = t[(t.index >= TEST_START) & (t["sun_elev"] > DAY_ELEV)]
    m = m.dropna(subset=[pred, "gen_kw"])
    e = m[pred] - m["gen_kw"]
    return {"n": len(m),
            "nMAE_%": 100 * e.abs().mean() / cap,
            "nRMSE_%": 100 * np.sqrt((e**2).mean()) / cap,
            "bias_%": 100 * e.mean() / cap}

results, preds_test = [], {}
for name, (h, lag_cols) in HORIZONS.items():
    feats = BASE_FEATS + [c for c in lag_cols if c in df.columns]
    data = df.dropna(subset=feats + ["gen_kw"]).copy()

    tr = data[data.index < VAL_START]
    va = data[(data.index >= VAL_START) & (data.index < TEST_START)]
    te = data[data.index >= TEST_START].copy()
    print(f"\n--- {name}: {len(feats)} feats | train {len(tr):,} "
          f"val {len(va):,} test {len(te):,} ---")

    for q in QUANTILES:
        m = xgb.XGBRegressor(objective="reg:quantileerror",
                             quantile_alpha=q, **PARAMS)
        m.fit(tr[feats], tr["gen_kw"],
              eval_set=[(va[feats], va["gen_kw"])], verbose=False)
        col = f"p{int(q*100)}"
        te[col] = m.predict(te[feats])
        m.save_model(f"output/xgb_models/xgb_{name}_{col}.json")

    # post-process: clip, night = 0, enforce P10 <= P50 <= P90
    for col in ("p10", "p50", "p90"):
        te[col] = te[col].clip(0, cap * 1.05)
        te.loc[te["sun_elev"] <= 0, col] = 0.0
    q = np.sort(te[["p10", "p50", "p90"]].values, axis=1)
    te["p10"], te["p50"], te["p90"] = q[:, 0], q[:, 1], q[:, 2]

    r = frozen_eval(te, "p50")
    day = te[te["sun_elev"] > DAY_ELEV]
    r["coverage_P10-P90_%"] = 100 * ((day["gen_kw"] >= day["p10"]) &
                                     (day["gen_kw"] <= day["p90"])).mean()
    r["horizon"] = name
    results.append(r)
    preds_test[name] = te

    # save test predictions so the dashboard can overlay them
    te[["gen_kw", "sun_elev", "csi", "p10", "p50", "p90"]].to_parquet(
        f"output/xgb_test_predictions_{name}.parquet")

    # top features (sanity: irradiance/csi/lags should dominate)
    imp = pd.Series(m.feature_importances_, index=feats).nlargest(5)
    print("top features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.items()))

res = pd.DataFrame(results).set_index("horizon").round(2)

# skill vs Phase 2 benchmarks (edit if your baseline numbers differ)
BENCH = {"15min": 5.93, "1h": 5.93, "24h": 9.80}   # smart-pers / pers_15m nMAE
res["benchmark_nMAE_%"] = pd.Series(BENCH)
res["skill_%"] = (100 * (1 - res["nMAE_%"] / res["benchmark_nMAE_%"])).round(1)

print("\n=== XGBOOST RESULTS (test 2023, daytime, frozen harness) ===")
print(res.to_string())
res.to_csv("output/xgb_results.csv")

# condition breakdown for the 24h model
te = preds_test["24h"]
day = te[te["sun_elev"] > DAY_ELEV]
daily_csi = day.groupby(day.index.date)["csi"].mean()
for label, days in [("CLEAR", set(daily_csi[daily_csi >= 0.8].index)),
                    ("CLOUDY", set(daily_csi[daily_csi < 0.5].index))]:
    sub = day[pd.Series(day.index.date, index=day.index).isin(days)]
    if len(sub):
        e = (sub["p50"] - sub["gen_kw"]).abs().mean() / cap * 100
        print(f"24h model, {label:6s} days ({len(days):3d}): nMAE {e:5.2f}%")

# sample week plot with uncertainty band
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wk = preds_test["24h"].loc["2023-03-06":"2023-03-12"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(wk.index, wk["p10"], wk["p90"], alpha=0.3,
                    label="P10-P90")
    ax.plot(wk.index, wk["p50"], lw=1.2, label="P50 forecast")
    ax.plot(wk.index, wk["gen_kw"], "k-", lw=1, label="actual")
    ax.set_ylabel("kW"); ax.legend()
    ax.set_title("XGBoost 24h-ahead forecast, sample week")
    fig.tight_layout()
    fig.savefig("output/xgb_sample_week.png", dpi=120)
    print("\nPlot -> output/xgb_sample_week.png")
except ImportError:
    pass

print("\nPhase 3 done. Compare nMAE against Phase 2: "
      "24h must beat physics (8.45%), 15min must beat pers_15m (5.93%).")
