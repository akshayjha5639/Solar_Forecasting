"""
Solar Forecasting Dashboard -- Phase 2 baselines viewer.

Run from the project root:
    pip install streamlit plotly
    streamlit run dashboard.py

Reads output/training_table.parquet and shows:
  - Overview: headline metrics for each baseline (frozen-harness rules)
  - Forecast explorer: pick any date, see actual vs baseline predictions
  - Data health: daily energy, coverage, clear/cloudy split
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Forecasting", page_icon="☀️",
                   layout="wide")

TABLE = "output/training_table.parquet"
TEST_START, DAY_ELEV = "2023-01-01", 5.0

# DKASC site / array constants (must match phase2_baselines.py)
LAT, LON, TZ = -23.7624, 133.8754, "Australia/Darwin"
TILT, AZIMUTH, GAMMA_PDC = 23.8, 0.0, -0.004


# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading training table...")
def load_data() -> tuple[pd.DataFrame, float]:
    df = pd.read_parquet(TABLE).sort_index()
    cap = float(df["capacity_kw"].median())

    # baseline 1 & 2: persistence / smart persistence (24h)
    df["Persistence (24h)"] = df["gen_lag_96"]
    ratio = (df["cs_ghi"] / df["cs_ghi"].shift(96).clip(lower=1)).clip(0, 2)
    df["Smart persistence (24h)"] = (df["gen_lag_96"] * ratio).clip(0, cap*1.1)
    df["Persistence (15min)"] = df["gen_lag_1"]

    # baseline 3: physics chain (needs pvlib; skipped gracefully if missing)
    try:
        import pvlib
        loc = pvlib.location.Location(LAT, LON, tz=TZ)
        sp = loc.get_solarposition(df.index)
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=TILT, surface_azimuth=AZIMUTH,
            solar_zenith=sp["apparent_zenith"], solar_azimuth=sp["azimuth"],
            dni=df["fc_dni"].fillna(0), ghi=df["fc_ghi"].fillna(0),
            dhi=df["fc_dhi"].fillna(0))["poa_global"].clip(lower=0)
        t_cell = pvlib.temperature.faiman(poa, df["fc_temp"].fillna(25),
                                          df["fc_wind"].fillna(1))
        pdc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=cap*1.1,
                                        gamma_pdc=GAMMA_PDC)
        df["Physics (PVLib)"] = (pdc * 0.96).clip(0, cap*1.05)
        df.loc[df["sun_elev"] <= 0, "Physics (PVLib)"] = 0.0
    except ImportError:
        st.warning("pvlib not installed -- physics baseline hidden")
    return df, cap


MODEL_COLS = ["Persistence (24h)", "Smart persistence (24h)",
              "Physics (PVLib)", "Persistence (15min)"]


def metrics(df: pd.DataFrame, col: str, cap: float) -> dict | None:
    """Frozen harness: 2023 test year, daytime only, % of capacity."""
    t = df[(df.index >= TEST_START) & (df["sun_elev"] > DAY_ELEV)]
    t = t.dropna(subset=[col, "gen_kw"])
    if t.empty:
        return None
    e = t[col] - t["gen_kw"]
    return {"nMAE %": 100*e.abs().mean()/cap,
            "nRMSE %": 100*np.sqrt((e**2).mean())/cap,
            "bias %": 100*e.mean()/cap, "points": len(t)}


# ----------------------------------------------------------------------------
df, cap = load_data()
models = [m for m in MODEL_COLS if m in df.columns]

st.title("☀️ Solar Generation Forecasting — Baseline Models")
st.caption(f"DKASC Alice Springs · {df.index.min():%d %b %Y} → "
           f"{df.index.max():%d %b %Y} · effective capacity "
           f"{cap:.2f} kW · test year 2023, daytime only")

tab_overview, tab_explorer, tab_health = st.tabs(
    ["📊 Model performance", "🔍 Forecast explorer", "🩺 Data health"])

# ---------------------------------------------------------------- overview --
with tab_overview:
    rows = {m: r for m in models if (r := metrics(df, m, cap))}
    res = pd.DataFrame(rows).T

    cols = st.columns(len(res))
    for c, (name, r) in zip(cols, res.iterrows()):
        c.metric(name, f"{r['nMAE %']:.2f}% nMAE",
                 f"bias {r['bias %']:+.2f}%", delta_color="off")

    fig = go.Figure()
    fig.add_bar(x=res.index, y=res["nMAE %"], name="nMAE %",
                marker_color="#f39c12")
    fig.add_bar(x=res.index, y=res["nRMSE %"], name="nRMSE %",
                marker_color="#3498db")
    fig.update_layout(barmode="group", height=380,
                      yaxis_title="% of capacity",
                      title="Error by baseline (lower = better)")
    st.plotly_chart(fig, use_container_width=True)

    # clear vs cloudy breakdown
    t = df[(df.index >= TEST_START) & (df["sun_elev"] > DAY_ELEV)]
    daily_csi = t.groupby(t.index.date)["csi"].mean()
    split = {"Clear days (CSI ≥ 0.8)": set(daily_csi[daily_csi >= .8].index),
             "Cloudy days (CSI < 0.5)": set(daily_csi[daily_csi < .5].index)}
    rows = []
    for label, days in split.items():
        sub = t[pd.Series(t.index.date, index=t.index).isin(days)]
        row = {"condition": f"{label} — {len(days)} days"}
        for m in models:
            s = sub.dropna(subset=[m])
            row[m] = round(100*(s[m]-s["gen_kw"]).abs().mean()/cap, 2)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("condition"),
                 use_container_width=True)
    st.info("Physics stays accurate on cloudy days because it uses the "
            "weather forecast; persistence assumes tomorrow repeats today. "
            "XGBoost (Phase 3) will combine both signals.")

# ---------------------------------------------------------------- explorer --
import os

@st.cache_data
def load_xgb(horizon: str) -> pd.DataFrame | None:
    p = f"output/xgb_test_predictions_{horizon}.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else None

with tab_explorer:
    c1, c2, c3 = st.columns([2, 2, 3])
    day = c1.date_input("Date", value=pd.Timestamp("2023-03-08").date(),
                        min_value=df.index.min().date(),
                        max_value=df.index.max().date())
    span = c2.radio("Window", ["1 day", "3 days", "7 days"], horizontal=True)
    shown = c3.multiselect("Models", models,
                           default=[m for m in
                                    ("Physics (PVLib)",
                                     "Smart persistence (24h)")
                                    if m in models])

    xgb_available = [h for h in ("15min", "1h", "24h")
                     if load_xgb(h) is not None]
    xgb_pick = None
    if xgb_available:
        xgb_pick = st.selectbox(
            "Overlay XGBoost forecast (Phase 3, test-year 2023 only)",
            ["(none)"] + xgb_available)
        if xgb_pick == "(none)":
            xgb_pick = None

    n = {"1 day": 1, "3 days": 3, "7 days": 7}[span]
    start = pd.Timestamp(day, tz=df.index.tz)
    wk = df.loc[start:start + pd.Timedelta(days=n)]

    fig = go.Figure()
    if xgb_pick:
        xp = load_xgb(xgb_pick).loc[start:start + pd.Timedelta(days=n)]
        if len(xp):
            fig.add_scatter(x=xp.index, y=xp["p90"], line=dict(width=0),
                            showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=xp.index, y=xp["p10"], fill="tonexty",
                            fillcolor="rgba(243,156,18,0.25)",
                            line=dict(width=0),
                            name=f"XGB {xgb_pick} P10-P90")
            fig.add_scatter(x=xp.index, y=xp["p50"],
                            line=dict(color="#e67e22", width=2),
                            name=f"XGB {xgb_pick} P50")
        else:
            st.caption("XGBoost predictions cover 2023 only -- "
                       "pick a 2023 date to see them.")
    fig.add_scatter(x=wk.index, y=wk["gen_kw"], name="Actual",
                    line=dict(color="black", width=2))
    for m in shown:
        fig.add_scatter(x=wk.index, y=wk[m], name=m, line=dict(dash="dash"))
    fig.update_layout(height=420, yaxis_title="kW",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    day_rows = wk[wk["sun_elev"] > DAY_ELEV]
    if len(day_rows):
        stats = []
        for m in shown:
            s = day_rows.dropna(subset=[m])
            if len(s):
                stats.append({
                    "model": m,
                    "window nMAE %": round(
                        100*(s[m]-s["gen_kw"]).abs().mean()/cap, 2),
                    "actual energy kWh": round(s["gen_kw"].sum()*0.25, 1),
                    "predicted energy kWh": round(s[m].sum()*0.25, 1)})
        st.dataframe(pd.DataFrame(stats).set_index("model"),
                     use_container_width=True)

# ------------------------------------------------------------------ health --
with tab_health:
    daily = (df["gen_kw"].resample("D").sum() * 0.25)  # kWh/day
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily.values, mode="lines",
                    line=dict(color="#e67e22", width=1),
                    name="daily energy")
    fig.add_scatter(x=daily.index, y=daily.rolling(30).mean(),
                    name="30-day mean", line=dict(color="#2c3e50", width=2))
    fig.update_layout(height=380, yaxis_title="kWh / day",
                      title="Daily energy across the full dataset")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Days", f"{df.index.normalize().nunique():,}")
    c3.metric("Peak output", f"{df['gen_kw'].max():.2f} kW")
    c4.metric("Capacity factor",
              f"{100*df['gen_kw'].mean()/cap:.1f}%")
    st.caption("Capacity factor = mean output / effective capacity, "
               "including nights. Typical fixed-tilt desert PV: 20-25%.")