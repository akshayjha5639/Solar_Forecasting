"""
Solar Forecasting -- Client Demo App
Tab 1: New Site Forecast  -- enter ANY location + system details, get live
        all-horizon forecasts (physics engine + live Open-Meteo weather).
Tab 2: Connected Site (ML) -- the DKASC site replayed as a connected
        customer: XGBoost calibrated forecasts vs actual, adjustable system
        size (no retraining needed), and the day-1 -> week-3 accuracy story.

Run:  streamlit run demo_app.py     (needs internet for Tab 1)
"""

import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Forecasting Demo", page_icon="☀️",
                   layout="wide")

try:
    import pvlib
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False

INV_EFF = 0.96

# ============================================================ helpers ======
@st.cache_data(ttl=900, show_spinner="Fetching live weather forecast...")
def fetch_weather(lat: float, lon: float) -> pd.DataFrame:
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "shortwave_radiation,direct_normal_irradiance,"
                  "diffuse_radiation,temperature_2m,wind_speed_10m,"
                  "cloud_cover",
        "forecast_days": 7, "timezone": "auto"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    w = pd.DataFrame(j["hourly"]).rename(columns={
        "time": "ts", "shortwave_radiation": "ghi",
        "direct_normal_irradiance": "dni", "diffuse_radiation": "dhi",
        "temperature_2m": "temp", "wind_speed_10m": "wind",
        "cloud_cover": "cloud"})
    w["ts"] = pd.to_datetime(w["ts"])
    w = w.set_index("ts").tz_localize(j["timezone"])
    return w.resample("15min").interpolate("linear")

@st.cache_data(ttl=3600)
def geocode(name: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": 1}, timeout=15)
    res = r.json().get("results")
    if res:
        return res[0]["latitude"], res[0]["longitude"], \
            f"{res[0]['name']}, {res[0].get('country','')}"
    return None

def physics_forecast(w, lat, lon, tilt, azim, cap_dc, inv_cap):
    loc = pvlib.location.Location(lat, lon, tz=str(w.index.tz))
    sp = loc.get_solarposition(w.index)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt, surface_azimuth=azim,
        solar_zenith=sp["apparent_zenith"], solar_azimuth=sp["azimuth"],
        dni=w["dni"].fillna(0), ghi=w["ghi"].fillna(0),
        dhi=w["dhi"].fillna(0))["poa_global"].clip(lower=0)
    t_cell = pvlib.temperature.faiman(poa, w["temp"].fillna(25),
                                      w["wind"].fillna(1))
    dc = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0=cap_dc,
                                   gamma_pdc=-0.004)
    ac = np.minimum(dc * INV_EFF, inv_cap)
    out = pd.DataFrame({"kw": ac, "cloud": w["cloud"],
                        "elev": sp["apparent_elevation"]}, index=w.index)
    out.loc[out["elev"] <= 0, "kw"] = 0.0
    # heuristic uncertainty: wider band when cloudier (physics has no ML band)
    width = (0.08 + 0.22 * (out["cloud"].fillna(0) / 100)) * out["kw"]
    out["lo"], out["hi"] = (out["kw"] - width).clip(lower=0), out["kw"] + width
    return out

# ============================================================== layout =====
st.title("☀️ Solar Generation Forecasting")
tab_new, tab_ml = st.tabs(["🆕 New Site Forecast (any location, live)",
                           "🔗 Connected Site — ML model"])

# ------------------------------------------------- TAB 1: new site, live --
with tab_new:
    st.caption("Day-1 experience for any customer: no sensors, no history — "
               "location + system details produce live forecasts for every "
               "horizon.")
    c = st.columns([3, 2, 2, 2])
    place = c[0].text_input("Location (city / address)", "Jamshedpur, India")
    n_panels = c[1].number_input("Panel count", 1, 100000, 12)
    watt = c[2].number_input("Panel rating (W)", 100, 800, 550, step=10)
    tariff = c[3].number_input("Tariff (₹/kWh)", 0.0, 50.0, 8.0, step=0.5)

    c2 = st.columns([2, 2, 2, 3])
    cap_dc = n_panels * watt / 1000
    inv_cap = c2[0].number_input("Inverter capacity (kW)", 0.5, 10000.0,
                                 round(cap_dc * 0.95, 1))
    tilt_in = c2[1].number_input("Tilt (°, 0=auto)", 0.0, 90.0, 0.0)
    azim_in = c2[2].number_input("Azimuth (°, -1=auto)", -1.0, 360.0, -1.0)
    c2[3].metric("System size (DC)", f"{cap_dc:.2f} kW")

    if st.button("Generate forecast", type="primary"):
        if not HAS_PVLIB:
            st.error("pvlib not installed")
            st.stop()
        g = geocode(place)
        if not g:
            st.error("Location not found")
            st.stop()
        lat, lon, label = g
        tilt = tilt_in if tilt_in > 0 else abs(lat)
        azim = azim_in if azim_in >= 0 else (180.0 if lat >= 0 else 0.0)
        w = fetch_weather(lat, lon)
        f = physics_forecast(w, lat, lon, tilt, azim, cap_dc, inv_cap)
        now = pd.Timestamp.now(tz=f.index.tz).floor("15min")
        fut = f[f.index >= now]

        st.success(f"Live forecast for **{label}**  ({lat:.3f}, {lon:.3f}) "
                   f"· tilt {tilt:.0f}° · azimuth {azim:.0f}°")

        # headline horizon cards
        def kwh(s):
            return s["kw"].sum() * 0.25
        h15 = fut.iloc[:1]["kw"].iloc[0] if len(fut) else 0
        e1h = kwh(fut.iloc[:4])
        e24 = kwh(fut.iloc[:96])
        e7d = kwh(fut)
        m = st.columns(5)
        m[0].metric("Next 15 min", f"{h15:.2f} kW")
        m[1].metric("Next 1 h", f"{e1h:.2f} kWh")
        m[2].metric("Next 24 h", f"{e24:.1f} kWh", f"₹{e24*tariff:,.0f}")
        m[3].metric("Next 7 days", f"{e7d:.0f} kWh", f"₹{e7d*tariff:,.0f}")
        m[4].metric("Capacity factor (7d)",
                    f"{100*e7d/(cap_dc*24*7):.1f}%")

        fig = go.Figure()
        nxt = fut.iloc[:192]           # 48h detail
        fig.add_scatter(x=nxt.index, y=nxt["hi"], line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=nxt.index, y=nxt["lo"], fill="tonexty",
                        fillcolor="rgba(243,156,18,.25)",
                        line=dict(width=0), name="uncertainty")
        fig.add_scatter(x=nxt.index, y=nxt["kw"],
                        line=dict(color="#e67e22", width=2),
                        name="forecast kW")
        fig.update_layout(height=380, yaxis_title="kW",
                          title="Next 48 hours (15-min resolution)")
        st.plotly_chart(fig, use_container_width=True)

        daily = fut["kw"].resample("D").sum() * 0.25
        fig2 = go.Figure(go.Bar(x=daily.index.strftime("%a %d %b"),
                                y=daily.values, marker_color="#f39c12",
                                text=[f"₹{v*tariff:,.0f}" for v in daily],
                                textposition="outside"))
        fig2.update_layout(height=320, yaxis_title="kWh / day",
                           title="7-day energy & revenue")
        st.plotly_chart(fig2, use_container_width=True)
        st.info("This site is now on the **physics engine**. Once live "
                "generation data connects, the ML models take over and "
                "accuracy improves by ~30-40% — see the next tab.")

# --------------------------------------------- TAB 2: connected site, ML --
with tab_ml:
    st.caption("A customer 3 weeks after connecting their inverter: the "
               "trained ML model (XGBoost + adaptive calibration) replayed "
               "on a real site with known ground truth.")
    P24 = "output/xgb_test_predictions_24h.parquet"
    if not os.path.exists(P24):
        st.warning("Run phase3d_adaptive.py first to create prediction "
                   "files.")
        st.stop()

    @st.cache_data
    def load_preds():
        return {h: pd.read_parquet(f"output/xgb_test_predictions_{h}"
                                   ".parquet")
                for h in ("15min", "1h", "24h")}
    preds = load_preds()
    site_cap = 4.15                     # trained site effective capacity

    c = st.columns([2, 2, 2, 2])
    day = c[0].date_input("Forecast date (2023 replay)",
                          pd.Timestamp("2023-03-08").date(),
                          min_value=pd.Timestamp("2023-01-02").date(),
                          max_value=pd.Timestamp("2023-12-30").date())
    n_p = c[1].number_input("Panel count", 1, 100000, 10, key="mlp")
    w_p = c[2].number_input("Panel rating (W)", 100, 800, 415, step=5,
                            key="mlw")
    trf = c[3].number_input("Tariff (₹/kWh)", 0.0, 50.0, 8.0, step=0.5,
                            key="mlt")
    user_cap = n_p * w_p / 1000
    scale = user_cap / site_cap
    st.caption(f"System size {user_cap:.2f} kW → outputs scaled ×{scale:.2f} "
               "from the reference array. **No retraining needed** — system "
               "size is a scaling parameter, not a model input.")

    hor = st.radio("Horizon", ["24h", "1h", "15min"], horizontal=True)
    t = preds[hor]
    start = pd.Timestamp(day, tz=t.index.tz)
    d = t.loc[start:start + pd.Timedelta(days=1)]
    if d.empty:
        st.warning("No data for this date (filtered day) — pick another.")
        st.stop()

    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d["p90"]*scale, line=dict(width=0),
                    showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=d.index, y=d["p10"]*scale, fill="tonexty",
                    fillcolor="rgba(52,152,219,.25)", line=dict(width=0),
                    name="P10–P90 (calibrated 80% band)")
    fig.add_scatter(x=d.index, y=d["p50"]*scale,
                    line=dict(color="#2980b9", width=2),
                    name=f"ML forecast ({hor} ahead)")
    fig.add_scatter(x=d.index, y=d["gen_kw"]*scale,
                    line=dict(color="black", width=2), name="Actual")
    fig.update_layout(height=400, yaxis_title="kW",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    dd = d[d["sun_elev"] > 5]
    e_act = dd["gen_kw"].sum()*0.25*scale
    e_p50 = dd["p50"].sum()*0.25*scale
    e_lo, e_hi = dd["p10"].sum()*0.25*scale, dd["p90"].sum()*0.25*scale
    err = 100*(dd["p50"]-dd["gen_kw"]).abs().mean()/site_cap
    m = st.columns(4)
    m[0].metric("Forecast energy", f"{e_p50:.1f} kWh",
                f"range {e_lo:.1f}–{e_hi:.1f}")
    m[1].metric("Actual energy", f"{e_act:.1f} kWh")
    m[2].metric("Revenue (expected)", f"₹{e_p50*trf:,.0f}",
                f"₹{e_lo*trf:,.0f}–₹{e_hi*trf:,.0f}")
    m[3].metric("This-day error (nMAE)", f"{err:.1f}%")

    st.divider()
    st.subheader("Why connecting data matters — the accuracy upgrade")
    a, b, cc = st.columns(3)
    a.metric("Day 1 · physics only", "8.45% error",
             help="No site history: physics engine + weather forecast")
    b.metric("~Week 3 · ML active", "5.77% error", "-32%",
             delta_color="inverse")
    cc.metric("Band reliability", "81% coverage",
              help="P10–P90 band contains the actual 8 times in 10 — "
                   "verified on a full held-out year")
    st.caption("Errors are normalised by system capacity, measured on a "
               "full held-out test year (2023), daytime only. Short-horizon "
               "gains are larger: 15-min error is 3.4% vs 5.9% baseline "
               "(-43%).")