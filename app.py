"""Crop Yield Monitor - Streamlit app.
Run:  streamlit run app.py
Demo: CROP_DATA_DIR=data_demo CROP_MODEL_DIR=models_demo streamlit run app.py
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import CROPS, N_MONTHS, bundle_predict, month_labels

DATA = Path(os.environ.get("CROP_DATA_DIR", "data"))
MODELS = Path(os.environ.get("CROP_MODEL_DIR", "models"))

st.set_page_config(page_title="Crop Yield Monitor", page_icon="🌾", layout="wide")

need = [DATA / "features_all.csv", MODELS / "metrics.csv", MODELS / "cv_predictions.csv"]
if any(not p.exists() for p in need):
    st.error("Data/model files not found. Run the pipeline first (00 → 01 → 02 → 03), "
             "or try the demo: `python make_demo_data.py`, then the demo commands in README.md.")
    st.stop()
if (DATA / "DEMO_DATA").exists():
    st.warning("⚠️ DEMO MODE: yields are real ICRISAT data but satellite/weather features are "
               "synthetic. Accuracy numbers here are meaningless. Do not quote them.")


# ---------------------------------------------------------------- loaders
@st.cache_data
def load_features(path):
    df = pd.read_csv(path)
    df["district_key"] = df.state + "|" + df.district
    mean_area = df.groupby(["crop", "district_key"]).area_kha.transform("mean")
    df["area_w"] = df.area_kha.fillna(mean_area).fillna(1.0)   # weight for rollups
    return df


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_geojson(path):
    return json.load(open(path)) if Path(path).exists() else None


@st.cache_resource
def load_bundle(path):
    return joblib.load(path) if Path(path).exists() else None


feat_all = load_features(DATA / "features_all.csv")
cv_all = load_csv(MODELS / "cv_predictions.csv")
metrics = load_csv(MODELS / "metrics.csv")
imps = load_csv(MODELS / "importance.csv") if (MODELS / "importance.csv").exists() else pd.DataFrame()
geo = load_geojson(DATA / "districts.geojson")
meta = json.load(open(MODELS / "meta.json")) if (MODELS / "meta.json").exists() else {}


def months_available(df):
    n = 0
    for i in range(1, N_MONTHS + 1):
        if df[f"ndvi_s{i}"].notna().mean() > 0.5:
            n = i
        else:
            break
    return max(n, 1)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌾 Crop Yield Monitor")
crops_here = [c for c in CROPS if c in set(feat_all.crop)]
crop = st.sidebar.selectbox("Crop", crops_here, format_func=lambda c: CROPS[c]["label"])
st.sidebar.caption(CROPS[crop]["season"])
feat = feat_all[feat_all.crop == crop]
years = sorted(feat.year.unique(), reverse=True)
year = st.sidebar.selectbox("Season (sowing year)", years)
maxk = months_available(feat[feat.year == year])
labels = month_labels(crop)
k = st.sidebar.slider("Months of season data used", 1, maxk, maxk,
                      help="Forecast using only the first k months of the season.")
st.sidebar.caption(f"Data through: **{labels[k - 1]}** (month {k} of {N_MONTHS})")
if meta:
    st.sidebar.caption(f"Models trained {meta.get('trained', '?')} on "
                       f"{meta.get('years', ['?', '?'])[0]}–{meta.get('years', ['?', '?'])[1]} yields.")

bundle = load_bundle(MODELS / f"model_{crop}.joblib")
cv = cv_all[cv_all.crop == crop]


# ---------------------------------------------------------------- core snapshot
def ndvi_to_date(df, kk):
    return df[[f"ndvi_s{i}" for i in range(1, kk + 1)]].mean(axis=1)


@st.cache_data(show_spinner=False)
def snapshot(crop_, year_, k_, _bundle, _cv, _feat):
    rows = _feat[_feat.year == year_].copy()
    rows["ndvi_td"] = ndvi_to_date(rows, k_)
    hist = _feat.copy()
    hist["ndvi_td"] = ndvi_to_date(hist, k_)
    st_ = hist.groupby("district_key").ndvi_td.agg(["mean", "std"])
    st_.columns = ["ndvi_norm", "ndvi_sd"]
    rows = rows.merge(st_, left_on="district_key", right_index=True, how="left")
    rows["ndvi_z"] = (rows.ndvi_td - rows.ndvi_norm) / rows.ndvi_sd.replace(0, np.nan)
    c = _cv[(_cv.lead == k_) & (_cv.year == year_)][["district_key", "pred", "lo", "hi", "base"]]
    rows = rows.merge(c.drop_duplicates("district_key"), on="district_key", how="left")
    rows["source"] = np.where(rows.pred.notna(), "backtest (out-of-sample)", "")
    miss = rows.pred.isna()
    if miss.any() and _bundle and k_ in _bundle:
        p = bundle_predict(_bundle, rows[miss], k_)
        for col in ["pred", "lo", "hi", "base"]:
            rows.loc[miss, col] = p[col].values
        rows.loc[miss & rows.pred.notna(), "source"] = "model forecast"
    rows["pct_vs_norm"] = (rows.pred / rows.base - 1) * 100
    return rows


snap = snapshot(crop, year, k, bundle, cv, feat)

st.title(f"{CROPS[crop]['label']} · {year}–{str(year + 1)[-2:]} season")
st.caption("Forecast from satellite NDVI, rainfall and temperature. "
           "Predictions exist only for districts with yield history.")

tab_map, tab_dist, tab_cmp, tab_state, tab_model = st.tabs(
    ["🗺️ Map & alerts", "📍 District", "🔁 Compare seasons", "🏛️ State rollup", "🧪 Model"])


# ---------------------------------------------------------------- tab 1: map
def choropleth(df, col, label, diverging):
    kw = dict(geojson=geo, locations="ADM2_CODE", featureidkey="properties.ADM2_CODE", color=col,
              hover_name="district", hover_data={"state": True, "ADM2_CODE": False},
              opacity=0.8, zoom=3.6, center={"lat": 22.5, "lon": 79},
              labels={col: label}, height=600)
    if diverging:
        kw.update(color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
    else:
        kw.update(color_continuous_scale="YlGn")
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(df, map_style="carto-positron", **kw)
    else:
        fig = px.choropleth_mapbox(df, mapbox_style="carto-positron", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


with tab_map:
    opts = {"Predicted yield (kg/ha)": ("pred", False),
            "Predicted vs district norm (%)": ("pct_vs_norm", True),
            "NDVI anomaly (z-score)": ("ndvi_z", True)}
    choice = st.radio("Colour map by", list(opts), horizontal=True)
    col, div = opts[choice]
    mdf = snap.dropna(subset=[col])
    if geo is None:
        st.info("No districts.geojson found (run 04_export_boundaries.py). Showing a table instead.")
        st.dataframe(mdf[["state", "district", col]].sort_values(col), width="stretch")
    elif mdf.empty:
        st.info("Nothing to show for this selection.")
    else:
        st.plotly_chart(choropleth(mdf, col, choice, div), width="stretch")

    st.subheader("⚠️ Stress alerts")
    z_thr = st.slider("Alert if NDVI z-score below", -3.0, 0.0, -1.0, 0.1)
    p_thr = st.slider("…or predicted yield below district norm by (%)", 0, 40, 10)
    alerts = snap[(snap.ndvi_z < z_thr) | (snap.pct_vs_norm < -p_thr)].sort_values("pct_vs_norm")
    st.write(f"**{len(alerts)}** of {len(snap)} districts flagged.")
    show = alerts[["state", "district", "ndvi_z", "pred", "base", "pct_vs_norm", "source"]].round(2)
    show.columns = ["State", "District", "NDVI z", "Predicted kg/ha", "Norm kg/ha", "% vs norm", "Source"]
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("Download alerts (CSV)", show.to_csv(index=False).encode(),
                       f"alerts_{crop}_{year}_lead{k}.csv", "text/csv")

# ---------------------------------------------------------------- tab 2: district
with tab_dist:
    c1, c2 = st.columns(2)
    states = sorted(feat.state.unique())
    s_sel = c1.selectbox("State", states, key="d_state")
    dists = sorted(feat[feat.state == s_sel].district.unique())
    d_sel = c2.selectbox("District", dists, key="d_dist")
    dkey = f"{s_sel}|{d_sel}"
    dh = feat[feat.district_key == dkey].sort_values("year")
    cur = snap[snap.district_key == dkey]
    if not cur.empty and pd.notna(cur.pred.iloc[0]):
        r = cur.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted yield", f"{r.pred:,.0f} kg/ha")
        m2.metric("80% range", f"{r.lo:,.0f}–{r.hi:,.0f}")
        m3.metric("District norm", f"{r.base:,.0f} kg/ha", f"{r.pct_vs_norm:+.1f}%")
        m4.metric("NDVI z-score", f"{r.ndvi_z:+.2f}" if pd.notna(r.ndvi_z) else "n/a")
        if pd.notna(r.yield_kgha):
            st.caption(f"Actual reported yield for this season: **{r.yield_kgha:,.0f} kg/ha** "
                       f"· prediction source: {r.source}")
    else:
        st.info("No prediction for this district/season (no yield history, or data not yet available).")

    # NDVI curve vs history
    cols = [f"ndvi_s{i}" for i in range(1, N_MONTHS + 1)]
    past = dh[dh.year != year][cols]
    this = dh[dh.year == year][cols]
    fig = go.Figure()
    if len(past) > 2:
        mu, sd = past.mean(), past.std()
        fig.add_trace(go.Scatter(x=labels, y=(mu + sd).values, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=labels, y=(mu - sd).values, fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(120,120,120,0.25)", name="Historical ±1 SD"))
        fig.add_trace(go.Scatter(x=labels, y=mu.values, line=dict(color="grey", dash="dash"),
                                 name="Historical mean"))
    if len(this):
        y_ = this.iloc[0].values.astype(float)
        fig.add_trace(go.Scatter(x=labels, y=y_, line=dict(color="rgba(46,125,50,0.35)"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=labels[:k], y=y_[:k], mode="lines+markers",
                                 line=dict(color="#2e7d32", width=3), name=f"{year} (used by model)"))
    fig.update_layout(title="NDVI through the season", yaxis_title="NDVI", height=380,
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    # Yield history
    hist_y = dh.dropna(subset=["yield_kgha"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist_y.year, y=hist_y.yield_kgha, mode="lines+markers",
                              name="Reported yield", line=dict(color="#1565c0")))
    cvd = cv[(cv.lead == k) & (cv.district_key == dkey)].sort_values("year")
    if len(cvd):
        fig2.add_trace(go.Scatter(
            x=cvd.year, y=cvd.pred, mode="markers", name=f"Backtest forecast ({k} mo.)",
            marker=dict(color="#ef6c00", size=9),
            error_y=dict(type="data", symmetric=False, array=(cvd.hi - cvd.pred).values,
                         arrayminus=(cvd.pred - cvd.lo).values, color="rgba(239,108,0,0.4)")))
    if len(cur) and cur.source.iloc[0] == "model forecast":
        r = cur.iloc[0]
        fig2.add_trace(go.Scatter(
            x=[year], y=[r.pred], mode="markers", name="Current forecast",
            marker=dict(color="#c62828", size=13, symbol="star"),
            error_y=dict(type="data", symmetric=False, array=[r.hi - r.pred],
                         arrayminus=[r.pred - r.lo], color="rgba(198,40,40,0.5)")))
    fig2.update_layout(title="Yield history and forecasts", yaxis_title="kg/ha", height=380,
                       margin=dict(t=50, b=10))
    st.plotly_chart(fig2, width="stretch")
    out = dh.drop(columns=["area_w"]).round(3)
    st.download_button("Download district data (CSV)", out.to_csv(index=False).encode(),
                       f"{crop}_{d_sel}.csv", "text/csv")

# ---------------------------------------------------------------- tab 3: compare seasons
with tab_cmp:
    st.markdown("Compare two seasons side by side, e.g. a drought year against a normal one.")
    yrs_asc = sorted(feat.year.unique())
    c1, c2, c3 = st.columns(3)
    ya = c1.selectbox("Season A", yrs_asc, index=max(len(yrs_asc) - 2, 0), key="ya")
    yb = c2.selectbox("Season B", yrs_asc, index=max(len(yrs_asc) - 1, 0), key="yb")
    scope = c3.selectbox("Region", ["All states"] + sorted(feat.state.unique()), key="scope")
    sc = feat if scope == "All states" else feat[feat.state == scope]

    def season_stats(y):
        d = sc[sc.year == y]
        w = d.area_w
        wavg = lambda s: np.nansum(s * w) / w[s.notna()].sum() if s.notna().any() else np.nan
        act = d.dropna(subset=["yield_kgha"])
        return dict(
            curve=[wavg(d[f"ndvi_s{i}"]) for i in range(1, N_MONTHS + 1)],
            rain=wavg(d[[f"rain_s{i}" for i in range(1, N_MONTHS + 1)]].sum(axis=1, min_count=1)),
            temp=wavg(d[[f"tmean_s{i}" for i in range(1, N_MONTHS + 1)]].mean(axis=1)),
            peak=wavg(d[[f"ndvi_s{i}" for i in range(1, N_MONTHS + 1)]].max(axis=1)),
            yld=(act.yield_kgha * act.area_w).sum() / act.area_w.sum() if len(act) else np.nan)

    A, B = season_stats(ya), season_stats(yb)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=A["curve"], name=str(ya), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=labels, y=B["curve"], name=str(yb), mode="lines+markers"))
    fig.update_layout(title=f"Area-weighted mean NDVI · {scope}", yaxis_title="NDVI", height=360,
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    rows_ = [("Peak NDVI", "peak", "{:.3f}"), ("Season rainfall (mm)", "rain", "{:,.0f}"),
             ("Mean temperature (°C)", "temp", "{:.1f}"), ("Reported yield (kg/ha)", "yld", "{:,.0f}")]
    tbl = pd.DataFrame({"Metric": [r[0] for r in rows_],
                        str(ya): [r[2].format(A[r[1]]) if pd.notna(A[r[1]]) else "n/a" for r in rows_],
                        str(yb): [r[2].format(B[r[1]]) if pd.notna(B[r[1]]) else "n/a" for r in rows_]})
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.caption("Yield is the area-weighted mean over districts with reported yields. "
               "Compare only seasons that had data for the same months.")

# ---------------------------------------------------------------- tab 4: state rollup
with tab_state:
    st.markdown("District predictions rolled up to state level (weighted by cropped area) "
                "and compared with reported state yields.")
    scope2 = st.selectbox("State", ["All states"] + sorted(feat.state.unique()), key="roll")
    m = cv[cv.lead == k].merge(feat[["district_key", "year", "area_w"]], on=["district_key", "year"], how="left")
    if scope2 != "All states":
        m = m[m.state == scope2]
    m["area_w"] = m.area_w.fillna(1.0)
    g = m.groupby("year").apply(
        lambda d: pd.Series({"Reported": (d.actual * d.area_w).sum() / d.area_w.sum(),
                             "Forecast": (d.pred * d.area_w).sum() / d.area_w.sum(),
                             "Districts": len(d)}), include_groups=False).reset_index()
    g["Error %"] = (g.Forecast / g.Reported - 1) * 100
    if g.empty:
        st.info("No backtest data for this selection.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.year, y=g.Reported, name="Reported", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=g.year, y=g.Forecast, name=f"Forecast ({k} mo., out-of-sample)",
                                 mode="lines+markers", line=dict(dash="dot")))
        fig.update_layout(title=f"{scope2} · {CROPS[crop]['label']} yield (kg/ha)", height=380,
                          margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.write(f"Mean absolute state-level error: **{g['Error %'].abs().mean():.1f}%**")
        st.dataframe(g.round(1), hide_index=True, width="stretch")
    fc = snap[snap.source == "model forecast"]
    if scope2 != "All states":
        fc = fc[fc.state == scope2]
    if len(fc):
        w = fc.area_w
        st.success(f"Current-season forecast for {scope2} ({year}, {k} months of data): "
                   f"**{(fc.pred * w).sum() / w.sum():,.0f} kg/ha** across {len(fc)} districts.")
    st.download_button("Download rollup (CSV)", g.round(2).to_csv(index=False).encode(),
                       f"rollup_{crop}_{scope2}.csv", "text/csv")

# ---------------------------------------------------------------- tab 5: model
with tab_model:
    ml = metrics[(metrics.crop == crop) & (metrics.scheme == "year")].sort_values("lead")
    st.subheader("Forecast skill vs. how early in the season")
    st.markdown("Leave-one-year-out backtest: each year is predicted by a model that never saw it. "
                "**Baseline** = the district's own average yield.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ml.lead, y=ml.rmse, name="Model RMSE", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ml.lead, y=ml.base_rmse, name="Baseline RMSE",
                             mode="lines", line=dict(dash="dash", color="grey")))
    fig.update_layout(xaxis_title="Months of season data", yaxis_title="RMSE (kg/ha)", height=340,
                      margin=dict(t=20, b=10), xaxis=dict(tickmode="array", tickvals=ml.lead,
                                                          ticktext=[f"{l} ({labels[l-1]})" for l in ml.lead]))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(ml[["lead", "n", "rmse", "mae", "r2", "base_rmse", "improvement_pct", "coverage80"]]
                 .round(2).rename(columns={"improvement_pct": "improvement vs baseline %",
                                           "coverage80": "80% interval coverage"}),
                 hide_index=True, width="stretch")
    ls = metrics[(metrics.crop == crop) & (metrics.scheme == "state")]
    if len(ls):
        r = ls.iloc[0]
        st.info(f"**Leave-one-state-out** (predicting a state the model never saw, full season): "
                f"RMSE {r.rmse:,.0f} kg/ha, R² {r.r2:.2f}. Baseline here is the global mean "
                f"(RMSE {r.base_rmse:,.0f}), a much weaker baseline than the district average.")
    a, b = st.columns(2)
    sc_df = cv[cv.lead == k]
    f1 = px.scatter(sc_df.sample(min(2500, len(sc_df)), random_state=0), x="actual", y="pred",
                    opacity=0.35, title=f"Predicted vs reported ({k} mo.)", height=380,
                    labels={"actual": "Reported kg/ha", "pred": "Backtest kg/ha"})
    lim = [0, float(sc_df[["actual", "pred"]].max().max())]
    f1.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="red", dash="dash"),
                            showlegend=False))
    a.plotly_chart(f1, width="stretch")
    ip = imps[(imps.crop == crop) & (imps.lead == k)].sort_values("importance") if len(imps) else imps
    if len(ip):
        b.plotly_chart(px.bar(ip, x="importance", y="feature", orientation="h", height=380,
                              title=f"What drives the forecast ({k} mo.)"), width="stretch")
    st.markdown("""**Limitations (read before quoting results)**
- District yields are official statistics with reporting errors; implausible values were removed with simple caps.
- ICRISAT uses fixed 1990-era district boundaries; split districts may be missing or merged.
- MODIS NDVI at 250 m averaged over a static cropland mask mixes crops; kharif seasons suffer monsoon cloud gaps.
- Models learn *anomalies relative to each district's history*; they cannot forecast districts with no yield record.
- The 80% interval is from quantile models; check the coverage column rather than assuming it is calibrated.
- Climate and technology shift over time; the model is only as good as the past resembles the future.""")
