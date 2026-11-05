import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import OUTCOMES
from src.features import FEATURE_LABELS
from src.models import NegBinGLM, RandomForestModel, XGBoostModel

site, outcome, model_name = ui.sidebar_controls()
wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)

st.title("What is driving the predicted risk?")
st.caption(f"{OUTCOMES[outcome]['label']}  ·  {site}  ·  {model_name}")
ui.prototype_banner()

area_fc, site_fc, fitted = ui.get_forecast(key, outcome, model_name, feat)
h = st.radio("Forecast horizon", [1, 2, 3], horizontal=True, format_func=lambda x: f"{x} month(s) ahead")
model, ds = fitted[h]
lab = lambda f: FEATURE_LABELS.get(f, f)


def hbar(df, x, y, colour, title, xlab, err=None):
    fig = go.Figure(go.Bar(x=df[x], y=df[y].map(lab), orientation="h", marker_color=colour,
                           error_x=err))
    fig.update_layout(height=max(300, 26 * len(df)), margin=dict(l=0, r=0, t=30, b=0),
                      title=title, xaxis_title=xlab, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")


if isinstance(model, XGBoostModel):
    live = area_fc[(area_fc.site == site) & (area_fc.horizon == h)]
    rows = ds.loc[(ds.site == site) & ds.y.isna() & (ds.month == ds.month.max())]
    sv = model.shap_values(rows).drop(columns="bias").mean()
    local = sv.sort_values(key=np.abs, ascending=False).head(10).reset_index()
    local.columns = ["feature", "shap"]
    c1, c2 = st.columns(2)
    with c1:
        hbar(local, "shap", "feature", ["#d64545" if v > 0 else "#2e9e5b" for v in local.shap],
             f"This forecast: {site}", "Effect on log expected cases (SHAP)")
        st.caption("Red pushes predicted cases up, green pulls them down, relative to the average area-month.")
    with c2:
        sample = ds[ds.y.notna()].sample(min(1500, ds.y.notna().sum()), random_state=0)
        glob = model.shap_values(sample).drop(columns="bias").abs().mean().sort_values(ascending=False)
        g = glob.head(12).reset_index()
        g.columns = ["feature", "mean_abs"]
        hbar(g, "mean_abs", "feature", "slateblue", "All sites: overall importance", "Mean |SHAP|")

elif isinstance(model, NegBinGLM):
    co = model.coefficients().sort_values("irr", key=lambda s: np.abs(np.log(s)), ascending=False)
    co = co.head(14)
    fig = go.Figure(go.Scatter(
        x=co.irr, y=co.feature.map(lab), mode="markers",
        error_x=dict(type="data", symmetric=False, array=co.irr_hi - co.irr, arrayminus=co.irr - co.irr_lo),
        marker=dict(size=9, color=["#d64545" if v > 1 else "#2e9e5b" for v in co.irr])))
    fig.add_vline(x=1, line_dash="dash", line_color="grey")
    fig.update_layout(height=480, margin=dict(l=0, r=0, t=30, b=0), xaxis_type="log",
                      title="Incidence rate ratio per 1 standard deviation increase (95% CI)",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("A ratio of 1.20 means a one-SD increase in that predictor goes with 20% more cases, "
               "holding the others constant. Lagged case rates are on the log scale.")
    with st.expander("Coefficient table"):
        t = co.assign(feature=co.feature.map(lab))[["feature", "irr", "irr_lo", "irr_hi", "p"]]
        st.dataframe(t.round(3), hide_index=True, width="stretch")

elif isinstance(model, RandomForestModel):
    imp = model.importance().sort_values("importance", ascending=False).head(12)
    hbar(imp, "importance", "feature", "slateblue", "Feature importance (impurity)", "Importance")
    st.caption("Switch to XGBoost for forecast-specific SHAP explanations.")
else:
    st.info("This model has no drivers to show: it simply repeats last year's value for the same "
            "month. It is included as a benchmark the other models must beat.")

st.divider()
st.markdown("**Current climate signals for this site** (latest month)")
latest = feat[(feat.site == site) & (feat.month == feat.month.max())]
sig = latest[["rain_anom_30", "rain_anom_90", "spei_3", "temp_anom", "heat_days",
              "flood_event", "drought_event"]].mean()
cols = st.columns(len(sig))
for c, (k, v) in zip(cols, sig.items()):
    c.metric(lab(k), f"{v:.2f}")
