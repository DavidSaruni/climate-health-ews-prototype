import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import ALERT_COLOURS, APP_TITLE, OUTCOMES, SITE_COORDS, SITES

site, outcome, model_name = ui.sidebar_controls(show_site=False)
wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)

st.title(APP_TITLE)
st.caption("Objective 5: a climate-health predictive model and decision-support tool for "
           "district and county health teams in Kenya and Malawi.")
ui.prototype_banner()

area_fc, site_fc, _ = ui.get_forecast(key, outcome, model_name, feat)
latest = feat["month"].max()

st.subheader(f"{OUTCOMES[outcome]['label']}: outlook for the next three months")
st.caption(f"Forecasts made from data up to **{latest:%B %Y}** using **{model_name}**. "
           "Risk score = probability (0 to 100) that cases exceed the surge threshold.")

# ---- summary grid: site x horizon
grid = site_fc.pivot(index="site", columns="target_month", values="alert").reindex(list(SITES))
scores = site_fc.pivot(index="site", columns="target_month", values="risk_score").reindex(list(SITES))
cols = st.columns([3, 2])
with cols[0]:
    header = "".join(f"<th style='padding:6px 10px'>{m:%b %Y}</th>" for m in grid.columns)
    body = ""
    for s in grid.index:
        cells = "".join(
            f"<td style='padding:6px 10px;text-align:center'>"
            f"<span style='background:{ALERT_COLOURS[grid.loc[s, m]]};color:white;padding:3px 10px;"
            f"border-radius:10px;font-weight:600;white-space:nowrap'>{grid.loc[s, m]} {scores.loc[s, m]}</span></td>"
            for m in grid.columns)
        body += (f"<tr><td style='padding:6px 10px'><b>{s}</b><br><span style='font-size:0.8rem;"
                 f"opacity:0.7'>{SITES[s]['country']}: {SITES[s]['profile']}</span></td>{cells}</tr>")
    st.markdown(f"<table style='width:100%;border-collapse:collapse'><tr><th></th>{header}</tr>"
                f"{body}</table>", unsafe_allow_html=True)

with cols[1]:
    h1 = site_fc[site_fc.horizon == 1].set_index("site")
    peak = site_fc.loc[site_fc.groupby("site")["p_surge"].idxmax()].set_index("site")
    fig = go.Figure()
    for s, (lat, lon) in SITE_COORDS.items():
        lvl = peak.loc[s, "alert"]
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], mode="markers+text", text=[s], textposition="middle right",
            marker=dict(size=10 + peak.loc[s, "risk_score"] / 4, color=ALERT_COLOURS[lvl],
                        line=dict(width=1, color="white")),
            hovertext=f"{s}: highest alert {lvl} ({peak.loc[s, 'risk_score']}/100) "
                      f"in {peak.loc[s, 'target_month']:%b %Y}", hoverinfo="text", showlegend=False))
    fig.update_geos(scope="africa", showcountries=True, lataxis_range=[-19, 6],
                    lonaxis_range=[28, 44], showland=True, landcolor="rgba(200,200,200,0.25)",
                    bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Marker colour and size: highest alert in the next three months.")

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Sites on red alert (any month ahead)", int((peak["alert"] == "Red").sum()))
c2.metric("Sites on amber alert", int((peak["alert"] == "Amber").sum()))
c3.metric("Sub-units monitored", feat["area_id"].nunique())

st.info("**How to use this prototype:** pick a health outcome and a model in the sidebar. Open "
        "**District view** for one county's trend, alerts, equity indicators and recommended "
        "actions. Then use **Give feedback** to tell us what works and what is missing.", icon="ℹ️")
