import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import (ACTIONS, ALERT_COLOURS, GROUP_RELATIVE_RISK, OUTCOME_ACTIONS, OUTCOMES,
                        SITES, VULNERABILITY_GROUPS)

site, outcome, model_name = ui.sidebar_controls()
wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)
meta = SITES[site]

st.title(f"{site} {meta['level']}, {meta['country']}")
st.caption(f"{meta['profile']}  ·  {OUTCOMES[outcome]['label']}  ·  {model_name}")
ui.prototype_banner()

area_fc, site_fc, _ = ui.get_forecast(key, outcome, model_name, feat)
sfc = site_fc[site_fc.site == site].sort_values("horizon")
afc = area_fc[area_fc.site == site]

# ---------------------------------------------------------------- 1 + 2: risk scores and alerts
st.subheader("1. Risk score and alert level, next three months")
cols = st.columns(3)
for col, (_, r) in zip(cols, sfc.iterrows()):
    with col:
        ui.alert_card(r.alert, f"{r.target_month:%B %Y} ({r.horizon} month{'s' if r.horizon > 1 else ''} ahead)",
                      r.risk_score, r["mean"], r.lo80, r.hi80, r.threshold)

# ---------------------------------------------------------------- trend chart
st.subheader("Trend: recent months and forecast")
hist = (wh[wh.site == site].groupby("month")[outcome].sum().reset_index())
hist = hist[hist.month > hist.month.max() - pd.DateOffset(months=36)]
bc = ui.get_backcast(key, outcome, model_name, feat)
bc = bc[bc.site == site].groupby("target_month")["fitted"].sum().reset_index()

fig = go.Figure()
x_fc = [hist.month.iloc[-1], *sfc.target_month]
anchor = hist[outcome].iloc[-1]
fig.add_trace(go.Scatter(x=x_fc + x_fc[::-1], y=[anchor, *sfc.hi95] + [anchor, *sfc.lo95][::-1],
                         mode="lines", fill="toself", fillcolor="rgba(70,130,180,0.15)", line=dict(width=0),
                         name="95% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x_fc + x_fc[::-1], y=[anchor, *sfc.hi80] + [anchor, *sfc.lo80][::-1],
                         mode="lines", fill="toself", fillcolor="rgba(70,130,180,0.3)", line=dict(width=0),
                         name="80% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist.month, y=hist[outcome], mode="lines+markers", name="Observed (HMIS)",
                         line=dict(color="#444"), marker=dict(size=4)))
fig.add_trace(go.Scatter(x=bc.target_month, y=bc.fitted, mode="lines", name="Model, out-of-sample",
                         line=dict(color="steelblue", dash="dot")))
fig.add_trace(go.Scatter(x=x_fc, y=[anchor, *sfc["mean"]], mode="lines+markers", name="Forecast",
                         line=dict(color="steelblue", width=3),
                         marker=dict(size=11, color=["#444", *[ALERT_COLOURS[a] for a in sfc.alert]])))
fig.add_trace(go.Scatter(x=sfc.target_month, y=sfc.threshold, mode="markers", name="Surge threshold",
                         marker=dict(symbol="line-ew-open", size=30, color="#d64545", line=dict(width=3, color="#d64545"))))
fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                  legend=dict(orientation="h", y=-0.15), yaxis_title="Cases per month")
st.plotly_chart(fig, width="stretch")
st.caption("Dotted line: what the model would have predicted one month ahead over the last 24 "
           "months, using a model trained only on earlier data.")

# ---------------------------------------------------------------- 3: equity
st.subheader("3. Equity indicators")
h_sel = st.radio("Month", sfc.horizon.tolist(), horizontal=True,
                 format_func=lambda h: f"{sfc.set_index('horizon').loc[h, 'target_month']:%B %Y}")
a = afc[afc.horizon == h_sel].copy()
a["rate"] = 1000 * a["mean"] / a["pop_target"]

e1, e2 = st.columns(2)
with e1:
    st.markdown("**Predicted pressure by sub-unit**")
    a = a.sort_values("risk_score")
    fig = go.Figure(go.Bar(x=a.risk_score, y=a.sub_unit, orientation="h",
                           marker_color=[ALERT_COLOURS[x] for x in a.alert],
                           text=[f"{r:.1f} per 1,000" for r in a.rate], textposition="auto",
                           hovertemplate="%{y}: risk %{x}/100<extra></extra>"))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(range=[0, 100], title="Risk score"))
    st.plotly_chart(fig, width="stretch")

with e2:
    st.markdown("**Predicted cases by vulnerability group**")
    rr = np.array(GROUP_RELATIVE_RISK[outcome])
    total = a["mean"].sum()
    groups = []
    for i, g in enumerate(VULNERABILITY_GROUPS):
        share = np.average(a[f"share_{i}"], weights=a["pop_target"])
        groups.append({"group": g, "share of population": share,
                       "expected cases": total * share * rr[i]})
    gdf = pd.DataFrame(groups)
    gdf["relative risk"] = rr
    gdf = gdf[gdf["relative risk"] > 0]
    fig = go.Figure(go.Bar(x=gdf["expected cases"], y=gdf.group, orientation="h", marker_color="slateblue",
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Expected cases")
    st.plotly_chart(fig, width="stretch")
    st.caption("Population share x placeholder relative risk. Groups overlap, so bars do not add up.")

st.markdown("**Sex-disaggregated demand projection**")
fem = np.average(a["opd_female_share"], weights=a["pop_target"]) if "opd_female_share" in a else 0.56
s1, s2, s3 = st.columns(3)
if outcome == "anc":
    s1.metric("ANC and maternal visits expected", f"{total:,.0f}")
    s2.metric("Women of reproductive age affected", "All")
    s3.metric("vs. surge threshold", f"{total - sfc.set_index('horizon').loc[h_sel, 'threshold']:+,.0f}")
else:
    s1.metric("Expected cases: women and girls", f"{total * fem:,.0f}")
    s2.metric("Expected cases: men and boys", f"{total * (1 - fem):,.0f}")
    anc_fc = ui.get_forecast(key, "anc", model_name, feat)[1]
    anc = anc_fc[(anc_fc.site == site) & (anc_fc.horizon == h_sel)].iloc[0]
    s3.metric("ANC and maternal visits expected", f"{anc['mean']:,.0f}",
              help=f"ANC alert: {anc.alert} ({anc.risk_score}/100)")

# ---------------------------------------------------------------- 4: actions
st.subheader("4. Recommended preparedness actions")
worst = sfc.loc[sfc.p_surge.idxmax()]
st.markdown(ui.alert_badge(worst.alert, f"highest alert in the next three months "
                                        f"({worst.target_month:%B %Y})"), unsafe_allow_html=True)
for act in ACTIONS[worst.alert]:
    st.markdown(f"- {act}")
if worst.alert != "Green":
    st.markdown(f"- **{OUTCOMES[outcome]['label']}:** {OUTCOME_ACTIONS[outcome]}")
top = afc[afc.horizon == worst.horizon].nlargest(2, "risk_score")
st.markdown(f"- Prioritise **{', '.join(top.sub_unit)}** (highest predicted risk).")
st.caption("Placeholder actions. Final actions will come from the qualitative analysis of "
           "effective response practices and will be agreed with the DHMT/CHMT.")

with st.expander("Forecast table (sub-units)"):
    show = afc[["sub_unit", "target_month", "mean", "lo80", "hi80", "threshold", "risk_score", "alert"]]
    st.dataframe(show.round({c: 0 for c in ["mean", "lo80", "hi80", "threshold"]}), hide_index=True, width="stretch")
    st.download_button("Download CSV", show.to_csv(index=False), f"forecast_{site}_{outcome}.csv")
