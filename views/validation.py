import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import pipeline, ui
from src.config import ALERT_COLOURS, OUTCOMES, TEMPORAL_HOLDOUT_MONTHS
from src.models import MODELS

site, outcome, model_name = ui.sidebar_controls(show_site=False)
wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)

st.title("Model validation")
st.caption(f"{OUTCOMES[outcome]['label']}. Three-stage validation from proposal section 5.7.5.")
ui.prototype_banner()
h = st.radio("Forecast horizon", [1, 2, 3], horizontal=True, format_func=lambda x: f"{x} month(s) ahead")

tab1, tab2, tab3 = st.tabs(["1. Temporal (last 24 months withheld)", "2. Spatial (leave one site out)",
                            "3. Stakeholder check"])

GLOSSARY = {"RMSE": "Typical prediction error, large misses weighted more (lower is better)",
            "MAE": "Average prediction error in cases (lower is better)",
            "AUC": "How well risk scores separate surge from non-surge months (0.5 = chance, 1 = perfect)",
            "Brier": "Accuracy of surge probabilities (lower is better)",
            "95% PI coverage": "Share of observed values inside the 95% interval (target 0.95)"}

with tab1:
    st.markdown(f"Models are trained on data up to {TEMPORAL_HOLDOUT_MONTHS} months before the "
                "latest month, then tested on those unseen 24 months.")
    rows = []
    for name in MODELS:
        met, _, _ = ui.get_temporal(key, outcome, name, h, feat)
        for _, r in met.iterrows():
            rows.append({"model": name, "metric": r.metric,
                         "value": f"{r.value:,.3f}" if r.value < 10 else f"{r.value:,.0f}",
                         "95% CI": f"{r.ci_lo:,.2f} to {r.ci_hi:,.2f}" if r.value < 10
                         else f"{r.ci_lo:,.0f} to {r.ci_hi:,.0f}"})
    comp = pd.DataFrame(rows)
    table = comp.pivot(index="model", columns="metric", values="value")
    cis = comp.pivot(index="model", columns="metric", values="95% CI")
    st.markdown("**Model comparison**")
    st.dataframe(table[list(GLOSSARY)], width="stretch")
    with st.expander("Confidence intervals (bootstrap, 200 resamples)"):
        st.dataframe(cis[list(GLOSSARY)], width="stretch")
    with st.expander("What the metrics mean"):
        for k, v in GLOSSARY.items():
            st.markdown(f"- **{k}**: {v}")

    st.markdown(f"**Detail for {model_name}**")
    met, scored, model = ui.get_temporal(key, outcome, model_name, h, feat)
    c1, c2 = st.columns(2)
    with c1:
        cal = pipeline.calibration_table(scored)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                                 name="Perfect"))
        fig.add_trace(go.Scatter(x=cal.predicted, y=cal.observed, mode="lines+markers", name=model_name,
                                 marker=dict(size=6 + cal.n / cal.n.max() * 12)))
        fig.update_layout(title="Calibration: predicted vs observed surge rate", height=360,
                          xaxis_title="Predicted probability", yaxis_title="Observed share",
                          margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        ap = pipeline.alert_performance(scored)
        fig = go.Figure(go.Bar(x=ap.alert, y=ap["surge rate"], marker_color=[ALERT_COLOURS[a] for a in ap.alert],
                               text=[f"{m:.0f} area-months" for m in ap.months], textposition="outside"))
        fig.update_layout(title="How often each alert was followed by a surge", height=360,
                          yaxis=dict(range=[0, 1.1], title="Surge rate"), margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    st.caption("Good alerts: red should usually be followed by a surge, green rarely. These "
               "results are used to set the amber and red cut-offs with district teams.")

    agg = scored.groupby(["site", "target_month"])[["y", "mean"]].sum().reset_index()
    fig = go.Figure()
    for s, d in agg.groupby("site"):
        fig.add_trace(go.Scatter(x=d.target_month, y=d.y, name=f"{s} observed", mode="lines",
                                 legendgroup=s))
        fig.add_trace(go.Scatter(x=d.target_month, y=d["mean"], name=f"{s} predicted", mode="lines",
                                 line=dict(dash="dot"), legendgroup=s, showlegend=False))
    fig.update_layout(title="Observed (solid) vs predicted (dotted), held-out period", height=380,
                      margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.markdown("Each site is held out in turn; the model is trained on the other five. This tests "
                "whether the model generalises to places it has not seen.")
    if st.button(f"Run spatial cross-validation for {model_name}", type="primary"):
        st.session_state[f"spatial_{outcome}_{model_name}_{h}"] = True
    if st.session_state.get(f"spatial_{outcome}_{model_name}_{h}"):
        per_site, _ = ui.get_spatial(key, outcome, model_name, h, feat)
        t = per_site.pivot(index="site", columns="metric", values="value")[list(GLOSSARY)]
        st.dataframe(t.style.format("{:,.3f}"), width="stretch")
        st.caption("Sites where AUC drops or coverage falls well below 0.95 need site-specific "
                   "terms or more local data.")

with tab3:
    st.markdown("Do the model's alerts match what DHMT and Ministry of Health teams experienced "
                "during recent climate events? Below are held-out months with a declared flood "
                "or drought.")
    _, scored, _ = ui.get_temporal(key, outcome, model_name, h, feat)
    events = feat[["area_id", "month", "flood_event", "drought_event"]].rename(columns={"month": "target_month"})
    ev = scored.drop(columns=["flood_event", "drought_event"]).merge(events, on=["area_id", "target_month"])
    ev = ev[(ev.flood_event == 1) | (ev.drought_event == 1)]
    ev = (ev.groupby(["site", "target_month"])
            .agg(flood=("flood_event", "max"), drought=("drought_event", "max"),
                 predicted=("mean", "sum"), observed=("y", "sum"), risk=("p_surge", "mean"))
            .reset_index().sort_values("target_month", ascending=False))
    ev["alert"] = ev.risk.apply(pipeline.alert_level)
    ev["event"] = ev.apply(lambda r: " + ".join([n for n, f in (("Flood", r.flood), ("Drought", r.drought)) if f]), axis=1)
    st.dataframe(ev[["site", "target_month", "event", "alert", "predicted", "observed"]].round({"predicted": 0, "observed": 0}),
                 hide_index=True, width="stretch")
    st.info("During co-design workshops, district teams review each event and record whether "
            "the alert matched their experience on the **Give feedback** page.", icon="🤝")
