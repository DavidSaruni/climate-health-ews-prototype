"""Geospatial Studio: pick any country, see satellite imagery, and colour its counties or
districts by flood, drought, heat or disease signals."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import climate as C
from src import geo, ui
from src.config import ALERT_THRESHOLDS, OUTCOMES, SITES

MIN_FLOOD_RAIN_MM = 50  # a "record" month must also be wet in absolute terms
NO_DATA = "rgba(0,0,0,0)"
GREY = "rgba(160,160,160,0.25)"

st.title("Geospatial Studio")
st.caption("Any country · satellite imagery · counties and districts coloured by climate hazard or disease signal. "
           "Climate: NASA POWER (2001-2020 baseline). Boundaries: geoBoundaries. Imagery: Esri and NASA GIBS.")


# ------------------------------------------------------------------ cached data access
@st.cache_data(ttl=86400, show_spinner=False)
def countries():
    return geo.list_countries()


@st.cache_data(ttl=86400, show_spinner=False)
def levels(iso, version=2):
    return geo.available_levels(iso)


@st.cache_data(ttl=86400, show_spinner="Loading boundaries...")
def bounds(iso, level, version=2):
    return geo.boundaries(iso, level)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def unit_climate(iso, level, points, version=2):
    monthly = C.fetch_many(list(points), C.power_monthly)
    daily = C.fetch_many(list(points), C.power_daily, start=date.today() - timedelta(days=45))
    out = []
    for (lat, _), m, d in zip(points, monthly, daily):
        mi = C.monthly_indices(m, lat) if m is not None and len(m) > 36 else None
        out.append((mi, d))
    return out


# ------------------------------------------------------------------ controls
cl = countries()
names = [c["name"] for c in cl]
iso_of = {c["name"]: c["iso"] for c in cl}
with st.sidebar:
    st.subheader("Studio controls")
    country = st.selectbox("Country", names, index=names.index("Kenya") if "Kenya" in names else 0)
    iso = iso_of[country]
    lv = levels(iso)
    if not lv:
        st.error("No boundaries available for this country.")
        st.stop()
    lv_names = {"ADM1": "Level 1 (e.g. counties, regions)", "ADM2": "Level 2 (e.g. sub-counties, districts)",
                "ADM3": "Level 3 (e.g. wards, TAs)"}
    keys = list(lv)
    level = st.selectbox("Administrative level", keys, index=keys.index(geo.default_level(lv)),
                         format_func=lambda k: f"{lv_names[k]}: {lv[k]} units")
    layer = st.radio("Colour units by", ["All hazards (most severe)", "flood", "drought", "heat", "malaria",
                                         "Model risk (project sites)", "Uploaded disease data"],
                     format_func=lambda k: C.HAZARDS[k]["label"] if k in C.HAZARDS else k)
    st.divider()
    basemap = st.selectbox("Base map", list(geo.BASEMAPS))
    overlay = st.selectbox("Satellite overlay", list(geo.OVERLAYS))
    img_day = st.date_input("Imagery date", date.today() - timedelta(days=1), max_value=date.today(),
                            help="Daily NASA imagery and flood maps. Cloud cover can hide the ground on some days.")
    fill_opacity = st.slider("Colour opacity", 0.0, 1.0, 0.55, 0.05)

gj = bounds(iso, level)
feats = gj["features"]
n_units = len(feats)
units = pd.DataFrame([f["properties"] for f in feats])

# ------------------------------------------------------------------ climate for every unit
state_key = f"studio_go_{iso}_{level}"
if n_units > 120 and not st.session_state.get(state_key):
    st.info(f"{country} has {n_units} units at this level. Loading climate data for all of them takes "
            f"about {max(1, n_units // 50)} to {max(2, n_units // 25)} minutes the first time (it is cached after).")
    if st.button(f"Load climate data for {n_units} units", type="primary"):
        st.session_state[state_key] = True
        st.rerun()
    clim = [(None, None)] * n_units
else:
    with st.spinner(f"Fetching NASA POWER climate for {n_units} units (first time only)..."):
        clim = unit_climate(iso, level, tuple((u.lat, u.lon) for u in units.itertuples()))

ok = [c[0] is not None for c in clim]
months = sorted({m for mi, _ in clim if mi is not None for m in mi.dropna(subset=["rain_mm"])["month"]}, reverse=True)
latest_daily = max((d["date"].max() for _, d in clim if d is not None and len(d)), default=None)
period_opts = ([f"Last 30 days (to {latest_daily:%d %b %Y})"] if latest_daily is not None else []) + \
              [f"{m:%B %Y}" for m in months[:36]]
with st.sidebar:
    period = st.selectbox("Period", period_opts) if period_opts else None

# ------------------------------------------------------------------ indicator table for the period
rows = []
for u, (mi, d) in zip(units.itertuples(), clim):
    r = {"uid": u.uid, "name": u.name, "lat": u.lat, "lon": u.lon}
    if mi is not None and period:
        if period.startswith("Last 30 days"):
            r.update(C.recent_window(d, mi) if d is not None else {})
        else:
            row = mi[mi["month"] == pd.to_datetime(period, format="%B %Y")]
            if len(row):
                r.update(row.iloc[0][["rain_mm", "rain_pct", "spei_3", "tmax", "tmax_anom", "tmax_z",
                                      "malaria_suit"]].to_dict())
    rows.append(r)
tab = pd.DataFrame(rows)
for c in ["rain_mm", "rain_pct", "spei_3", "tmax_anom", "tmax_z", "malaria_suit"]:
    if c not in tab:
        tab[c] = np.nan
for hz, spec in C.HAZARDS.items():
    tab[f"lvl_{hz}"] = [C.level(v, hz) for v in tab[spec["metric"]]]
tab.loc[tab["rain_mm"] < MIN_FLOOD_RAIN_MM, "lvl_flood"] = tab["lvl_flood"].clip(upper=0)

# model risk for project sites (matched by name)
tab["lvl_model"], tab["model_score"] = -1, np.nan
if layer == "Model risk (project sites)":
    wh, feat, key, source = ui.get_data()
    st.session_state.setdefault("outcome", "malaria")
    st.session_state.setdefault("model", ui.DEFAULT_MODEL)
    with st.sidebar:
        outs = list(OUTCOMES)
        st.session_state.outcome = st.selectbox("Outcome", outs, index=outs.index(st.session_state.outcome),
                                                format_func=lambda k: OUTCOMES[k]["label"])
    _, site_fc, _ = ui.get_forecast(key, st.session_state.outcome, st.session_state.model, feat)
    fc = site_fc[site_fc.horizon == 1].set_index("site")
    lookup = {geo.norm_name(s): s for s in SITES}
    for i, nm in enumerate(tab["name"]):
        s = lookup.get(geo.norm_name(nm))
        if s in fc.index:
            tab.loc[i, "model_score"] = fc.loc[s, "risk_score"]
            tab.loc[i, "lvl_model"] = {"Green": 0, "Amber": 2, "Red": 3}[fc.loc[s, "alert"]]

# uploaded disease data
tab["lvl_upload"], tab["upload_value"] = -1, np.nan
if layer == "Uploaded disease data":
    with st.sidebar:
        up = st.file_uploader("CSV: unit name + value (e.g. DHIS2 cases per 1,000)", type="csv")
    if up is not None:
        u = pd.read_csv(up)
        name_col = next((c for c in u.columns if u[c].dtype == object), u.columns[0])
        val_col = next((c for c in u.columns if c != name_col and pd.api.types.is_numeric_dtype(u[c])), None)
        if val_col:
            vals = dict(zip(u[name_col].map(geo.norm_name), u[val_col]))
            tab["upload_value"] = tab["name"].map(geo.norm_name).map(vals)
            q = tab["upload_value"].quantile([0.25, 0.5, 0.75]).values
            tab["lvl_upload"] = [(-1 if pd.isna(v) else int(sum(v >= x for x in q))) for v in tab["upload_value"]]
            st.sidebar.caption(f"Matched {tab['upload_value'].notna().sum()} of {len(u)} rows by name "
                               f"(column '{val_col}'). Levels = quartiles.")
    else:
        st.sidebar.caption("Columns: first text column = unit name, first numeric column = value.")

# ------------------------------------------------------------------ colour assignment
PURPLES = C.HAZARDS["malaria"]["colours"]
ALERTS = [GREY, "#2e9e5b", "#e8a317", "#d64545"]


def colour_and_label(r):
    if layer == "All hazards (most severe)":
        best = max(("flood", "drought", "heat"), key=lambda h: r[f"lvl_{h}"])
        lvl = r[f"lvl_{best}"]
        if lvl < 0:
            return NO_DATA, "No data"
        if lvl == 0:
            return GREY, "Normal"
        return C.HAZARDS[best]["colours"][lvl], f"{C.LEVELS[lvl]}: {best}"
    if layer in C.HAZARDS:
        lvl = r[f"lvl_{layer}"]
        return (NO_DATA, "No data") if lvl < 0 else (C.HAZARDS[layer]["colours"][lvl], C.LEVELS[lvl])
    if layer == "Model risk (project sites)":
        lvl = r["lvl_model"]
        return (NO_DATA, "Not a project site") if lvl < 0 else (
            ALERTS[{0: 1, 2: 2, 3: 3}[lvl]], {0: "Green", 2: "Amber", 3: "Red"}[lvl])
    lvl = r["lvl_upload"]
    return (NO_DATA, "No data") if lvl < 0 else (PURPLES[lvl], ["Lowest quarter", "2nd quarter", "3rd quarter",
                                                                  "Highest quarter"][lvl])


tab[["colour", "status"]] = tab.apply(lambda r: pd.Series(colour_and_label(r)), axis=1)


# ------------------------------------------------------------------ map
def fmt(v, f):
    return "n/a" if v is None or pd.isna(v) else f.format(v)


tab["hover"] = [
    f"<b>{r['name']}</b><br>{r['status']}<br>Rainfall {fmt(r['rain_mm'], '{:,.0f}')} mm "
    f"(percentile {fmt(r['rain_pct'], '{:.0%}')})<br>SPEI-3 {fmt(r['spei_3'], '{:+.2f}')}<br>"
    f"Max temp anomaly {fmt(r['tmax_anom'], '{:+.1f}')} °C<br>Malaria suitability {fmt(r['malaria_suit'], '{:.2f}')}"
    + (f"<br>Model risk {r['model_score']:.0f}/100" if pd.notna(r["model_score"]) else "")
    + (f"<br>Uploaded value {r['upload_value']:,.1f}" if pd.notna(r["upload_value"]) else "")
    for _, r in tab.iterrows()]

palette = list(dict.fromkeys(tab["colour"]))
tab["z"] = tab["colour"].map({c: i for i, c in enumerate(palette)})
n = max(len(palette), 1)
scale = []
for i, c in enumerate(palette):
    scale += [[i / n, c], [(i + 1) / n, c]]
if len(scale) < 2:
    scale = [[0, GREY], [1, GREY]]

fig = go.Figure(go.Choroplethmap(
    geojson=gj, featureidkey="properties.uid", locations=tab["uid"], z=tab["z"], zmin=-0.5, zmax=n - 0.5,
    colorscale=scale, showscale=False, marker_opacity=fill_opacity,
    marker_line_width=1, marker_line_color="rgba(255,255,255,0.85)",
    text=tab["hover"], hovertemplate="%{text}<extra></extra>"))

layers = []
bm = geo.BASEMAPS[basemap]
style = bm.get("style", "white-bg")
if "url" in bm:
    layers.append(dict(sourcetype="raster", source=[bm["url"]], below="traces", sourceattribution=bm["attribution"]))
elif "layer" in bm:
    layers.append(dict(sourcetype="raster", source=[geo.gibs_url(bm, img_day.isoformat())], below="traces",
                       sourceattribution=bm["attribution"], maxzoom=bm["maxzoom"]))
ov = geo.OVERLAYS[overlay]
if ov:
    layers.append(dict(sourcetype="raster", source=[geo.gibs_url(ov, img_day.isoformat())], below="traces",
                       opacity=0.85, sourceattribution="NASA EOSDIS GIBS"))
b = geo.bbox(gj)
fig.update_layout(map=dict(style=style, layers=layers, center=dict(lat=(b[1] + b[3]) / 2, lon=(b[0] + b[2]) / 2),
                           zoom=geo.zoom_for(b)),
                  height=620, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")

OVERLAY_NOTES = {
    "flood": "NASA flood map: red = flood water, blue = normal water (lakes, rivers), grey = cloud or no data.",
    "IMERG": "Rainfall rate at the imagery date: light blue = light rain through to red = very heavy rain.",
    "Land surface": "Daytime land surface temperature: blue = cool through to red = hot. Gaps = cloud.",
    "NDVI": "Vegetation greenness: brown = bare or dry, green = dense vegetation. Low values in a normally green "
            "season suggest drought stress.",
    "soil": "Root-zone soil moisture: brown = dry, blue = wet.",
}
left, right = st.columns([3, 1])
with left:
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    note = next((v for k, v in OVERLAY_NOTES.items() if k.lower() in overlay.lower()), None)
    if note:
        st.caption(f"Overlay ({img_day:%d %b %Y}): {note}")
with right:
    st.markdown(f"**{country}** · {lv_names[level].split(' (')[0]} · {n_units} units")
    st.caption(f"Boundaries: {gj.get('source', 'geoBoundaries')}")
    if period:
        st.markdown(f"Period: **{period}**")
    if not all(ok):
        st.warning(f"No climate data for {len(ok) - sum(ok)} units.")
    # legend
    rank = lambda st_: next((i for i, w in enumerate(["Severe", "Warning", "Watch", "Red", "Amber", "Highest",
                                                       "3rd", "2nd", "Green", "Lowest", "Normal"]) if st_.startswith(w)), 99)
    legend = tab.drop_duplicates("status").assign(r=lambda d: d.status.map(rank)).sort_values(["r", "status"])
    for _, r in legend.iterrows():
        c = r["colour"] if r["colour"] != NO_DATA else "transparent"
        st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{c};"
                    f"border:1px solid #999;margin-right:6px;vertical-align:middle'></span>{r['status']} "
                    f"({(tab.status == r['status']).sum()})", unsafe_allow_html=True)
    st.divider()
    for hz in ("flood", "drought", "heat", "malaria"):
        k = int((tab[f"lvl_{hz}"] >= 2).sum())
        st.metric(C.HAZARDS[hz]["label"].split(" (")[0].replace("Disease: ", ""), f"{k} at warning+")

with st.expander("How to read this map"):
    st.markdown(f"""
- **Flood (heavy rainfall):** percentile of rainfall against the same month in 2001-2020. Watch ≥ 75th,
  warning ≥ 90th, severe ≥ 97th, and at least {MIN_FLOOD_RAIN_MM} mm. This is a rainfall signal, not observed
  flooding: switch on the *Observed flooding* satellite overlay to see water on the ground.
- **Drought (SPEI-3):** 3-month rainfall minus evaporative demand (Hargreaves), standardised. Watch ≤ −0.8,
  warning ≤ −1.3, severe ≤ −1.6.
- **Heat:** standardised maximum-temperature anomaly. Watch ≥ +1, warning ≥ +1.5, severe ≥ +2.
- **Malaria suitability:** temperature and rainfall suitability for transmission (Craig et al. 1999). Climate only,
  not case data. For case data use *Uploaded disease data* or *Model risk*.
- **Model risk:** the early-warning model's one-month-ahead alert for the six project sites (simulated data in the prototype).
- Climate comes from NASA POWER at each unit's centroid (about 50 km resolution), so small units share values.
""")

# ------------------------------------------------------------------ ranking and detail
st.subheader("Units ranked by severity")
tab["max_level"] = tab[["lvl_flood", "lvl_drought", "lvl_heat"]].max(axis=1)
show = tab.sort_values(["max_level", "spei_3"], ascending=[False, True])[
    ["name", "status", "rain_mm", "rain_pct", "spei_3", "tmax_anom", "malaria_suit"]].rename(columns={
        "name": "Unit", "status": "Status", "rain_mm": "Rainfall (mm)", "rain_pct": "Rain percentile",
        "spei_3": "SPEI-3", "tmax_anom": "Max temp anomaly (°C)", "malaria_suit": "Malaria suitability"})
st.dataframe(show.round(2), hide_index=True, width="stretch", height=300)
st.download_button("Download table (CSV)", show.to_csv(index=False), f"studio_{iso}_{level}.csv")

st.subheader("Unit detail")
order = tab.sort_values("max_level", ascending=False)["name"].tolist()
pick = st.selectbox("County / district", order)
i = tab.index[tab["name"] == pick][0]
mi = clim[i][0]
if mi is None:
    st.info("No climate data for this unit.")
else:
    recent = mi[mi["month"] >= mi["month"].max() - pd.DateOffset(months=35)]
    clim_mean = mi[(mi.month.dt.year >= C.BASELINE[0]) & (mi.month.dt.year <= C.BASELINE[1])] \
        .groupby("cal_month")["rain_mm"].mean()
    c1, c2 = st.columns(2)
    with c1:
        f1 = go.Figure()
        f1.add_bar(x=recent["month"], y=recent["rain_mm"], name="Rainfall", marker_color="#3182bd")
        f1.add_scatter(x=recent["month"], y=recent["cal_month"].map(clim_mean), name="2001-2020 average",
                       line=dict(color="#444", dash="dot"))
        f1.update_layout(title="Monthly rainfall (mm)", height=300, margin=dict(l=0, r=0, t=40, b=0),
                         legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(f1, width="stretch")
    with c2:
        f2 = go.Figure()
        f2.add_scatter(x=recent["month"], y=recent["spei_3"], name="SPEI-3", line=dict(color="#d7301f"))
        f2.add_scatter(x=recent["month"], y=recent["tmax_z"], name="Max temp anomaly (z)", line=dict(color="#99000d", dash="dot"))
        f2.add_hline(y=-1.3, line_dash="dash", line_color="#7f2704", annotation_text="drought")
        f2.add_hline(y=1.5, line_dash="dash", line_color="#99000d", annotation_text="heat")
        f2.update_layout(title="Drought and heat indices", height=300, margin=dict(l=0, r=0, t=40, b=0),
                         legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(f2, width="stretch")
