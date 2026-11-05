"""Shared Streamlit helpers: cached data/model loaders, sidebar controls, small widgets."""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from . import pipeline
from .config import ALERT_COLOURS, APP_VERSION, OUTCOMES, SITES
from .data import build_warehouse, fill_optional
from .features import climate_features
from .models import DEFAULT_MODEL, MODELS


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Building synthetic data warehouse...")
def _synthetic():
    clim, wh = build_warehouse()
    return clim, wh


@st.cache_data(show_spinner="Engineering features...")
def _features(data_key: str, _wh: pd.DataFrame, _clim):
    return climate_features(_wh, _clim)


def get_data():
    """Return (warehouse, features, data_key, source_label)."""
    up = st.session_state.get("uploaded_wh")
    if up is not None:
        wh = fill_optional(up)
        key = "upload-" + hashlib.md5(pd.util.hash_pandas_object(wh).values.tobytes()).hexdigest()
        return wh, _features(key, wh, None), key, "Uploaded file"
    clim, wh = _synthetic()
    return wh, _features("synthetic", wh, clim), "synthetic", "Synthetic (simulated)"


# ------------------------------------------------------------------ models (cached across users)
@st.cache_resource(show_spinner="Training models and forecasting...", max_entries=40)
def get_forecast(data_key: str, outcome: str, model_name: str, _feat):
    return pipeline.forecast(_feat, outcome, model_name)


@st.cache_resource(show_spinner="Scoring recent months out-of-sample...", max_entries=40)
def get_backcast(data_key: str, outcome: str, model_name: str, _feat):
    return pipeline.backcast(_feat, outcome, model_name)


@st.cache_resource(show_spinner="Running temporal validation (last 24 months withheld)...", max_entries=80)
def get_temporal(data_key: str, outcome: str, model_name: str, h: int, _feat):
    return pipeline.temporal_validation(_feat, outcome, model_name, h)


@st.cache_resource(show_spinner="Running spatial cross-validation (one site held out at a time)...",
                   max_entries=40)
def get_spatial(data_key: str, outcome: str, model_name: str, h: int, _feat):
    return pipeline.spatial_validation(_feat, outcome, model_name, h)


# ------------------------------------------------------------------ sidebar
def sidebar_controls(show_site=True, show_outcome=True, show_model=True):
    ss = st.session_state
    ss.setdefault("site", list(SITES)[0])
    ss.setdefault("outcome", "malaria")
    ss.setdefault("model", DEFAULT_MODEL)
    with st.sidebar:
        if show_site:
            sites = list(SITES)
            ss.site = st.selectbox("County / district", sites, index=sites.index(ss.site),
                                   format_func=lambda s: f"{s} ({SITES[s]['country']})")
        if show_outcome:
            outs = list(OUTCOMES)
            ss.outcome = st.selectbox("Health outcome", outs, index=outs.index(ss.outcome),
                                      format_func=lambda k: OUTCOMES[k]["label"])
        if show_model:
            names = list(MODELS)
            ss.model = st.selectbox("Model", names, index=names.index(ss.model),
                                    help="Compare models on the Model validation page.")
    return ss.site, ss.outcome, ss.model


def sidebar_footer(source_label: str):
    with st.sidebar:
        st.divider()
        st.caption(f"Data: **{source_label}**  \nVersion {APP_VERSION}")


def prototype_banner():
    st.warning("**Prototype for feedback.** All figures use simulated data and placeholder "
               "thresholds. They are not real forecasts for any county or district.", icon="⚠️")


def alert_badge(level: str, text: str = "") -> str:
    c = ALERT_COLOURS[level]
    return (f"<span style='background:{c};color:white;padding:2px 10px;border-radius:12px;"
            f"font-weight:600;font-size:0.85rem'>{level}</span> {text}")


def alert_card(level: str, title: str, score: int, mean: float, lo: float, hi: float, thr: float):
    c = ALERT_COLOURS[level]
    st.markdown(
        f"""<div style="border-left:6px solid {c};padding:10px 14px;border-radius:6px;
        background:rgba(128,128,128,0.08);margin-bottom:8px">
        <div style="font-size:0.85rem;opacity:0.8">{title}</div>
        <div style="font-size:1.6rem;font-weight:700;color:{c}">{level} &middot; {score}/100</div>
        <div style="font-size:0.9rem">Expected <b>{mean:,.0f}</b> cases
        (80% range {lo:,.0f} to {hi:,.0f})<br>Surge threshold: {thr:,.0f}</div></div>""",
        unsafe_allow_html=True)
