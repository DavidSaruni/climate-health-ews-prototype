"""Training, forecasting and validation (proposal sections 5.7.4 and 5.7.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ALERT_THRESHOLDS, HORIZONS, SURGE_FACTOR, TEMPORAL_HOLDOUT_MONTHS
from .features import make_dataset
from .models import make_model


def alert_level(p: float) -> str:
    if p >= ALERT_THRESHOLDS["red"]:
        return "Red"
    if p >= ALERT_THRESHOLDS["amber"]:
        return "Amber"
    return "Green"


# --------------------------------------------------------------------------- forecast
def forecast(feat: pd.DataFrame, outcome: str, model_name: str, n_sims: int = 2000):
    """Fit on all history and forecast 1-3 months ahead from the latest month.

    Returns (area_fc, site_fc, models):
      area_fc: one row per area x horizon with mean, intervals, surge probability
      site_fc: same, aggregated to site by Monte Carlo summation of area draws
    """
    rng = np.random.default_rng(0)
    area_rows, site_rows, fitted = [], [], {}
    latest = feat["month"].max()
    for h in HORIZONS:
        ds = make_dataset(feat, outcome, h)
        train, live = ds[ds["y"].notna()], ds[(ds["y"].isna()) & (ds["month"] == latest)].copy()
        model = make_model(model_name).fit(train)
        fitted[h] = (model, ds)
        mu = model.predict_mean(live)
        live["mean"] = mu
        for lvl in (0.8, 0.95):
            lo, hi = model.interval(mu, lvl)
            live[f"lo{int(lvl*100)}"], live[f"hi{int(lvl*100)}"] = lo, hi
        live["p_surge"] = model.prob_exceed(mu, live["threshold"])
        live["horizon"] = h
        area_rows.append(live)

        draws = model.sample(mu, n_sims, rng)
        site_med = (train.groupby(["site", "target_month"])["y"].sum().reset_index()
                    .assign(cm=lambda d: d.target_month.dt.month))
        for site, idx in live.groupby("site").groups.items():
            cols = live.index.get_indexer(idx)
            tot = draws[:, cols].sum(axis=1)
            tm = live.loc[idx, "target_month"].iloc[0]
            hist = site_med[(site_med.site == site) & (site_med.cm == tm.month)]["y"]
            thr = np.ceil(SURGE_FACTOR * hist.median())
            site_rows.append({"site": site, "horizon": h, "target_month": tm,
                              "mean": float(mu[cols].sum()),
                              "lo80": np.quantile(tot, 0.1), "hi80": np.quantile(tot, 0.9),
                              "lo95": np.quantile(tot, 0.025), "hi95": np.quantile(tot, 0.975),
                              "threshold": thr, "p_surge": float((tot > thr).mean())})
    area_fc = pd.concat(area_rows, ignore_index=True)
    site_fc = pd.DataFrame(site_rows)
    for d in (area_fc, site_fc):
        d["risk_score"] = (100 * d["p_surge"]).round().astype(int)
        d["alert"] = d["p_surge"].apply(alert_level)
    return area_fc, site_fc, fitted


def backcast(feat: pd.DataFrame, outcome: str, model_name: str, h: int = 1, months: int = 24):
    """Rolling-origin style one-step fitted values for the recent past (for the trend chart).

    Uses a model trained before the window, so the line shows genuine out-of-sample skill.
    """
    ds = make_dataset(feat, outcome, h)
    known = ds[ds["y"].notna()]
    cutoff = known["target_month"].max() - pd.DateOffset(months=months)
    model = make_model(model_name).fit(known[known.target_month <= cutoff])
    recent = known[known.target_month > cutoff].copy()
    recent["fitted"] = model.predict_mean(recent)
    return recent


# --------------------------------------------------------------------------- validation
def _metrics(df: pd.DataFrame, model, n_boot: int = 200, seed: int = 0) -> dict:
    y, mu = df["y"].values, df["mean"].values
    surge, p = df["surge"].values, df["p_surge"].values
    lo, hi = model.interval(mu, 0.95)

    def stat(idx):
        out = {"RMSE": np.sqrt(np.mean((y[idx] - mu[idx]) ** 2)),
               "MAE": np.mean(np.abs(y[idx] - mu[idx])),
               "Brier": np.mean((p[idx] - surge[idx]) ** 2),
               "95% PI coverage": np.mean((y[idx] >= lo[idx]) & (y[idx] <= hi[idx]))}
        s = surge[idx]
        out["AUC"] = roc_auc_score(s, p[idx]) if 0 < s.sum() < len(s) else np.nan
        return out

    point = stat(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = pd.DataFrame([stat(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    rows = []
    for k, v in point.items():
        rows.append({"metric": k, "value": v, "ci_lo": boots[k].quantile(0.025),
                     "ci_hi": boots[k].quantile(0.975)})
    return pd.DataFrame(rows)


def _score(model, test):
    test = test.copy()
    test["mean"] = model.predict_mean(test)
    test["p_surge"] = model.prob_exceed(test["mean"], test["threshold"])
    test["alert"] = test["p_surge"].apply(alert_level)
    return test


def temporal_validation(feat, outcome, model_name, h):
    """Withhold the most recent 24 months; train on everything before."""
    last = feat["month"].max()
    cutoff = last - pd.DateOffset(months=TEMPORAL_HOLDOUT_MONTHS)
    ds = make_dataset(feat, outcome, h, baseline_end=cutoff)
    known = ds[ds["y"].notna()]
    train, test = known[known.target_month <= cutoff], known[known.target_month > cutoff]
    model = make_model(model_name).fit(train)
    scored = _score(model, test)
    return _metrics(scored, model), scored, model


def spatial_validation(feat, outcome, model_name, h):
    """Leave one study site out at a time."""
    ds = make_dataset(feat, outcome, h)
    known = ds[ds["y"].notna()]
    per_site, scored_all = [], []
    for site in known["site"].unique():
        train, test = known[known.site != site], known[known.site == site]
        model = make_model(model_name).fit(train)
        scored = _score(model, test)
        m = _metrics(scored, model, n_boot=100).assign(site=site)
        per_site.append(m)
        scored_all.append(scored)
    return pd.concat(per_site, ignore_index=True), pd.concat(scored_all, ignore_index=True)


def alert_performance(scored: pd.DataFrame) -> pd.DataFrame:
    """How often each alert level was followed by an observed surge."""
    t = scored.groupby("alert").agg(months=("surge", "size"), surges=("surge", "sum")).reindex(
        ["Green", "Amber", "Red"]).fillna(0)
    t["surge rate"] = t["surges"] / t["months"].replace(0, np.nan)
    total = scored["surge"].sum()
    t["share of all surges"] = t["surges"] / total if total else np.nan
    return t.reset_index()


def calibration_table(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    b = pd.cut(scored["p_surge"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return scored.groupby(b, observed=True).agg(predicted=("p_surge", "mean"),
                                                observed=("surge", "mean"),
                                                n=("surge", "size")).reset_index(drop=True)
