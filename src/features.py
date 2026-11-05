"""Feature engineering (proposal section 5.7.3).

For a forecast made at origin month t for target month t+h, every predictor uses only
information available at t, except the seasonal rainfall outlook, which stands in for
the KMD / DCMS seasonal forecast that would be available at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUTCOMES, SURGE_FACTOR

FEATURE_LABELS = {
    "rain_anom_30": "Rainfall anomaly, 30-day",
    "rain_anom_90": "Rainfall anomaly, 90-day",
    "spei_1": "SPEI, 1-month",
    "spei_3": "SPEI, 3-month",
    "temp_anom": "Temperature anomaly",
    "heat_days": "Heat-threshold exceedance days",
    "flood_event": "Flood declared (this month)",
    "drought_event": "Drought declared (this month)",
    "flood_last3": "Flood in last 3 months",
    "drought_last3": "Drought in last 3 months",
    "outlook_rain_anom": "Seasonal rainfall outlook (target month)",
    "rate_lag0": "Cases per 1,000, latest month",
    "rate_lag1": "Cases per 1,000, one month earlier",
    "rate_lag2": "Cases per 1,000, two months earlier",
    "rate_same_month_ly": "Cases per 1,000, same month last year",
    "opd_rate": "Outpatient visits per 1,000",
    "stockout_rate": "Medicine stock-out rate",
    "referral_functional": "Referral pathway functionality",
    "readiness_score": "Facility readiness score",
    "surge_capacity": "Facility surge capacity",
    "vulnerability_index": "Household vulnerability index",
    "month_sin": "Season (sine)",
    "month_cos": "Season (cosine)",
}
FEATURES = list(FEATURE_LABELS)
CLIMATE_FEATURES = FEATURES[:11]


def _baseline(clim: pd.DataFrame) -> pd.DataFrame:
    """Site x calendar-month climatology from the (up to) 20-year climate record."""
    c = clim.sort_values(["site", "month"]).copy()
    c["rain3"] = c.groupby("site")["rainfall_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    c["wb1"] = c["rainfall_mm"] - _pet(c["temp_c"])
    c["wb3"] = c.groupby("site")["wb1"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    c["cal_month"] = c["month"].dt.month
    g = c.groupby(["site", "cal_month"])
    out = g.agg(rain_mu=("rainfall_mm", "mean"), rain_sd=("rainfall_mm", "std"),
                rain3_mu=("rain3", "mean"), rain3_sd=("rain3", "std"),
                wb1_mu=("wb1", "mean"), wb1_sd=("wb1", "std"),
                wb3_mu=("wb3", "mean"), wb3_sd=("wb3", "std"),
                temp_mu=("temp_c", "mean"))
    return out.reset_index()


def _pet(temp_c: pd.Series) -> pd.Series:
    # Crude temperature-based potential evapotranspiration (mm/month). The full SPEI
    # (log-logistic fit, Hargreaves PET) replaces this once station data are in hand.
    return 4.5 * temp_c.clip(lower=0)


def climate_features(wh: pd.DataFrame, clim: pd.DataFrame | None) -> pd.DataFrame:
    """Add area-month climate features to the warehouse."""
    if clim is None or clim.empty:  # uploaded data: build baseline from the warehouse itself
        clim = wh.groupby(["site", "month"], as_index=False)[["rainfall_mm", "temp_c"]].mean()
    base = _baseline(clim)
    df = wh.sort_values(["area_id", "month"]).copy()
    df["cal_month"] = df["month"].dt.month
    df = df.merge(base, on=["site", "cal_month"], how="left")
    g = df.groupby("area_id")
    rain3 = g["rainfall_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    wb1 = df["rainfall_mm"] - _pet(df["temp_c"])
    wb3 = wb1.groupby(df["area_id"]).transform(lambda s: s.rolling(3, min_periods=1).sum())
    sd = lambda s: s.replace(0, np.nan).fillna(1)
    df["rain_anom_30"] = (df["rainfall_mm"] - df["rain_mu"]) / sd(df["rain_sd"])
    df["rain_anom_90"] = (rain3 - df["rain3_mu"]) / sd(df["rain3_sd"])
    df["spei_1"] = (wb1 - df["wb1_mu"]) / sd(df["wb1_sd"])
    df["spei_3"] = (wb3 - df["wb3_mu"]) / sd(df["wb3_sd"])
    df["temp_anom"] = df["temp_c"] - df["temp_mu"]
    df["flood_last3"] = g["flood_event"].transform(lambda s: s.rolling(3, min_periods=1).max())
    df["drought_last3"] = g["drought_event"].transform(lambda s: s.rolling(3, min_periods=1).max())
    df["opd_rate"] = 1000 * df["opd_total"] / df["population"]
    return df.drop(columns=[c for c in base.columns if c not in ("site", "cal_month")])


def make_dataset(feat: pd.DataFrame, outcome: str, h: int, *, baseline_end=None,
                 outlook_skill: float = 0.7, seed: int = 7) -> pd.DataFrame:
    """Design matrix for one outcome at horizon h.

    Rows with a missing target are the live forecast rows (origin = latest month).
    `baseline_end` limits the data used for surge thresholds, so validation does not
    peek at the test period.
    """
    df = feat.sort_values(["area_id", "month"]).copy()
    g = df.groupby("area_id")
    rate = 1000 * df[outcome] / df["population"]
    df["rate_lag0"] = rate
    df["rate_lag1"] = rate.groupby(df["area_id"]).shift(1)
    df["rate_lag2"] = rate.groupby(df["area_id"]).shift(2)
    df["rate_same_month_ly"] = rate.groupby(df["area_id"]).shift(12 - h)

    df["target_month"] = df["month"] + pd.DateOffset(months=h)
    df["y"] = g[outcome].shift(-h)
    df["pop_target"] = g["population"].shift(-h).fillna(df["population"])
    tm = df["target_month"].dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * tm / 12), np.cos(2 * np.pi * tm / 12)

    # Simulated seasonal outlook: the true target-month anomaly plus forecast error.
    rng = np.random.default_rng(seed + h)
    future = g["rain_anom_30"].shift(-h)
    noise = pd.Series(rng.normal(0, np.sqrt(1 / outlook_skill**2 - 1), len(df)), index=df.index)
    df["outlook_rain_anom"] = (future + noise).fillna(noise)

    # Surge threshold: area's median count for the target calendar month x SURGE_FACTOR.
    hist = df if baseline_end is None else df[df["month"] <= baseline_end]
    med = hist.groupby(["area_id", "cal_month"])[outcome].median().rename("median_cases").reset_index()
    df["_tm"] = df["target_month"].dt.month
    df = df.merge(med.rename(columns={"cal_month": "_tm"}), on=["area_id", "_tm"],
                  how="left").drop(columns="_tm")
    df["threshold"] = np.ceil(SURGE_FACTOR * df["median_cases"])
    df["surge"] = (df["y"] > df["threshold"]).astype(float).where(df["y"].notna())
    df = df.dropna(subset=["rate_lag2", "rate_same_month_ly"])
    for c in FEATURES:
        df[c] = df[c].astype(float).fillna(df[c].median())
    return df.reset_index(drop=True)


def all_outcomes():
    return list(OUTCOMES)
