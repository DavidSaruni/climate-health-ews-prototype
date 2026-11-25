"""NASA POWER climate client and hazard indices.

NASA POWER (https://power.larc.nasa.gov) gives free daily and monthly weather for any point,
1981 to a few days ago, no key needed. Used for:
  - the Geospatial Studio (any country, any county/district)
  - the climate stream of the real-data warehouse until KMD / DCCMS station data arrive

Indices (per calendar month, against a 2001-2020 baseline):
  rain_pct   percentile of monthly rainfall           -> flood signal (heavy-rain anomaly)
  spei_3     standardised 3-month water balance       -> drought (z-score approximation of SPEI)
  tmax_z     standardised max-temperature anomaly     -> heat
  malaria_suit  climatic suitability for malaria     -> disease (Craig et al. 1999 fuzzy rules)
"""
from __future__ import annotations

import calendar
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

POWER = "https://power.larc.nasa.gov/api/temporal"
PARAMS = "PRECTOTCORR,T2M_MAX,T2M_MIN,T2M"
BASELINE = (2001, 2020)
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
HEADERS = {"User-Agent": "climate-health-ews-prototype/0.2 (research)"}


# ------------------------------------------------------------------ download (with disk cache)
def _get(kind: str, params: dict, ttl_hours: float) -> dict:
    key = hashlib.md5(json.dumps([kind, params], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"power_{key}.json"
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return json.loads(path.read_text())
    for attempt in range(3):
        try:
            r = requests.get(f"{POWER}/{kind}/point", params=params, headers=HEADERS, timeout=90)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _frame(data: dict) -> pd.DataFrame:
    p = data["properties"]["parameter"]
    df = pd.DataFrame(p).replace(-999.0, np.nan)
    return df


def power_monthly(lat: float, lon: float, start: int = BASELINE[0], end: int | None = None) -> pd.DataFrame:
    """Monthly rainfall (mm) and temperatures (deg C). Latest months may be missing (POWER lag)."""
    end = end or date.today().year
    data = _get("monthly", dict(parameters=PARAMS, community="AG", longitude=round(lon, 3),
                                latitude=round(lat, 3), start=start, end=end, format="JSON"), ttl_hours=72)
    df = _frame(data)
    df = df[~df.index.str.endswith("13")]  # drop annual totals
    df.index = pd.to_datetime(df.index, format="%Y%m")
    days = df.index.days_in_month
    out = pd.DataFrame({"month": df.index, "rain_mm": df["PRECTOTCORR"].values * days,
                        "tmax": df["T2M_MAX"].values, "tmin": df["T2M_MIN"].values,
                        "tmean": df["T2M"].values}).dropna(subset=["rain_mm", "tmax"])
    return out.reset_index(drop=True)


def power_daily(lat: float, lon: float, start: date, end: date | None = None) -> pd.DataFrame:
    end = end or date.today()
    data = _get("daily", dict(parameters=PARAMS, community="AG", longitude=round(lon, 3), latitude=round(lat, 3),
                              start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"), format="JSON"),
                ttl_hours=12)
    df = _frame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    out = pd.DataFrame({"date": df.index, "rain_mm": df["PRECTOTCORR"].values, "tmax": df["T2M_MAX"].values,
                        "tmin": df["T2M_MIN"].values, "tmean": df["T2M"].values})
    return out.dropna(subset=["rain_mm", "tmax"]).reset_index(drop=True)


def fetch_many(points: list[tuple[float, float]], fn, workers: int = 8, **kw) -> list:
    """Run fn(lat, lon) for many points in parallel; failed points return None."""
    def one(pt):
        try:
            return fn(pt[0], pt[1], **kw)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, points))


# ------------------------------------------------------------------ indices
def extraterrestrial_radiation(lat: float, month_index: pd.DatetimeIndex) -> np.ndarray:
    """Ra (MJ/m2/day) at mid-month (FAO-56 eq. 21)."""
    doy = (month_index + pd.Timedelta(days=14)).day_of_year.values
    phi = math.radians(lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def malaria_suitability(tmean: np.ndarray, rain_mm: np.ndarray) -> np.ndarray:
    """Fuzzy climatic suitability for P. falciparum transmission (simplified Craig et al. 1999).
    Temperature: 0 at <=18 C, rising to 1 at 22 C, 1 to 32 C, falling to 0 at 40 C.
    Rainfall: 0 at 0 mm/month, rising to 1 at 80 mm/month. Suitability = min of the two."""
    t = np.interp(tmean, [18, 22, 32, 40], [0, 1, 1, 0])
    r = np.clip(rain_mm / 80.0, 0, 1)
    return np.minimum(t, r)


def monthly_indices(m: pd.DataFrame, lat: float, baseline=BASELINE) -> pd.DataFrame:
    """Add PET, water balance, SPEI-1/3, rainfall percentile and heat anomaly to a monthly frame."""
    df = m.sort_values("month").reset_index(drop=True).copy()
    idx = pd.DatetimeIndex(df["month"])
    ra = extraterrestrial_radiation(lat, idx)
    trange = np.sqrt(np.clip(df["tmax"] - df["tmin"], 0, None))
    df["pet_mm"] = 0.0023 * (df["tmean"] + 17.8) * trange * ra * 0.408 * idx.days_in_month  # Hargreaves
    df["wb1"] = df["rain_mm"] - df["pet_mm"]
    df["wb3"] = df["wb1"].rolling(3, min_periods=3).sum()
    df["rain3"] = df["rain_mm"].rolling(3, min_periods=3).sum()
    df["cal_month"] = idx.month
    base = df[(idx.year >= baseline[0]) & (idx.year <= baseline[1])]
    g = base.groupby("cal_month")
    stats = pd.DataFrame({
        "wb1_mu": g["wb1"].mean(), "wb1_sd": g["wb1"].std(), "wb3_mu": g["wb3"].mean(), "wb3_sd": g["wb3"].std(),
        "rain_mu": g["rain_mm"].mean(), "rain_sd": g["rain_mm"].std(),
        "tmax_mu": g["tmax"].mean(), "tmax_sd": g["tmax"].std()}).reset_index()
    df = df.merge(stats, on="cal_month", how="left")
    sd = lambda s: s.replace(0, np.nan)
    df["spei_1"] = (df["wb1"] - df["wb1_mu"]) / sd(df["wb1_sd"])
    df["spei_3"] = (df["wb3"] - df["wb3_mu"]) / sd(df["wb3_sd"])
    df["rain_z"] = (df["rain_mm"] - df["rain_mu"]) / sd(df["rain_sd"])
    df["tmax_anom"] = df["tmax"] - df["tmax_mu"]
    df["tmax_z"] = df["tmax_anom"] / sd(df["tmax_sd"])
    # empirical percentile of this month's rainfall among the baseline years for that calendar month
    base_vals = {k: np.sort(v.values) for k, v in base.groupby("cal_month")["rain_mm"]}
    df["rain_pct"] = [np.searchsorted(base_vals.get(cm, np.array([])), r, side="right") / max(len(base_vals.get(cm, [])), 1)
                      for cm, r in zip(df["cal_month"], df["rain_mm"])]
    df["malaria_suit"] = malaria_suitability(df["tmean"].values, df["rain_mm"].values)
    return df.drop(columns=[c for c in stats.columns if c != "cal_month"])


def recent_window(daily: pd.DataFrame, monthly_idx: pd.DataFrame, days: int = 30) -> dict:
    """Indices for the last `days` days of daily data, compared with the 2001-2020 monthly baseline.
    Approximate: the window is compared with the baseline of the calendar months it covers."""
    from scipy.stats import norm
    d = daily.dropna().tail(days)
    if d.empty:
        return {}
    mi = pd.DatetimeIndex(monthly_idx["month"])
    base = monthly_idx[(mi.year >= BASELINE[0]) & (mi.year <= BASELINE[1])].copy()
    base["cal_month"] = pd.DatetimeIndex(base["month"]).month
    base["rain_daily"] = base["rain_mm"] / pd.DatetimeIndex(base["month"]).days_in_month
    g = base.groupby("cal_month")
    cm = d["date"].dt.month.values
    exp_rain = g["rain_daily"].mean().reindex(cm).sum()
    mid = int(pd.Series(cm).mode()[0])
    sd_rain = g["rain_mm"].std().get(mid, np.nan) * len(d) / 30
    tmax_mu, tmax_sd = g["tmax"].mean().reindex(cm).values, g["tmax"].std().reindex(cm).values
    rain = float(d["rain_mm"].sum())
    rain_z = (rain - exp_rain) / sd_rain if sd_rain and np.isfinite(sd_rain) else np.nan
    latest = monthly_idx.dropna(subset=["spei_3"]).tail(1)
    return {
        "rain_mm": rain, "rain_pct_normal": 100 * rain / exp_rain if exp_rain else np.nan,
        "rain_z": rain_z, "rain_pct": float(norm.cdf(rain_z)) if np.isfinite(rain_z) else np.nan,
        "tmax": float(d["tmax"].mean()), "tmax_anom": float(np.nanmean(d["tmax"].values - tmax_mu)),
        "tmax_z": float(np.nanmean((d["tmax"].values - tmax_mu) / tmax_sd)),
        "hot_days": int(np.nansum(d["tmax"].values > tmax_mu + 2 * tmax_sd)),
        "spei_3": float(latest["spei_3"].iloc[0]) if len(latest) else np.nan,
        "malaria_suit": float(malaria_suitability(np.array([d["tmean"].mean()]),
                                                  np.array([rain * 30 / len(d)]))[0]),
        "start": d["date"].min(), "end": d["date"].max(),
    }


# ------------------------------------------------------------------ hazard levels
LEVELS = ["Normal", "Watch", "Warning", "Severe"]
HAZARDS = {
    "flood": {"label": "Flood (heavy rainfall)", "metric": "rain_pct", "fmt": "{:.0%}",
              "metric_label": "Rainfall percentile vs 2001-2020",
              "cuts": [0.75, 0.90, 0.97], "higher_is_worse": True,
              "colours": ["rgba(160,160,160,0.25)", "#9ecae1", "#3182bd", "#08306b"]},
    "drought": {"label": "Drought (SPEI-3)", "metric": "spei_3", "fmt": "{:+.2f}",
                "metric_label": "SPEI-3 (z-score; below -1.3 = drought)",
                "cuts": [-0.8, -1.3, -1.6], "higher_is_worse": False,
                "colours": ["rgba(160,160,160,0.25)", "#fdd49e", "#d7301f", "#7f2704"]},
    "heat": {"label": "Heat (max temperature anomaly)", "metric": "tmax_z", "fmt": "{:+.2f}",
             "metric_label": "Max temperature anomaly (z-score)",
             "cuts": [1.0, 1.5, 2.0], "higher_is_worse": True,
             "colours": ["rgba(160,160,160,0.25)", "#fcbba1", "#ef3b2c", "#99000d"]},
    "malaria": {"label": "Disease: malaria climatic suitability", "metric": "malaria_suit", "fmt": "{:.2f}",
                "metric_label": "Suitability (0-1), climate only, not case data",
                "cuts": [0.25, 0.5, 0.75], "higher_is_worse": True,
                "colours": ["rgba(160,160,160,0.25)", "#dadaeb", "#9e9ac8", "#54278f"]},
}


def level(value: float, hazard: str) -> int:
    h = HAZARDS[hazard]
    if value is None or not np.isfinite(value):
        return -1
    c = h["cuts"]
    if h["higher_is_worse"]:
        return int(sum(value >= x for x in c))
    return int(sum(value <= x for x in c))


def latest_complete_month(today: date | None = None) -> date:
    today = today or date.today()
    first = today.replace(day=1)
    return (first - timedelta(days=1)).replace(day=1)


def days_in(month: pd.Timestamp) -> int:
    return calendar.monthrange(month.year, month.month)[1]
