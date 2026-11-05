"""Synthetic data warehouse.

Builds one row per area (sub-unit) per month, with the four streams from proposal
section 5.7.2 already joined on the area and month keys:

  1. Meteorological records (KMD / DCMS)    -> rainfall, temperature, heat days, events
  2. Routine health data (HMIS / DHIS2)       -> monthly cases, OPD, stock-outs, referrals
  3. Household survey                         -> vulnerability index, group shares
  4. Facility assessments                     -> readiness and surge capacity

ALL VALUES ARE SIMULATED. The relationships are plausible, not estimated. When real
extracts arrive, replace `build_warehouse()` with a loader that returns the same columns
(see `DATA_DICTIONARY`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUTCOMES, SITES, VULNERABILITY_GROUPS

CLIMATE_START = "2006-01-01"   # gives a 20-year climate baseline
HEALTH_START = "2018-01-01"    # > 5 years of facility-level monthly HMIS data
END = "2026-09-01"             # latest complete month in the prototype

DATA_DICTIONARY = pd.DataFrame([
    # name, stream, unit, description
    ("area_id", "Keys", "code", "Unique sub-unit code (site prefix + number)"),
    ("site", "Keys", "name", "County (Kenya) or district (Malawi)"),
    ("country", "Keys", "name", "Kenya or Malawi"),
    ("sub_unit", "Keys", "name", "Sub-county / health area"),
    ("month", "Keys", "YYYY-MM-01", "Calendar month (temporal key)"),
    ("rainfall_mm", "Meteorological", "mm/month", "Total monthly rainfall"),
    ("temp_c", "Meteorological", "deg C", "Mean monthly maximum temperature"),
    ("heat_days", "Meteorological", "days", "Days above the site-specific heat threshold"),
    ("flood_event", "Meteorological", "0/1", "Declared flood in the month"),
    ("drought_event", "Meteorological", "0/1", "Declared drought in the month"),
    ("population", "HMIS/DHIS2", "persons", "Catchment population"),
    ("opd_total", "HMIS/DHIS2", "visits", "Total outpatient attendance"),
    ("opd_female_share", "HMIS/DHIS2", "0-1", "Share of OPD visits by women and girls"),
    *[(k, "HMIS/DHIS2", "cases", v["label"]) for k, v in OUTCOMES.items()],
    ("stockout_rate", "HMIS/DHIS2", "0-1", "Share of tracer medicines stocked out"),
    ("referral_functional", "Facility assessment", "0-1", "Share of referrals completed"),
    ("readiness_score", "Facility assessment", "0-1", "Facility climate readiness score"),
    ("surge_capacity", "Facility assessment", "0-1", "Surge capacity score (beds, staff, triage)"),
    ("vulnerability_index", "Household survey", "0-1", "Composite socio-economic vulnerability"),
    *[(f"share_{i}", "Household survey", "0-1", f"Population share: {g}")
      for i, g in enumerate(VULNERABILITY_GROUPS)],
], columns=["variable", "stream", "unit", "description"])


def _ar1(n: int, phi: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _site_climate(site: str, months: pd.DatetimeIndex, enso: np.ndarray,
                  rng: np.random.Generator) -> pd.DataFrame:
    c = SITES[site]["climate"]
    n = len(months)
    m = months.month.values
    seasonal = np.where(np.isin(m, c["rain_season"]), 2.6, 0.45)
    seasonal = seasonal / seasonal.mean()
    local = _ar1(n, 0.6, 0.35, rng)
    anomaly = 0.7 * enso + local
    rain = c["rain_mean"] * seasonal * np.exp(0.45 * anomaly) * rng.gamma(4, 1 / 4, n)
    years = (months.year.values - 2006) + m / 12
    temp_anom = _ar1(n, 0.5, 0.5, rng) - 0.25 * anomaly
    temp = c["temp_mean"] + c["temp_amp"] * np.cos(2 * np.pi * (m - 2) / 12) + 0.03 * years + temp_anom
    return pd.DataFrame({"month": months, "rainfall_mm": rain.round(1), "temp_c": temp.round(2),
                         "_anom": anomaly})


def build_warehouse(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (climate_history, warehouse).

    climate_history: site x month from 2006 (for baselines).
    warehouse: area x month from 2018, all four streams joined.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(CLIMATE_START, END, freq="MS")
    n = len(months)
    enso = {"Kenya": _ar1(n, 0.9, 0.3, rng), "Malawi": _ar1(n, 0.9, 0.3, rng)}

    climate_frames, rows = [], []
    for site, meta in SITES.items():
        clim = _site_climate(site, months, enso[meta["country"]], rng)
        clim["site"] = site
        # heat days relative to the site's own 90th percentile of temperature
        thr = np.quantile(clim["temp_c"], 0.75)
        clim["heat_days"] = rng.poisson(np.clip(3 + 6 * (clim["temp_c"] - thr), 0.2, 25))
        # declared events: extreme rainfall / persistent deficit, scaled by how prone the site is
        r_pct = clim.groupby(clim.month.dt.month)["rainfall_mm"].rank(pct=True)
        dry3 = clim["_anom"].rolling(3, min_periods=1).mean()
        clim["flood_event"] = ((r_pct > 0.9) & (rng.random(n) < meta["climate"]["flood_prone"])).astype(int)
        clim["drought_event"] = ((dry3 < -0.55) & (rng.random(n) < meta["climate"]["drought_prone"])).astype(int)
        climate_frames.append(clim)

        for i, sub in enumerate(meta["sub_units"]):
            area_id = f"{site[:3].upper()}{i + 1:02d}"
            sub_rng = np.random.default_rng(rng.integers(1e9))
            vuln = float(np.clip(meta["vulnerability"] + sub_rng.normal(0, 0.06), 0.2, 0.95))
            readiness = float(np.clip(1 - vuln + sub_rng.normal(0, 0.08), 0.15, 0.9))
            surge_cap = float(np.clip(readiness + sub_rng.normal(0, 0.07), 0.1, 0.9))
            pop = int(sub_rng.uniform(80_000, 320_000))
            shares = np.clip(np.array([0.30, 0.17, 0.06, 0.04, 0.20]) * (0.8 + 0.4 * vuln)
                             + sub_rng.normal(0, 0.01, 5), 0.01, 0.6)
            df = clim.copy()
            df["rainfall_mm"] = (df["rainfall_mm"] * sub_rng.normal(1, 0.08, n)).clip(0).round(1)
            df["temp_c"] = (df["temp_c"] + sub_rng.normal(0, 0.3)).round(2)
            df["area_id"], df["sub_unit"], df["country"] = area_id, sub, meta["country"]
            df["population"] = (pop * (1.025 ** ((df.month.dt.year - 2018) + df.month.dt.month / 12))).astype(int)
            df["vulnerability_index"] = round(vuln, 3)
            df["readiness_score"] = round(readiness, 3)
            df["surge_capacity"] = round(surge_cap, 3)
            for j, s in enumerate(shares):
                df[f"share_{j}"] = round(float(s), 3)
            rows.append(_simulate_health(df, site, vuln, readiness, sub_rng))

    climate_history = pd.concat(climate_frames).drop(columns="_anom").reset_index(drop=True)
    wh = pd.concat(rows).drop(columns="_anom")
    wh = wh[wh.month >= HEALTH_START].reset_index(drop=True)
    cols = DATA_DICTIONARY["variable"].tolist()
    return climate_history, wh[cols]


def _simulate_health(df: pd.DataFrame, site: str, vuln: float, readiness: float,
                     rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    m = df.month.dt.month.values
    z = lambda s: (s - s.mean()) / s.std()
    rain_z = z(np.log1p(df["rainfall_mm"]))
    temp_z = z(df["temp_c"])
    heat_z = z(df["heat_days"].astype(float))
    lag = lambda s, k: pd.Series(s).shift(k).bfill().values
    flood, drought = df["flood_event"].values, df["drought_event"].values
    dry = -pd.Series(df["_anom"]).rolling(3, min_periods=1).mean().values

    df["stockout_rate"] = np.clip(0.08 + 0.15 * (1 - readiness) + 0.18 * flood + 0.08 * drought
                                  + rng.normal(0, 0.03, n), 0, 0.9).round(3)
    df["referral_functional"] = np.clip(0.55 + 0.4 * readiness - 0.25 * flood
                                        + rng.normal(0, 0.04, n), 0.05, 1).round(3)

    site_mult = {"Turkana": {"malaria": 0.35, "malnutrition": 2.2, "respiratory": 1.2},
                 "Kisumu": {"malaria": 1.8}, "Mombasa": {"diarrhoea": 1.3, "malaria": 0.7},
                 "Nsanje": {"diarrhoea": 1.4, "malaria": 1.3}, "Chikwawa": {"malaria": 1.3},
                 "Balaka": {"malnutrition": 1.6}}.get(site, {})
    pop_k = df["population"].values / 150_000
    latent = _ar1(n, 0.7, 0.08, rng)
    stock_lag = lag(df["stockout_rate"].values, 1)

    eta = {
        "malaria": 0.45 * lag(rain_z, 1) + 0.25 * lag(rain_z, 2) + 0.2 * lag(temp_z, 1)
                   + 0.35 * lag(flood, 1) + 0.6 * lag(flood, 1) * (1 - readiness),
        "diarrhoea": 0.55 * flood + 0.3 * lag(flood, 1) + 0.15 * rain_z + 0.25 * drought
                     + 0.8 * flood * vuln + 0.6 * stock_lag,
        "respiratory": 0.25 * heat_z + 0.2 * drought + 0.15 * np.cos(2 * np.pi * (m - 7) / 12),
        "malnutrition": 0.45 * lag(drought, 2) + 0.3 * lag(dry, 2) + 0.3 * lag(dry, 3)
                        + 0.5 * lag(drought, 2) * vuln,
        "anc": -0.25 * flood - 0.15 * lag(flood, 1) + 0.03 * ((df.month.dt.year - 2018).values),
    }
    opd = np.zeros(n)
    for k, meta in OUTCOMES.items():
        mu = meta["base"] * pop_k * site_mult.get(k, 1.0) * (0.85 + 0.3 * vuln) * np.exp(eta[k] + latent)
        alpha = 0.06
        y = rng.poisson(rng.gamma(1 / alpha, mu * alpha))
        df[k] = y
        if k != "anc":
            opd += y
    df["opd_total"] = (opd + rng.poisson(2500 * pop_k)).astype(int)
    df["opd_female_share"] = np.clip(0.56 + rng.normal(0, 0.02, n), 0.4, 0.7).round(3)
    return df


def validate_upload(df: pd.DataFrame) -> list[str]:
    """Return a list of problems with an uploaded warehouse file (empty if it is usable)."""
    problems = []
    required = ["area_id", "site", "sub_unit", "month", "population", "rainfall_mm", "temp_c",
                "heat_days", "flood_event", "drought_event", *OUTCOMES.keys()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {', '.join(missing)}")
    if "month" in df.columns:
        try:
            pd.to_datetime(df["month"])
        except Exception:
            problems.append("`month` could not be read as a date (use YYYY-MM-01).")
    if not problems and df.duplicated(["area_id", "month"]).any():
        problems.append("Duplicate rows found for the same area_id and month.")
    return problems


def fill_optional(df: pd.DataFrame) -> pd.DataFrame:
    """Fill optional columns with neutral defaults so partial real data still runs."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    defaults = {"country": "", "opd_female_share": 0.56, "stockout_rate": 0.1,
                "referral_functional": 0.8, "readiness_score": 0.5, "surge_capacity": 0.5,
                "vulnerability_index": 0.5, "opd_total": np.nan}
    defaults.update({f"share_{i}": 0.1 for i in range(len(VULNERABILITY_GROUPS))})
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    return df.sort_values(["area_id", "month"]).reset_index(drop=True)
