"""Hazard indices and geometry helpers (offline)."""
import numpy as np
import pandas as pd

from src import climate as C
from src import geo


def _monthly(seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2001-01-01", "2026-09-01", freq="MS")
    season = 80 + 70 * np.sin(2 * np.pi * months.month / 12)
    return pd.DataFrame({"month": months, "rain_mm": rng.gamma(4, season / 4), "tmax": 30 + rng.normal(0, 1, len(months)),
                         "tmin": 18 + rng.normal(0, 1, len(months)), "tmean": 24 + rng.normal(0, 1, len(months))})


def test_indices_are_standardised_over_baseline():
    mi = C.monthly_indices(_monthly(), lat=-1.0)
    base = mi[(mi.month.dt.year >= 2001) & (mi.month.dt.year <= 2020)]
    assert abs(base["spei_3"].mean()) < 0.1 and abs(base["spei_3"].std() - 1) < 0.15
    assert mi["rain_pct"].between(0, 1).all() and mi["malaria_suit"].between(0, 1).all()
    assert (mi["pet_mm"] > 50).all()


def test_levels():
    assert C.level(0.98, "flood") == 3 and C.level(0.5, "flood") == 0
    assert C.level(-2.0, "drought") == 3 and C.level(0.3, "drought") == 0
    assert C.level(float("nan"), "heat") == -1


def test_malaria_suitability_rules():
    s = C.malaria_suitability(np.array([15, 25, 25, 42]), np.array([200, 200, 20, 200]))
    assert list(np.round(s, 2)) == [0.0, 1.0, 0.25, 0.0]


def test_centroid_and_simplify():
    sq = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
    lat, lon = geo.centroid(sq)
    assert abs(lat - 1) < 1e-9 and abs(lon - 1) < 1e-9
    ring = [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 500)]
    assert len(geo.simplify_geometry({"type": "Polygon", "coordinates": [ring]}, 0.01)["coordinates"][0]) < 100
    assert geo.norm_name("Turkana County") == geo.norm_name("turkana")
