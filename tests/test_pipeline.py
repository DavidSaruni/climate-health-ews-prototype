"""Pipeline tests: synthetic data, CSV upload round trip, forecasts."""
import io

import pandas as pd

from src.data import build_warehouse, fill_optional, validate_upload
from src.features import climate_features
from src.pipeline import forecast


def test_upload_round_trip():
    _, wh = build_warehouse()
    csv = io.StringIO(wh.to_csv(index=False))
    up = pd.read_csv(csv)
    assert validate_upload(up) == []
    feat = climate_features(fill_optional(up), None)
    area, site, _ = forecast(feat, "diarrhoea", "Negative binomial (GLM)")
    assert len(site) == 18 and site["p_surge"].between(0, 1).all()


def test_upload_rejects_missing_columns():
    _, wh = build_warehouse()
    problems = validate_upload(wh.drop(columns=["malaria", "month"]))
    assert problems and "malaria" in problems[0]
