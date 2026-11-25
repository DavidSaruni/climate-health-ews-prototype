"""Kobo / DHIS2 import: sample exports -> warehouse -> forecast."""
import io

import pandas as pd

from src import importers as I
from src import samples
from src.data import build_warehouse
from src.features import climate_features
from src.pipeline import forecast


def _warehouse():
    hs = I.read_kobo(io.BytesIO(samples.household_export(n_per_area=6)), "h.xlsx")
    fs = I.read_kobo(io.BytesIO(samples.facility_export(per_area=1)), "f.xlsx")
    hh = I.household_scores(hs["main"])
    hh_area, rr = I.household_area_summary(hh)
    fac = I.facility_scores(fs["main"])
    hmis = I.hmis_from_kobo(fs, I.facility_scores(fs["main"]))
    wh, _, report = I.build_real_warehouse(hmis, hh_area, I.facility_area_summary(fac), fetch_climate=False)
    return hh, rr, fac, wh, report


def test_household_and_facility_scores():
    hh, rr, fac, wh, report = _warehouse()
    assert hh["area_id"].notna().all() and hh["vulnerability_index"].between(0, 1).all()
    assert len(rr) == 5
    assert fac["readiness_score"].between(0, 1).all()
    assert wh["area_id"].nunique() == 56 and wh["month"].nunique() == 60
    assert wh[["malaria", "diarrhoea", "respiratory", "malnutrition"]].notna().all().all()


def test_real_warehouse_runs_through_model():
    _, _, _, wh, _ = _warehouse()
    # offline test: borrow simulated climate instead of calling NASA POWER
    clim, syn = build_warehouse()
    cols = ["rainfall_mm", "temp_c", "heat_days", "flood_event", "drought_event"]
    wh = wh.drop(columns=cols).merge(syn[["area_id", "month"] + cols], on=["area_id", "month"], how="left")
    feat = climate_features(wh, clim)
    _, site_fc, _ = forecast(feat, "malaria", "Negative binomial (GLM)")
    assert len(site_fc) == 18 and site_fc["p_surge"].between(0, 1).all()


def test_dhis2_import_matches_org_units():
    long = I.read_dhis2(io.BytesIO(samples.dhis2_export(months=12)))
    mapping = I.dhis2_default_mapping(long["element"].unique().tolist())
    assert set(mapping.values()) >= {"malaria", "diarrhoea", "anc", "opd_total"}
    wide = I.hmis_from_dhis2(long, mapping)
    assert not wide.attrs["unmatched_orgunits"] and wide["area_id"].nunique() == 56


def test_labels_instead_of_codes():
    df = pd.DataFrame({"A3_subcounty": ["9"], "A10": ["Female"], "CONSENT": ["Yes"], "H4": ["No"]})
    hh = I.household_scores(df)
    assert hh.loc[0, "area_id"] == "TUR-09" and hh.loc[0, "female_head"] == 1 and hh.loc[0, "no_insurance"] == 1
