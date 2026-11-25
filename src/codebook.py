"""Shared codebook: every question in the household and facility Kobo forms, tagged with
its role in the predictive model.

Source of the question lists: docs/forms/*.csv (re-generate with tools/extract_kobo_forms.py).
The role rules below are the single place that says which question feeds which feature.
Regenerate docs/codebook.csv with:  python -m src.codebook
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FORMS_DIR = Path(__file__).resolve().parent.parent / "docs" / "forms"

# Roles (model_use)
TARGET = "Outcome (model target)"
FEATURE = "Model feature"
INDEX = "Index component"
KEY = "Linkage key"
VALIDATION = "Validation / alert calibration"
EQUITY = "Equity disaggregation"
QUALITY = "Data quality"
COST = "Cost analysis (not a model input)"
CONTEXT = "Context / decision-support uptake"
PII = "Personal data: remove before warehouse"
ADMIN = "Survey administration"

# (form, regex on question name) -> (domain, model_use, feeds, transform)
RULES: list[tuple[str, str, tuple[str, str, str, str]]] = [
    # ---------------- shared / administrative
    ("*", r"^(consent_signature|consent_print|interviewer_signature|collector_signature|contact_phone|"
          r"SUP_name|SUP_signature|supervisor_check|supervisor_sign|A6_interviewer|A6_supervisor|"
          r"A8_collector|A8_supervisor|A3_village|A5_household|consent_method)$",
     ("Administration", PII, "-", "Drop or hash before data leave the Kobo server")),
    ("*", r"^(CONSENT|consent|record_mode|questionnaire_number|cluster_number|A6_code|intro|.*_intro|"
          r".*_instructions|.*_Instructions|consent_script|no_consent|thank_you|E_instructions|_i_.*)$",
     ("Administration", ADMIN, "-", "-")),
    ("*", r"^(.*consent_date|SUP_date|supervisor_date|A14|A16|D1|G3)$", ("Administration", ADMIN, "-", "Question header or date")),

    # ---------------- household: keys
    ("household", r"^A1$", ("Location", KEY, "country", "1=Kenya, 2=Malawi")),
    ("household", r"^A2$", ("Location", KEY, "site", "District code -> site name (docs/forms/household_district.csv)")),
    ("household", r"^A3_subcounty$", ("Location", KEY, "area_id", "site prefix + code, e.g. TUR-09")),
    ("household", r"^A4$", ("Location", KEY, "area centroid (climate extraction)",
                            "GPS averaged per area, then dropped (re-identification risk)")),
    # household: vulnerability index components
    ("household", r"^A10$", ("Household head", INDEX, "vulnerability_index; share_0 (female-headed)", "female head = 2")),
    ("household", r"^A10_age$", ("Household head", INDEX, "vulnerability_index", "head aged 60+")),
    ("household", r"^A12$", ("Household head", INDEX, "vulnerability_index", "no education = 1")),
    ("household", r"^A13", ("Household head", INDEX, "vulnerability_index", "climate-sensitive livelihood (livestock, crops, fishing)")),
    ("household", r"^A14_[a-f]$", ("Disability (Washington Group)", INDEX,
                                   "vulnerability_index; share_3 (disability)", "any item = 'A lot' or 'Cannot do'")),
    ("household", r"^A15$", ("Household composition", INDEX, "denominator for shares", "household size")),
    ("household", r"^A16_a_", ("Household composition", EQUITY, "share_1 (children under five)", "persons under 5 / A15")),
    ("household", r"^A16_e_", ("Household composition", EQUITY, "share_2 (older persons 60+)", "persons 60+ / A15")),
    ("household", r"^A16_", ("Household composition", INDEX, "vulnerability_index (dependency ratio)",
                             "(under 18 + 60+) / (18-59)")),
    ("household", r"^A17", ("Household composition", EQUITY, "ANC demand (pregnancies per household)", "-")),
    ("household", r"^A18", ("Household composition", EQUITY, "share_3 (disability)", "-")),
    ("household", r"^B[1-6]", ("Climate exposure (24 months)", VALIDATION,
                               "exposure index; check against NASA POWER / KMD / DCCMS events",
                               "months recorded without year: see review note")),
    ("household", r"^C(1|2|3|4|5|6|7|8|9|10)(_|$)", ("Recent illness", VALIDATION,
                                                     "relative risk by vulnerability group; HMIS under-reporting check",
                                                     "2-week prevalence")),
    ("household", r"^C1[1-4]", ("Illness during or after worst event", VALIDATION,
                                "event-health link; stakeholder validation", "-")),
    ("household", r"^D1_", ("Food security (FIES)", INDEX, "vulnerability_index (FIES raw score 0-8)", "count of Yes")),
    ("household", r"^D2$", ("Food security (FIES)", VALIDATION, "event-food link", "-")),
    ("household", r"^D3_", ("Diet diversity (HDDS)", INDEX, "vulnerability_index (HDDS 0-12, inverted)", "count of Yes")),
    ("household", r"^D[45]", ("Child nutrition (MUAC)", VALIDATION, "malnutrition prevalence vs HMIS admissions", "-")),
    ("household", r"^E", ("Mental health (K10)", COST, "K10 score (10-50); not yet a model input", "sum of E1 items")),
    ("household", r"^F", ("Cost of illness", COST, "Objective 3 cost estimates", "currency KES / MWK by country")),
    ("household", r"^G1_", ("Coping with illness costs", COST, "distress financing", "-")),
    ("household", r"^G2", ("Coping with illness costs", COST, "-", "-")),
    ("household", r"^G3_", ("Coping (rCSI)", INDEX, "vulnerability_index (reduced coping strategies index)",
                            "1*a + 2*b + 1*c + 3*d + 1*e (0-56)")),
    ("household", r"^G[45]", ("Coping with illness costs", INDEX, "vulnerability_index (catastrophic spending)", "Yes = 1")),
    ("household", r"^H1_", ("Assets", INDEX, "wealth index -> quintile; share_4 (lowest quintile)",
                            "first principal component of asset ownership")),
    ("household", r"^H2", ("Income", INDEX, "vulnerability_index (climate-sensitive income)", "-")),
    ("household", r"^H3$", ("Consumption", COST, "household spending (catastrophic health spending denominator)", "-")),
    ("household", r"^H4", ("Health insurance", INDEX, "vulnerability_index (no insurance)", "No = 1")),
    ("household", r"^A[5-9]", ("Respondent", ADMIN, "-", "-")),
    ("household", r"^A11$", ("Household head", INDEX, "vulnerability_index", "widowed / separated")),

    # ---------------- facility: keys
    ("facility", r"^A1_country$", ("Location", KEY, "country", "1=Kenya, 2=Malawi")),
    ("facility", r"^A2_district$", ("Location", KEY, "site", "district code -> site name")),
    ("facility", r"^A3_subcounty_ta$", ("Location", KEY, "area_id", "site prefix + code")),
    ("facility", r"^A3_facility_(code|name)$", ("Location", KEY, "facility_id (DHIS2 org unit)", "match to DHIS2 org unit")),
    ("facility", r"^A6_gps$", ("Location", KEY, "area centroid (climate extraction)", "kept: facilities are not personal data")),
    ("facility", r"^A[45]_", ("Facility profile", FEATURE, "readiness weighting", "-")),
    ("facility", r"^A9", ("Reference event", VALIDATION, "event dates for Section C, D5, D8, E2", "-")),
    ("facility", r"^A[78]", ("Respondent", ADMIN, "-", "-")),
    # facility: monthly HMIS (repeat B_monthly)
    ("facility", r"^B1_month$", ("Monthly HMIS", KEY, "month", "first day of month")),
    ("facility", r"^B1_(source|complete)$", ("Monthly HMIS", QUALITY, "reporting completeness weight", "-")),
    ("facility", r"^B2_", ("Monthly HMIS", FEATURE, "population", "sum of facility catchments per area")),
    ("facility", r"^B3_a_", ("Monthly HMIS", FEATURE, "opd_total; opd_female_share", "total; female / total")),
    ("facility", r"^B3_[bc]_", ("Monthly HMIS", TARGET, "admissions / deaths (extra outcomes)", "-")),
    ("facility", r"^B3_[de]_", ("Monthly HMIS", TARGET, "diarrhoea", "d + e (non-bloody + dysentery)")),
    ("facility", r"^B3_[fgh]_", ("Monthly HMIS", TARGET, "cholera (extra outcome)", "suspected f; confirmed g; deaths h")),
    ("facility", r"^B3_k_", ("Monthly HMIS", TARGET, "malaria", "confirmed cases (k)")),
    ("facility", r"^B3_[ijlm]_", ("Monthly HMIS", QUALITY, "malaria test positivity; severity", "k / j")),
    ("facility", r"^B3_[nopq]_", ("Monthly HMIS", TARGET, "respiratory", "n + o + p + q")),
    ("facility", r"^B3_[rstuv]_", ("Monthly HMIS", TARGET, "skin, eye, injury, snakebite, heat (extra outcomes)", "-")),
    ("facility", r"^B3_[wx]_", ("Monthly HMIS", TARGET, "malnutrition", "MAM (w) + SAM (x) admissions")),
    # facility: event response
    ("facility", r"^C1_", ("Facility finance", COST, "health-system cost of climate events", "-")),
    ("facility", r"^C[23]_", ("Event costs", COST, "health-system cost of climate events", "-")),
    ("facility", r"^C[4-7]_", ("Service disruption during event", VALIDATION,
                               "surge / disruption check for alert calibration", "during vs 3 months before")),
    # facility: readiness
    ("facility", r"^D1_.*_guideline$", ("Readiness: guidelines", INDEX, "readiness_score", "Observed=1, Reported=0.5, None=0")),
    ("facility", r"^D1_.*_offered$", ("Readiness: services", INDEX, "readiness_score", "Yes = 1")),
    ("facility", r"^D2_", ("Readiness: diagnostics", INDEX, "readiness_score", "Observed functioning = 1")),
    ("facility", r"^D[34]_", ("Readiness: preparedness", INDEX, "readiness_score; surge_capacity", "Yes, observed = 1")),
    ("facility", r"^D5_", ("Early warning use", VALIDATION, "did warnings reach the facility", "-")),
    ("facility", r"^D6_", ("Readiness: staffing", INDEX, "readiness_score", "sum in post / sum approved (cap 1)")),
    ("facility", r"^D7_", ("Readiness: training", INDEX, "readiness_score", "any trained = 1")),
    ("facility", r"^D8_", ("Staff absence during event", VALIDATION, "service disruption", "-")),
    ("facility", r"^E1_.*_availability$", ("Readiness: tracer medicines", INDEX, "readiness_score", "Observed, valid = 1")),
    ("facility", r"^E1_.*_(stockout|days)$", ("Stock-outs", FEATURE, "stockout_rate",
                                             "share of 14 tracer items stocked out in last 3 months")),
    ("facility", r"^E[234]", ("Supply chain", VALIDATION, "event supply disruption", "-")),
    ("facility", r"^F[1-7]", ("Infrastructure resilience", INDEX, "readiness_score", "water, power, cold chain, flood-proofing, WASH")),
    ("facility", r"^F[89]_|^F1[01]_", ("Referral", FEATURE, "referral_functional",
                                       "1 - referrals delayed (F11) / referrals (F10); transport; phone")),
    ("facility", r"^F1[2-5]_", ("Surge capacity", INDEX, "surge_capacity", "extra beds / beds, triage, surge plan, cholera area")),
    ("facility", r"^G", ("County / district planning", CONTEXT, "decision-support uptake; action lists", "once per county or district")),
]


def _role(form: str, name: str):
    for f, pat, role in RULES:
        if f in ("*", form) and re.match(pat, name):
            return role
    return ("Other", ADMIN, "-", "-")


def build_codebook() -> pd.DataFrame:
    frames = []
    for form in ("household", "facility"):
        path = FORMS_DIR / f"{form}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path).fillna("")
        roles = df["name"].apply(lambda n: _role(form, n))
        df["domain"], df["model_use"], df["feeds"], df["transform"] = zip(*roles)
        df.insert(0, "form", form)
        frames.append(df)
    cols = ["form", "group", "name", "label", "type", "choices", "domain", "model_use", "feeds",
            "transform", "relevant", "required"]
    return pd.concat(frames, ignore_index=True)[cols]


def choice_map(codebook: pd.DataFrame, form: str, name: str) -> dict[str, str]:
    """code -> label for one question."""
    row = codebook[(codebook.form == form) & (codebook.name == name)]
    if row.empty or not row.iloc[0]["choices"]:
        return {}
    out = {}
    for part in str(row.iloc[0]["choices"]).split(" | "):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# Known gaps between the forms and what the model needs (shown in the app and the docs).
GAPS = pd.DataFrame([
    ("facility", "B3 monthly HMIS", "No ANC (1st, 4+) or skilled delivery counts",
     "ANC / maternal demand outcome and sex-disaggregated ANC projections (proposal 5.7.6)",
     "Add B3 rows for ANC1, ANC4+, deliveries, or pull from DHIS2"),
    ("facility", "E1 stock-outs", "Stock-outs only for the last 3 months",
     "Monthly 'medicine stock-out rate' feature (proposal 5.7.3)",
     "Pull monthly stock status from DHIS2 / LMIS; E1 used as a static facility score until then"),
    ("facility", "F10-F11 referrals", "Referral totals are one-off, not monthly",
     "Monthly 'referral pathway functionality' feature",
     "Add monthly referrals out to B3, or extract from referral registers"),
    ("facility", "B monthly repeat", "60 months x ~115 numbers entered by hand per facility",
     "Data quality and field time", "Use the DHIS2 export as default; form repeat only where DHIS2 is missing"),
    ("household", "B1 months affected", "Month recorded without year (24-month recall)",
     "Linking exposure to the month-year key", "Record month and year, as B6 already does"),
    ("both", "Money amounts", "Currency not stated", "Cost analysis across countries",
     "Add KES / MWK or derive from country"),
], columns=["form", "where", "gap", "affects", "suggested fix"])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "docs" / "codebook.csv"
    cb = build_codebook()
    cb.to_csv(out, index=False)
    print(f"{len(cb)} variables -> {out}")
    print(cb.groupby(["form", "model_use"]).size().to_string())
