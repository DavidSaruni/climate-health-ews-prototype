"""Sample Kobo exports in the same layout KoboToolbox produces (XLSX: main sheet with full group
paths + one sheet per repeat group carrying _index / _parent_index). Values are SIMULATED.

Used as downloadable templates in the app and to test the importers end to end.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .codebook import build_codebook
from .config import SITES
from .data import build_warehouse


_CH: dict = {}


def _choices(cb, form, name):
    if not _CH:
        for r in cb.itertuples():
            _CH[(r.form, r.name)] = [p.split("=", 1) for p in str(r.choices).split(" | ") if "=" in p]                 if r.choices else []
            _CH[("path", r.form, r.name)] = f"{r.group}/{r.name}" if r.group else r.name
    return _CH.get((form, name), [])


def _code(cb, form, name, label):
    for k, v in _choices(cb, form, name):
        if v.lower() == label.lower():
            return k
    return ""


def _fill(cb, form, rows: pd.DataFrame, rng) -> dict:
    """Random but valid answer for every question in `rows`."""
    out = {}
    for q in rows[["name", "type"]].to_dict("records"):
        n = q["name"]
        if q["type"] == "select_one":
            ch = _choices(cb, form, n)
            out[n] = ch[rng.integers(len(ch))][0] if ch else ""
        elif q["type"] == "select_multiple":
            ch = _choices(cb, form, n)
            out[n] = " ".join(sorted({c[0] for c in [ch[i] for i in rng.integers(0, len(ch), 2)]})) if ch else ""
        elif q["type"] in ("int", "decimal"):
            out[n] = int(rng.integers(0, 6))
        else:
            out[n] = ""
    return out


def _path(cb, form, name):
    _choices(cb, form, name)
    return _CH.get(("path", form, name), name)


def household_export(n_per_area: int = 12, seed: int = 1) -> bytes:
    cb = build_codebook()
    rng = np.random.default_rng(seed)
    qs = cb[(cb.form == "household") & ~cb["model_use"].str.startswith("Personal")]
    main = []
    yes = lambda n: _code(cb, "household", n, "Yes")
    no = lambda n: _code(cb, "household", n, "No")
    idx = 0
    for site, m in SITES.items():
        lat0, lon0 = {"Turkana": (3.1, 35.6), "Mombasa": (-4.0, 39.7), "Kisumu": (-0.1, 34.8), "Nsanje": (-16.9, 35.3),
                      "Chikwawa": (-16.0, 34.8), "Balaka": (-15.0, 35.0)}[site]
        for code in m["sub_unit_codes"]:
            for _ in range(n_per_area):
                idx += 1
                r = _fill(cb, "household", qs, rng)
                v = m["vulnerability"]
                r.update({"CONSENT": yes("CONSENT"), "A1": "1" if m["country"] == "Kenya" else "2",
                          "A2": str(m["code"]), "A3_subcounty": str(code),
                          "A4": f"{lat0 + rng.normal(0, .15):.5f} {lon0 + rng.normal(0, .15):.5f} 0 5",
                          "A10": _code(cb, "household", "A10", "Female" if rng.random() < 0.3 + 0.2 * v else "Male"),
                          "A10_age": int(rng.integers(20, 80)), "A15": int(rng.integers(2, 10))})
                for k in "abcdefgh":
                    r[f"D1_{k}"] = yes(f"D1_{k}") if rng.random() < 0.2 + 0.5 * v else no(f"D1_{k}")
                for k in "abcdefghijkl":
                    r[f"D3_{k}"] = yes(f"D3_{k}") if rng.random() < 0.75 - 0.4 * v else no(f"D3_{k}")
                for k in "abcdefghijklmn":
                    r[f"H1_{k}"] = yes(f"H1_{k}") if rng.random() < 0.65 - 0.5 * v else no(f"H1_{k}")
                for g in "abcde":
                    for s in ("male", "female"):
                        r[f"A16_{g}_{s}"] = int(rng.integers(0, 3))
                for q in ("C1", "C5", "C8"):
                    r[q] = _code(cb, "household", q, "Yes" if rng.random() < 0.15 + 0.2 * v else "No")
                r["_index"], r["_uuid"] = idx, f"hh-{idx:05d}"
                main.append(r)
    df = pd.DataFrame(main)
    df.columns = [c if c.startswith("_") else _path(cb, "household", c) for c in df.columns]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Household Questionnaire", index=False)
    return buf.getvalue()


def facility_export(per_area: int = 2, months: int = 60, seed: int = 2) -> bytes:
    """Facilities with a 60-month HMIS series built from the synthetic warehouse (split across facilities)."""
    cb = build_codebook()
    rng = np.random.default_rng(seed)
    _, wh = build_warehouse()
    wh = wh[wh["month"] > wh["month"].max() - pd.DateOffset(months=months)]
    qs = cb[(cb.form == "facility") & ~cb["group"].str.contains("B_monthly") & ~cb["model_use"].str.startswith("Personal")]
    main, monthly = [], []
    idx = 0
    letters = {"malaria": "k", "diarrhoea": "d", "respiratory": "n", "malnutrition": "w"}
    for site, m in SITES.items():
        for code in m["sub_unit_codes"]:
            aid = f"{site[:3].upper()}-{code:02d}"
            series = wh[wh["area_id"] == aid]
            shares = rng.dirichlet(np.ones(per_area))
            for j in range(per_area):
                idx += 1
                r = _fill(cb, "facility", qs, rng)
                r.update({"record_mode": "1", "consent": _code(cb, "facility", "consent", "Yes"),
                          "A1_country": "1" if m["country"] == "Kenya" else "2", "A2_district": str(m["code"]),
                          "A3_subcounty_ta": str(code), "A3_facility_code": f"F{idx:04d}",
                          "A3_facility_name": f"{site} facility {idx}",
                          "A6_gps": ""})
                r["F12_beds"], r["F12_extra"] = int(rng.integers(0, 40)), int(rng.integers(0, 15))
                r["_index"], r["_uuid"] = idx, f"fac-{idx:04d}"
                main.append(r)
                for mrow in series.to_dict("records"):
                    rec = {"B1_month": mrow["month"].strftime("%Y-%m-%d"), "B1_source": "1",
                           "B1_complete": _code(cb, "facility", "B1_complete", "Yes"),
                           "B2_population": int(mrow["population"] * shares[j]),
                           "B3_a_total": int(mrow["opd_total"] * shares[j]),
                           "B3_a_u5_female": int(mrow["opd_total"] * shares[j] * 0.12),
                           "B3_a_5plus_female": int(mrow["opd_total"] * shares[j] * (mrow["opd_female_share"] - 0.12))}
                    for out, l in letters.items():
                        rec[f"B3_{l}_total"] = int(rng.binomial(int(mrow[out]), shares[j]))
                    rec["_parent_index"], rec["_parent_table_name"] = idx, "Health Facility Questionnaire"
                    monthly.append(rec)
    df = pd.DataFrame(main)
    df.columns = [c if c.startswith("_") else _path(cb, "facility", c) for c in df.columns]
    mdf = pd.DataFrame(monthly)
    mdf.insert(0, "_index", range(1, len(mdf) + 1))
    mdf.columns = [c if c.startswith("_") else _path(cb, "facility", c) for c in mdf.columns]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Health Facility Questionnaire", index=False)
        mdf.to_excel(xw, sheet_name="B_monthly", index=False)
    return buf.getvalue()


def dhis2_export(months: int = 24) -> bytes:
    """DHIS2 plain-table CSV (Data, Period, Organisation unit, Value) at sub-county / TA level."""
    _, wh = build_warehouse()
    wh = wh[wh["month"] > wh["month"].max() - pd.DateOffset(months=months)]
    names = {"malaria": "Malaria confirmed cases", "diarrhoea": "Diarrhoea cases", "anc": "ANC 1st visits",
             "malnutrition": "SAM admissions", "respiratory": "Pneumonia cases", "opd_total": "Outpatient attendance"}
    rows = [{"Data": names[c], "Period": r["month"].strftime("%Y%m"), "Organisation unit": r["sub_unit"],
             "Organisation unit parent": r["site"], "Value": int(r[c])} for r in wh.to_dict("records") for c in names]
    return pd.DataFrame(rows).to_csv(index=False).encode()
