"""Load real field data into the warehouse format.

Inputs
  - Household survey: Kobo export (XLSX with all sheets, or CSV of the main sheet)
  - Facility survey:  Kobo export (XLSX; the B_monthly repeat sheet carries the 60-month HMIS series)
  - DHIS2:            CSV "plain table" export (Data, Period, Organisation unit, Value), optional
  - Climate:          NASA POWER at each area's centroid (until KMD / DCCMS station data arrive)

Output: a warehouse with the same columns as src/data.DATA_DICTIONARY, plus extra outcomes.
Every formula here is documented in docs/CODEBOOK.md.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .codebook import build_codebook, choice_map
from .config import OUTCOMES, SITE_COORDS, SITES, VULNERABILITY_GROUPS

_CB = None


def codebook() -> pd.DataFrame:
    global _CB
    if _CB is None:
        _CB = build_codebook()
    return _CB


# ------------------------------------------------------------------ reading Kobo exports
def _short(col: str) -> str:
    """'consented/facility/section_B/B_monthly/B3_k_total' -> 'B3_k_total'; keep Kobo meta columns."""
    if col.startswith("_"):
        return col
    return col.split("/")[-1]


def read_kobo(file, name: str = "") -> dict[str, pd.DataFrame]:
    """Return {'main': df, '<repeat name>': df, ...} with short column names."""
    name = name or getattr(file, "name", "")
    if str(name).lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(file, sheet_name=None, dtype=str)
    else:
        raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
        sheets = {"main": pd.read_csv(io.StringIO(text), dtype=str, sep=sep)}
    out = {}
    for sname, df in sheets.items():
        df = df.copy()
        df.columns = [_short(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        is_repeat = "_parent_index" in df.columns
        out[sname if is_repeat else "main"] = df
    return out


def _codes(df: pd.DataFrame, form: str) -> pd.DataFrame:
    """Convert choice labels to codes where an export used labels instead of XML values."""
    cb = codebook()
    df = df.copy()
    for col in df.columns:
        cm = choice_map(cb, form, col)
        if not cm:
            continue
        lab2code = {v.lower(): k for k, v in cm.items()}
        s = df[col].astype("string").str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        df[col] = s.map(lambda v: v if pd.isna(v) or v in cm else lab2code.get(str(v).lower(), v))
    return df


def _is(df, form, col, *labels) -> pd.Series:
    """True where the answer's label is one of `labels` (robust to code differences between forms)."""
    if col not in df:
        return pd.Series(False, index=df.index)
    cm = choice_map(codebook(), form, col)
    codes = {k for k, v in cm.items() if v.lower() in {l.lower() for l in labels}}
    return df[col].astype("string").isin(codes).fillna(False)


def _num(df, col) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def _area_lookup(form: str) -> dict:
    """(district code, sub-unit code) -> (site, area_id, sub_unit)."""
    out = {}
    for site, m in SITES.items():
        for code, label in zip(m.get("sub_unit_codes", []), m["sub_units"]):
            out[str(code)] = (site, f"{site[:3].upper()}-{code:02d}", label)
    return out


def _assign_area(df: pd.DataFrame, col: str):
    """Sub-county / TA code -> (site, area_id) as two plain object series."""
    look = _area_lookup("")
    codes = df.get(col, pd.Series(index=df.index, dtype=object)).astype(object).map(
        lambda v: None if pd.isna(v) else str(v).strip().removesuffix(".0"))
    site = codes.map(lambda c: look[c][0] if c in look else np.nan).astype(object)
    aid = codes.map(lambda c: look[c][1] if c in look else np.nan).astype(object)
    return site.values, aid.values


def _gps(series: pd.Series) -> pd.DataFrame:
    """Kobo geopoint 'lat lon alt acc' -> lat, lon."""
    parts = series.astype("string").str.split(" ", expand=True)
    if parts.shape[1] < 2:
        return pd.DataFrame({"lat": np.nan, "lon": np.nan}, index=series.index)
    return pd.DataFrame({"lat": pd.to_numeric(parts[0], errors="coerce"),
                         "lon": pd.to_numeric(parts[1], errors="coerce")}, index=series.index)


# ------------------------------------------------------------------ household survey
def household_scores(main: pd.DataFrame) -> pd.DataFrame:
    """One row per household with the components of the vulnerability index."""
    F = "household"
    df = _codes(main, F)
    if "CONSENT" in df:
        df = df[_is(df, F, "CONSENT", "Yes") | df["CONSENT"].isna()]
    out = pd.DataFrame(index=df.index)
    out["site"], out["area_id"] = _assign_area(df, "A3_subcounty")
    out[["lat", "lon"]] = _gps(df["A4"]) if "A4" in df else np.nan

    out["female_head"] = _is(df, F, "A10", "Female").astype(float)
    out["head_60"] = (_num(df, "A10_age") >= 60).astype(float)
    out["no_education"] = _is(df, F, "A12", "None").astype(float)
    out["widowed_sep"] = _is(df, F, "A11", "Widowed", "Separated or divorced").astype(float)
    wg = [c for c in df if re.match(r"^A14_[a-f]$", c)]
    out["head_disability"] = pd.concat([_is(df, F, c, "A lot of difficulty", "Cannot do at all") for c in wg],
                                       axis=1).any(axis=1).astype(float) if wg else 0.0
    out["member_disability"] = _is(df, F, "A18", "Yes").astype(float)
    out["disability_any"] = out[["head_disability", "member_disability"]].max(axis=1)

    grp = lambda g: _num(df, f"A16_{g}_male").fillna(0) + _num(df, f"A16_{g}_female").fillna(0)
    u5, c517, a1835, a3659, o60 = (grp(g) for g in "abcde")
    size = _num(df, "A15").where(lambda s: s > 0, u5 + c517 + a1835 + a3659 + o60)
    out["size"], out["u5"], out["o60"] = size, u5, o60
    out["dependency"] = ((u5 + c517 + o60) / (a1835 + a3659).replace(0, np.nan)).clip(upper=4).fillna(4)
    out["has_u5"] = (u5 > 0).astype(float)
    out["has_60"] = (o60 > 0).astype(float)

    d1 = [c for c in df if re.match(r"^D1_[a-h]$", c)]
    out["fies"] = sum(_is(df, F, c, "Yes").astype(int) for c in d1) if d1 else np.nan
    d3 = [c for c in df if re.match(r"^D3_[a-l]$", c)]
    out["hdds"] = sum(_is(df, F, c, "Yes").astype(int) for c in d3) if d3 else np.nan
    w = {"a": 1, "b": 2, "c": 1, "d": 3, "e": 1}
    out["rcsi"] = sum(_num(df, f"G3_{k}").fillna(0).clip(0, 7) * v for k, v in w.items())
    out["catastrophic"] = _is(df, F, "G5", "Yes").astype(float)
    out["no_insurance"] = _is(df, F, "H4", "No").astype(float)
    out["any_illness"] = (_is(df, F, "C1", "Yes") | _is(df, F, "C5", "Yes") | _is(df, F, "C8", "Yes")).astype(float)
    out["exposed"] = pd.concat([_is(df, F, f"B1_{k}", "Yes") for k in "abcd"], axis=1).any(axis=1).astype(float)

    # wealth index: first principal component of asset ownership (H1_a..n yes/no)
    h1 = [c for c in df if re.match(r"^H1_[a-n]$", c)]
    if h1:
        X = pd.concat([_is(df, F, c, "Yes").astype(float) for c in h1], axis=1).values
        Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
        _, _, vt = np.linalg.svd(Xs, full_matrices=False)
        pc = Xs @ vt[0]
        if np.corrcoef(pc, X.sum(1))[0, 1] < 0:
            pc = -pc
        out["wealth"] = pc
        out["wealth_q"] = pd.qcut(pd.Series(pc).rank(method="first"), 5, labels=False).values + 1
    else:
        out["wealth"], out["wealth_q"] = np.nan, np.nan
    out["lowest_q"] = (out["wealth_q"] == 1).astype(float)

    comps = pd.DataFrame({
        "female_head": out["female_head"], "head_60": out["head_60"], "no_education": out["no_education"],
        "disability": out["disability_any"], "dependency": out["dependency"] / 4, "fies": out["fies"] / 8,
        "low_diet": 1 - out["hdds"] / 12, "rcsi": out["rcsi"] / 56, "catastrophic": out["catastrophic"],
        "no_insurance": out["no_insurance"],
        "poverty": 1 - (out["wealth_q"] - 1) / 4})
    out["vulnerability_index"] = comps.mean(axis=1, skipna=True)
    return out


GROUP_FLAGS = ["female_head", "has_u5", "has_60", "disability_any", "lowest_q"]


def household_area_summary(hh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Area-level vulnerability index and group shares; relative risk of illness by group."""
    g = hh.dropna(subset=["area_id"]).groupby("area_id")
    area = pd.DataFrame({
        "site": g["site"].first(), "households": g.size(),
        "vulnerability_index": g["vulnerability_index"].mean(),
        "share_0": g["female_head"].mean(),
        "share_1": g["u5"].sum() / g["size"].sum(),
        "share_2": g["o60"].sum() / g["size"].sum(),
        "share_3": g["disability_any"].mean(),
        "share_4": g["lowest_q"].mean(),
        "lat": g["lat"].mean(), "lon": g["lon"].mean()}).reset_index()
    rr = []
    for i, flag in enumerate(GROUP_FLAGS):
        a, b = hh[hh[flag] == 1]["any_illness"], hh[hh[flag] == 0]["any_illness"]
        rr.append({"group": VULNERABILITY_GROUPS[i], "households": int(len(a)),
                   "illness_in_group": a.mean(), "illness_others": b.mean(),
                   "relative_risk": a.mean() / b.mean() if len(b) and b.mean() > 0 else np.nan})
    return area, pd.DataFrame(rr)


# ------------------------------------------------------------------ facility survey
TRACERS = list("abcdefghijklmn")


def facility_scores(main: pd.DataFrame) -> pd.DataFrame:
    F = "facility"
    df = _codes(main, F)
    if "record_mode" in df:
        df = df[~_is(df, F, "record_mode", "County/district health management team (Section G)")]
    out = pd.DataFrame(index=df.index)
    out["_index"] = df.get("_index", pd.Series(df.index.astype(str), index=df.index))
    out["facility_code"] = df.get("A3_facility_code")
    out["facility_name"] = df.get("A3_facility_name")
    out["site"], out["area_id"] = _assign_area(df, "A3_subcounty_ta")
    out[["lat", "lon"]] = _gps(df["A6_gps"]) if "A6_gps" in df else np.nan

    mean_of = lambda cols: pd.concat(cols, axis=1).mean(axis=1) if cols else pd.Series(np.nan, index=df.index)
    guide = [(_is(df, F, c, "Observed").astype(float) + 0.5 * _is(df, F, c, "Reported, not seen").astype(float))
             for c in df if re.match(r"^D1_[a-f]_guideline$", c)]
    offered = [_is(df, F, c, "Yes").astype(float) for c in df if re.match(r"^D1_[a-f]_offered$", c)]
    diag = [_is(df, F, c, "Observed functioning").astype(float) for c in df if re.match(r"^D2_[a-f]$", c)]
    tracer = [_is(df, F, f"E1_{k}_availability", "Observed, valid item").astype(float) for k in TRACERS
              if f"E1_{k}_availability" in df]
    stock = [_is(df, F, f"E1_{k}_stockout", "Yes").astype(float) for k in TRACERS if f"E1_{k}_stockout" in df]
    inpost = sum(_num(df, c).fillna(0) for c in df if re.match(r"^D6_[a-g]_inpost$", c))
    approved = sum(_num(df, c).fillna(0) for c in df if re.match(r"^D6_[a-g]_approved$", c))
    out["staffing"] = (inpost / approved.replace(0, np.nan)).clip(upper=1) if isinstance(approved, pd.Series) else np.nan
    prep = [_is(df, F, "D3_plan", "Yes, observed").astype(float) + 0.5 * _is(df, F, "D3_plan", "Yes, not seen").astype(float),
            _is(df, F, "D4_risk", "Yes").astype(float), _is(df, F, "F13_triage", "Observed").astype(float),
            _is(df, F, "F13_plan", "Yes, observed").astype(float) + 0.5 * _is(df, F, "F13_plan", "Yes, not seen").astype(float),
            _is(df, F, "F14_cholera", "Yes").astype(float)]
    wash = (_num(df, "F7_functioning") / _num(df, "F7_total").replace(0, np.nan)).clip(0, 1)
    infra = [_is(df, F, "F1_water_today", "Yes").astype(float),
             (~_is(df, F, "F3_backup", "None") & df.get("F3_backup", pd.Series(index=df.index)).notna()).astype(float),
             (_is(df, F, "F4_fridge", "Yes") & _is(df, F, "F4_temperature", "Yes")).astype(float),
             _is(df, F, "F6_raised", "Yes").astype(float), _is(df, F, "F6_cooling", "Yes").astype(float),
             wash.fillna(0), _is(df, F, "F8_communication", "Yes").astype(float)]

    out["guidelines"], out["services"], out["diagnostics"] = mean_of(guide), mean_of(offered), mean_of(diag)
    out["tracer_meds"], out["preparedness"], out["infrastructure"] = mean_of(tracer), mean_of(prep), mean_of(infra)
    domains = ["guidelines", "services", "diagnostics", "tracer_meds", "staffing", "preparedness", "infrastructure"]
    out["readiness_score"] = out[domains].mean(axis=1, skipna=True)
    out["stockout_rate"] = mean_of(stock)

    beds, extra = _num(df, "F12_beds"), _num(df, "F12_extra")
    transport = df.get("F8_transport", pd.Series("", index=df.index)).astype("string").fillna("")
    none_code = {k for k, v in choice_map(codebook(), F, "F8_transport").items() if v == "None"}
    has_transport = transport.map(lambda v: bool(v) and not set(str(v).split()) <= none_code).astype(float)
    out["surge_capacity"] = pd.concat([(extra / beds.replace(0, np.nan)).clip(0, 1).fillna(0), has_transport,
                                       prep[3], prep[4], out["staffing"].fillna(0)], axis=1).mean(axis=1)
    referred, delayed = _num(df, "F10_referrals"), _num(df, "F11_number").fillna(0)
    delayed_share = (delayed / referred.replace(0, np.nan)).clip(0, 1).fillna(0) * _is(df, F, "F11_delayed", "Yes")
    out["referral_functional"] = pd.concat([1 - delayed_share, has_transport,
                                            _is(df, F, "F8_communication", "Yes").astype(float)], axis=1).mean(axis=1)
    return out


def facility_area_summary(fac: pd.DataFrame) -> pd.DataFrame:
    g = fac.dropna(subset=["area_id"]).groupby("area_id")
    return pd.DataFrame({
        "site": g["site"].first(), "facilities": g.size(),
        "readiness_score": g["readiness_score"].mean(), "surge_capacity": g["surge_capacity"].mean(),
        "referral_functional": g["referral_functional"].mean(), "stockout_rate": g["stockout_rate"].mean(),
        "lat": g["lat"].mean(), "lon": g["lon"].mean()}).reset_index()


# ------------------------------------------------------------------ monthly HMIS
OUTCOME_SOURCES = {  # warehouse column -> B3 indicator letters (totals)
    "malaria": ["k"], "diarrhoea": ["d", "e"], "respiratory": ["n", "o", "p", "q"], "malnutrition": ["w", "x"],
    "cholera": ["f"], "admissions": ["b"], "deaths": ["c"], "injuries": ["t"], "heat_illness": ["v"],
    "skin_eye": ["r", "s"], "snakebite": ["u"],
}
EXTRA_OUTCOMES = [k for k in OUTCOME_SOURCES if k not in OUTCOMES]


def _b3(df, letter, part="total"):
    col = f"B3_{letter}_{part}"
    if col in df:
        return _num(df, col)
    parts = [f"B3_{letter}_{p}" for p in ("u5_male", "u5_female", "5plus_male", "5plus_female")]
    have = [c for c in parts if c in df]
    return sum(_num(df, c).fillna(0) for c in have) if have else pd.Series(np.nan, index=df.index)


def hmis_from_kobo(sheets: dict[str, pd.DataFrame], fac: pd.DataFrame) -> pd.DataFrame:
    """Facility-month HMIS from the B_monthly repeat, aggregated to area-month."""
    rep = next((v for k, v in sheets.items() if k != "main" and "B1_month" in v.columns), None)
    if rep is None:
        return pd.DataFrame()
    rep = _codes(rep, "facility")
    keys = fac[["_index", "area_id", "site"]].astype({"_index": str})
    rep = rep.merge(keys, left_on=rep["_parent_index"].astype(str), right_on="_index", how="left")
    out = pd.DataFrame({"area_id": rep["area_id"], "site": rep["site"],
                        "month": pd.to_datetime(rep["B1_month"], errors="coerce").dt.to_period("M").dt.to_timestamp(),
                        "population": _num(rep, "B2_population"), "opd_total": _b3(rep, "a")})
    out["opd_female"] = _b3(rep, "a", "u5_female").fillna(0) + _b3(rep, "a", "5plus_female").fillna(0)
    for col, letters in OUTCOME_SOURCES.items():
        out[col] = sum(_b3(rep, l).fillna(0) for l in letters)
        out.loc[pd.concat([_b3(rep, l) for l in letters], axis=1).isna().all(axis=1), col] = np.nan
    out["complete"] = _is(rep, "facility", "B1_complete", "Yes").astype(float)
    return _area_month(out.dropna(subset=["area_id", "month"]))


def _area_month(fm: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in fm.columns if c not in ("area_id", "site", "month")]
    agg = fm.groupby(["area_id", "site", "month"])[num].sum(min_count=1).reset_index()
    if "opd_female" in agg:
        agg["opd_female_share"] = (agg["opd_female"] / agg["opd_total"].replace(0, np.nan)).clip(0, 1)
        agg = agg.drop(columns="opd_female")
    return agg


DHIS2_KEYWORDS = {  # data element name pattern -> warehouse column
    "malaria": r"malaria.*(confirm|positive|tested positive)|confirmed malaria",
    "diarrhoea": r"diarr|dysentery",
    "respiratory": r"pneumonia|respiratory|urti|\bari\b|asthma",
    "malnutrition": r"malnutrition|\bsam\b|\bmam\b|rutf|wasting",
    "anc": r"\banc\b|antenatal",
    "opd_total": r"outpatient|\bopd\b|attendance",
    "cholera": r"cholera",
}


def dhis2_default_mapping(elements: list[str]) -> dict[str, str]:
    out = {}
    for e in elements:
        for col, pat in DHIS2_KEYWORDS.items():
            if re.search(pat, e, re.I):
                out[e] = col
                break
    return out


def read_dhis2(file) -> pd.DataFrame:
    """Long-format DHIS2 export -> columns element, period, orgunit, value."""
    df = pd.read_csv(file, dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    e, p = pick("data", "data element", "dataelement", "dx"), pick("period", "pe")
    o, v = pick("organisation unit", "organisation unit name", "orgunit", "ou", "org unit"), pick("value")
    if not all([e, p, o, v]):
        raise ValueError("Expected columns: Data, Period, Organisation unit, Value (DHIS2 plain-table export).")
    parent = pick("organisation unit parent", "parent", "district", "county", "organisation unit parent name")
    out = pd.DataFrame({"element": df[e], "orgunit": df[o], "value": pd.to_numeric(df[v], errors="coerce"),
                        "month": df[p].map(_period),
                        "parent": df[parent] if parent else ""})
    return out.dropna(subset=["month"])


def _period(p) -> pd.Timestamp | None:
    s = str(p).strip()
    for fmt in ("%Y%m", "%Y-%m", "%B %Y", "%b %Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt).to_period("M").to_timestamp()
        except Exception:
            continue
    return None


def hmis_from_dhis2(long: pd.DataFrame, mapping: dict[str, str], fac: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map DHIS2 org units to areas by sub-county / TA name (with the parent district when the export
    has it, because some names repeat, e.g. 'TA Ngabu' in Chikwawa and Nsanje) or by facility name."""
    from .geo import norm_name
    by_site, by_name = {}, {}
    for site, m in SITES.items():
        for code, label in zip(m.get("sub_unit_codes", []), m["sub_units"]):
            aid = f"{site[:3].upper()}-{code:02d}"
            by_site[(norm_name(site), norm_name(label))] = aid
            by_name.setdefault(norm_name(label), set()).add(aid)
    if fac is not None and len(fac):
        for r in fac.dropna(subset=["area_id"]).to_dict("records"):
            for key in (r.get("facility_name"), r.get("facility_code")):
                if isinstance(key, str) and key:
                    by_name.setdefault(norm_name(key), set()).add(r["area_id"])
    site_of = {f"{s[:3].upper()}": s for s in SITES}

    def match(org, parent):
        k = norm_name(org)
        if parent and (norm_name(parent), k) in by_site:
            return by_site[(norm_name(parent), k)]
        hits = by_name.get(k, set())
        return next(iter(hits)) if len(hits) == 1 else np.nan

    d = long.copy()
    d["col"] = d["element"].map(mapping)
    d = d.dropna(subset=["col"])
    pairs = d[["orgunit", "parent"]].drop_duplicates()
    pairs["area_id"] = [match(o, p) for o, p in zip(pairs["orgunit"], pairs["parent"].fillna(""))]
    d = d.merge(pairs, on=["orgunit", "parent"], how="left")
    unmatched = sorted(d.loc[d["area_id"].isna(), "orgunit"].astype(str).unique())
    d = d.dropna(subset=["area_id"])
    d["site"] = d["area_id"].str[:3].map(site_of)
    wide = d.pivot_table(index=["area_id", "site", "month"], columns="col", values="value", aggfunc="sum").reset_index()
    wide.columns.name = None
    wide.attrs["unmatched_orgunits"] = unmatched
    return wide


# ------------------------------------------------------------------ climate from NASA POWER
def climate_for_areas(centroids: pd.DataFrame, start_year: int = 2001):
    """centroids: area_id, site, lat, lon -> (area-month climate, site-month climate history)."""
    from . import climate as C
    pts = list(zip(centroids["lat"], centroids["lon"]))
    monthly = C.fetch_many(pts, C.power_monthly, start=start_year)
    frames = []
    for (_, r), m in zip(centroids.iterrows(), monthly):
        if m is None:
            continue
        mi = C.monthly_indices(m, r["lat"])
        thr = mi["tmax"].quantile(0.9)
        frames.append(pd.DataFrame({
            "area_id": r["area_id"], "site": r["site"], "month": mi["month"], "rainfall_mm": mi["rain_mm"],
            "temp_c": mi["tmax"],
            # monthly data: heat days approximated from the share of the month above the 90th percentile
            "heat_days": ((mi["tmax"] - thr).clip(lower=0) * 6).round().clip(upper=31),
            "flood_event": ((mi["rain_pct"] >= 0.95) & (mi["rain_mm"] >= 50)).astype(int),
            "drought_event": (mi["spei_3"] <= -1.3).astype(int)}))
    clim = pd.concat(frames, ignore_index=True)
    hist = clim.groupby(["site", "month"], as_index=False)[["rainfall_mm", "temp_c"]].mean()
    return clim, hist


def area_centroids(*frames: pd.DataFrame) -> pd.DataFrame:
    """Mean GPS per area from household / facility data; site centre where none."""
    rows = []
    for site, m in SITES.items():
        for code, label in zip(m.get("sub_unit_codes", []), m["sub_units"]):
            aid = f"{site[:3].upper()}-{code:02d}"
            lat = lon = np.nan
            for f in frames:
                if f is not None and len(f) and aid in set(f["area_id"]):
                    r = f[f["area_id"] == aid][["lat", "lon"]].mean()
                    if r.notna().all():
                        lat, lon = r["lat"], r["lon"]
                        break
            if np.isnan(lat):
                lat, lon = SITE_COORDS[site]
            rows.append({"area_id": aid, "site": site, "sub_unit": label, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ assembly
def build_real_warehouse(hmis: pd.DataFrame, hh_area: pd.DataFrame | None, fac_area: pd.DataFrame | None,
                         fetch_climate: bool = True):
    """Join everything on area_id x month. Returns (warehouse, climate_history, report)."""
    from .data import DATA_DICTIONARY
    report = []
    cents = area_centroids(fac_area, hh_area)
    areas = cents[cents["area_id"].isin(hmis["area_id"].unique())] if len(hmis) else cents
    wh = hmis.merge(areas[["area_id", "sub_unit"]], on="area_id", how="left")
    wh["country"] = wh["site"].map(lambda s: SITES.get(s, {}).get("country", ""))

    for src, cols, label in [(hh_area, ["vulnerability_index"] + [f"share_{i}" for i in range(5)], "household survey"),
                             (fac_area, ["readiness_score", "surge_capacity", "referral_functional", "stockout_rate"],
                              "facility survey")]:
        if src is not None and len(src):
            wh = wh.merge(src[["area_id"] + cols], on="area_id", how="left", suffixes=("", "_s"))
            for c in cols:
                if f"{c}_s" in wh:  # monthly value (e.g. DHIS2 stock-outs) wins over the survey value
                    wh[c] = wh[c].fillna(wh.pop(f"{c}_s"))
                site_mean = wh.groupby("site")[c].transform("mean")
                wh[c] = wh[c].fillna(site_mean)
            report.append(f"Joined {label}: {src['area_id'].nunique()} areas.")
        else:
            report.append(f"No {label}: neutral defaults used for {', '.join(cols[:2])}...")

    hist = None
    if fetch_climate:
        clim, hist = climate_for_areas(areas)
        wh = wh.merge(clim.drop(columns="site"), on=["area_id", "month"], how="left")
        report.append(f"Climate from NASA POWER for {clim['area_id'].nunique()} areas "
                      f"(flood/drought events derived from rainfall and SPEI-3, not declared events).")
    for c in DATA_DICTIONARY["variable"]:
        if c not in wh:
            wh[c] = np.nan
    if "population" in wh:
        wh["population"] = wh.groupby("area_id")["population"].transform(lambda s: s.ffill().bfill())
    extra = [c for c in EXTRA_OUTCOMES if c in wh]
    wh = wh[DATA_DICTIONARY["variable"].tolist() + extra].sort_values(["area_id", "month"]).reset_index(drop=True)
    n_out = [k for k in OUTCOMES if wh[k].notna().sum() > 0]
    report.append(f"Warehouse: {wh['area_id'].nunique()} areas x {wh['month'].nunique()} months; outcomes with data: "
                  f"{', '.join(n_out) or 'none'}.")
    return wh, hist, report
