# Codebook: field data to model

How the household and facility Kobo forms feed the climate-health early warning model.
The full variable list (664 questions, with choice codes and skip logic) is in
[`codebook.csv`](codebook.csv); this page explains the joins and formulas. Code:
`src/codebook.py` (roles), `src/importers.py` (formulas).

Regenerate after a form changes in KoboToolbox:

```bash
python tools/extract_kobo_forms.py
python -m src.codebook
```

## 1. Keys: how everything joins

| Key | Household form | Facility form | DHIS2 | Climate |
|---|---|---|---|---|
| Site | `A2` (1 Turkana, 2 Mombasa, 3 Kisumu, 4 Nsanje, 5 Chikwawa, 6 Balaka) | `A2_district` | org unit hierarchy | - |
| **Area** (`area_id`) | `A3_subcounty` code | `A3_subcounty_ta` code | org unit name matched to sub-county / TA or facility name | NASA POWER at area centroid |
| Month | interview date (cross-sectional) | `B1_month` (repeat `B_monthly`) | Period | month |

`area_id` = first three letters of the site + the Kobo sub-county / TA code, e.g. `TUR-09`
(Turkana Central), `CHI-21` (TA Ngabu, Chikwawa). The 56 codes are in
`docs/forms/household_subcounty_TA.csv`. Note that Chikwawa and Nsanje both have a "TA Ngabu"
(codes 21 and 41), so codes, not names, are the key.

Area centroid for climate: mean of facility GPS (`A6_gps`), else household GPS (`A4`), else the
site centre. Household GPS is used only for this, then dropped.

## 2. Model targets (facility form, Section B, monthly)

Totals per area-month are the sum over facilities.

| Warehouse column | B3 rows |
|---|---|
| `malaria` | k (confirmed) |
| `diarrhoea` | d + e (non-bloody + dysentery) |
| `respiratory` | n + o + p + q (URTI, pneumonia, severe pneumonia, asthma) |
| `malnutrition` | w + x (MAM + SAM admissions) |
| `opd_total`, `opd_female_share` | a total; (a under-5 female + a 5+ female) / a total |
| `population` | B2 total catchment |
| extra: `cholera`, `admissions`, `deaths`, `injuries`, `heat_illness`, `skin_eye`, `snakebite` | f, b, c, t, v, r + s, u |
| `anc` | **not in the form**: from DHIS2 until B3 rows are added |

`B1_complete` (report complete and on time) is kept for weighting.

## 3. Household vulnerability index (Section A, D, G, H)

Per household, each component scaled 0 to 1 (1 = more vulnerable); index = mean of available
components; area value = mean over households.

| Component | Questions | Scaling |
|---|---|---|
| Female-headed | A10 = Female | 0/1 |
| Head aged 60+ | A10_age | 0/1 |
| Head has no education | A12 = None | 0/1 |
| Disability | any A14 item "A lot" / "Cannot do", or A18 = Yes | 0/1 |
| Dependency ratio | (under 18 + 60+) / (18 to 59) from A16 | ratio / 4, capped |
| Food insecurity (FIES) | count of Yes in D1a to D1h | / 8 |
| Low diet diversity (HDDS) | count of Yes in D3a to D3l | 1 − HDDS / 12 |
| Reduced coping (rCSI) | G3: 1·a + 2·b + 1·c + 3·d + 1·e (days, 0 to 7) | / 56 |
| Catastrophic health spending | G5 = Yes | 0/1 |
| No health insurance | H4 = No | 0/1 |
| Poverty | wealth quintile from H1 assets (first principal component) | (5 − quintile) / 4 |

Equal weights are a starting point; alternatives (PCA weights, expert weights) can be compared
during validation.

**Group shares for the equity view** (`share_0` to `share_4`): female-headed households, persons
under 5 (A16a / A15), persons 60+ (A16e / A15), households with disability, households in the
lowest wealth quintile.

**Relative risk by group**: share of households reporting any illness in the last 2 weeks
(C1, C5 or C8 = Yes) in the group, divided by the share in all other households. Replaces the
placeholder relative risks in the District view when a household export is loaded.

## 4. Facility readiness, surge capacity, referral, stock-outs (Sections D, E, F)

Per facility; area value = mean over facilities.

| Score | Domains (each 0 to 1, equal weight) |
|---|---|
| `readiness_score` | guidelines (D1, Observed = 1, Reported = 0.5); services offered (D1); diagnostics functioning (D2); tracer medicines available (E1, 14 items); staffing (D6 in post / approved, capped at 1); preparedness (D3 plan, D4 risk assessment, F13 triage and surge plan, F14 cholera area); infrastructure (F1 water today, F3 backup power, F4 cold chain in range, F6 raised building and cooling, F7 WASH functioning share, F8 phone or radio) |
| `surge_capacity` | extra beds / beds (F12); emergency transport (F8); surge plan (F13); cholera area (F14); staffing |
| `referral_functional` | 1 − delayed referrals / referrals (F10, F11); emergency transport; communication |
| `stockout_rate` | share of the 14 tracer items with a stock-out in the last 3 months (E1). Replaced by the monthly DHIS2 series when available |

## 5. Climate (NASA POWER until KMD / DCCMS data arrive)

Monthly rainfall, maximum and minimum temperature at each area centroid, 2001 onward.
Baseline 2001 to 2020. Derived: rainfall anomalies, SPEI-1/3 (Hargreaves PET, z-score),
temperature anomaly, heat days (approximate from monthly data), `flood_event` (rainfall at or
above the 95th percentile for that month and at least 50 mm), `drought_event` (SPEI-3 ≤ −1.3).
These are derived signals, not declared events: replace with KMD / DCCMS / disaster-agency
event chronologies when available.

## 6. Used for validation, not as model inputs

- Household B1 to B6 (exposure), C1 to C14 (illness, deaths), D2, D4 and D5 (nutrition):
  check model alerts against what households reported.
- Facility A9 (reference event), C4 to C7 (service use and disruption), D5 (early warning
  received), D8 (staff absence), E2 and E3 (supply disruption), F11 (referral delays): the
  stakeholder validation step.
- Household E (K10 mental health), F and G1 to G2 (costs), H3; facility C1 to C3: cost analysis
  (Objective 3).
- Facility G (county / district planning): how alerts should be delivered and used.

## 7. Personal data

Removed before data enter the warehouse: signatures, thumbprint photo, phone numbers,
interviewer and supervisor names, village, household number. Household GPS is averaged per
area and then dropped.

## 8. Gaps (see also the Data warehouse page)

1. No monthly ANC or delivery counts in the facility form (needed for the ANC / maternal outcome).
2. Stock-outs only for the last 3 months, not monthly.
3. Referrals are one-off totals, not monthly.
4. 60 months × ~115 numbers per facility by hand: use DHIS2 exports by default.
5. Household B1 months have no year.
6. Money amounts have no currency.
