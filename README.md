# Climate-Health Early Warning Prototype

Prototype of the climate-health predictive model and decision-support tool (Objective 5) for
*Quantifying the Hidden Health Costs of Climate Change for Equitable Health System Adaptation
in Eastern and Southern Africa*. Kabarak University with EED Advisory Limited; sub-award from
Amref Health Africa in Kenya; funded by IDRC.

> **All data in this prototype are simulated.** Thresholds, relative risks and preparedness
> actions are placeholders for co-design with DHMT/CHMT and Ministry of Health users.

## What it does

| Page | Content | Proposal |
|---|---|---|
| Overview | Alert grid and map for all six sites, next 1 to 3 months | 5.7.6 |
| District view | Risk score, alert level, trend + forecast, equity indicators (sub-units, vulnerability groups, sex-disaggregated), recommended actions | 5.7.6 |
| Risk drivers | SHAP (XGBoost), incidence rate ratios (GLM), feature importance (RF) | 5.7.4 |
| Geospatial Studio | Any country: satellite basemap (Esri, NASA daily imagery), counties / districts coloured by flood, drought, heat, malaria suitability, model risk or uploaded disease data; NASA flood, rainfall, temperature, NDVI and soil-moisture overlays | 5.7.6 |
| Model validation | 24-month temporal hold-out, leave-one-site-out, calibration, alert performance, stakeholder event review | 5.7.5 |
| Data warehouse | Import Kobo household / facility exports and DHIS2; codebook (664 questions mapped to the model); data dictionary; completeness | 5.7.2 |
| Methods and roadmap | What is built, what is still to do, open questions | all |
| Give feedback | Co-design feedback form (files GitHub issues if configured) | 5.7.6 |

Sites: Turkana, Mombasa, Kisumu (Kenya); Nsanje, Chikwawa, Balaka (Malawi): 56 sub-counties / TAs,
using the same codes as the Kobo forms.
Outcomes: malaria-like febrile illness, diarrhoeal disease, respiratory illness, malnutrition
admissions, ANC and maternal service demand. Models: negative binomial and quasi-Poisson GLMs,
Random Forest, XGBoost, seasonal-naive benchmark (LSTM planned for real data).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` then `python -m pytest tests` (Studio tests need internet)

## Project layout

```
app.py               entry point and navigation
src/config.py        sites, outcomes, thresholds, actions  <- most feedback changes go here
src/data.py          synthetic warehouse + data dictionary + upload validation
src/features.py      feature engineering (anomalies, SPEI, lags, outlook)
src/models.py        NB, quasi-Poisson, RF, XGBoost, seasonal naive
src/pipeline.py      forecasting, temporal and spatial validation, alert metrics
src/codebook.py      every Kobo question tagged with its role in the model
src/importers.py     Kobo / DHIS2 exports -> indices -> warehouse (+ NASA POWER climate)
src/climate.py       NASA POWER client; SPEI, rainfall percentile, heat, malaria suitability
src/geo.py           boundaries for any country (geoBoundaries, OCHA COD-AB fallback), imagery layers
src/samples.py       sample Kobo / DHIS2 exports (templates, tests)
docs/CODEBOOK.md     joins and formulas; docs/codebook.csv full variable list
tools/               re-extract the Kobo form definitions
src/ui.py            caching and shared widgets
views/               one file per page
tests/               smoke tests for every page and model
```

## Deploy on Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`. Under **Advanced settings** choose
   Python 3.12.
4. Optional, to keep feedback permanently as GitHub issues: in the app's **Settings > Secrets** add
   ```toml
   github_token = "<fine-grained token with Issues: read and write on this repo>"
   github_repo = "<owner>/<repo>"
   ```
5. Every push to `main` redeploys the app automatically.

The first visit after a restart takes about 20 to 40 seconds while models train; after that
results are cached for all users.

## Loading real data

**Data warehouse > Import field data**: upload the household and facility Kobo exports (XLSX,
"XML values and headers", all sheets) and optionally a DHIS2 plain-table CSV. The app computes
the vulnerability and readiness indices, joins the monthly HMIS series, fetches NASA POWER
climate for each sub-county / TA, and switches every page to the real warehouse. Sample files in
the same layout can be downloaded from the same tab. Formulas: [docs/CODEBOOK.md](docs/CODEBOOK.md).

If the Kobo forms change: `python tools/extract_kobo_forms.py` then `python -m src.codebook`.

## Data sources (all open)

NASA POWER (climate), NASA GIBS (daily imagery, flood maps, IMERG, MODIS LST, NDVI, SMAP),
Esri World Imagery (basemap), geoBoundaries (boundaries, CC BY 4.0), OCHA COD-AB via HDX
(fallback boundaries, CC BY-IGO; Malawi districts are bundled in `data/boundaries`).

## Iteration log

| Version | Date | Changes |
|---|---|---|
| 0.1.0 | 2026-11-05 | First prototype for round 1 of co-design |
| 0.2.0 | 2026-11-26 | Geospatial Studio; Kobo / DHIS2 importers; codebook; real sub-county / TA codes; NASA POWER climate |
