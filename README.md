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
| Model validation | 24-month temporal hold-out, leave-one-site-out, calibration, alert performance, stakeholder event review | 5.7.5 |
| Data warehouse | Data dictionary, browse, completeness, CSV upload of real data | 5.7.2 |
| Methods and roadmap | What is built, what is still to do, open questions | all |
| Give feedback | Co-design feedback form (files GitHub issues if configured) | 5.7.6 |

Sites: Turkana, Mombasa, Kisumu (Kenya); Nsanje, Chikwawa, Balaka (Malawi), 4 sub-units each.
Outcomes: malaria-like febrile illness, diarrhoeal disease, respiratory illness, malnutrition
admissions, ANC and maternal service demand. Models: negative binomial and quasi-Poisson GLMs,
Random Forest, XGBoost, seasonal-naive benchmark (LSTM planned for real data).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `python -m pytest tests`

## Project layout

```
app.py               entry point and navigation
src/config.py        sites, outcomes, thresholds, actions  <- most feedback changes go here
src/data.py          synthetic warehouse + data dictionary + upload validation
src/features.py      feature engineering (anomalies, SPEI, lags, outlook)
src/models.py        NB, quasi-Poisson, RF, XGBoost, seasonal naive
src/pipeline.py      forecasting, temporal and spatial validation, alert metrics
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

## Swapping in real data

Download the warehouse CSV from the **Data warehouse** page as a template, fill it with real
extracts (same column names, one row per area per month), and upload it. For a permanent switch,
replace `build_warehouse()` in `src/data.py` with a loader for the real warehouse. Site names
must match `SITES` in `src/config.py`.

## Iteration log

| Version | Date | Changes |
|---|---|---|
| 0.1.0 | 2026-11-05 | First prototype for round 1 of co-design |
