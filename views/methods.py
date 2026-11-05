import streamlit as st

from src import ui
from src.config import ALERT_THRESHOLDS, APP_VERSION, SURGE_FACTOR

wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)

st.title("Methods and roadmap")
st.caption(f"Version {APP_VERSION}. Kabarak University with EED Advisory, sub-award from Amref "
           "Health Africa in Kenya, funded by IDRC.")

st.markdown(f"""
### The seven stages in this prototype

| Stage | Proposal | In this prototype | Still to do |
|---|---|---|---|
| 1. Data | 5.7.2 | Four simulated streams for 6 sites, 24 sub-units, Jan 2018 to latest month; 20-year climate record for baselines | Real KMD/DCMS and DHIS2 extracts (Month 14); survey and facility data (Months 15 to 17) |
| 2. Warehouse | 5.7.2 | One row per area per month; data dictionary; CSV upload and validation | DHIS2 and KMD/DCMS API connectors; geopandas boundaries |
| 3. Features | 5.7.3 | 30/90-day rainfall anomalies vs baseline; SPEI-1/3 (approximation); temperature anomaly; heat days; flood and drought flags; lagged case rates; stock-outs; referral functionality; readiness; vulnerability index; seasonal outlook | Full SPEI (log-logistic fit, Hargreaves PET); site-specific heat thresholds agreed with KMD/DCMS |
| 4. Models | 5.7.4 | Negative binomial and quasi-Poisson GLMs; Random Forest; XGBoost; seasonal-naive benchmark; SHAP via XGBoost TreeSHAP | LSTM for temporal sequences once real series are available; per outcome-site GLMs |
| 5. Validation | 5.7.5 | Temporal hold-out (24 months), leave-one-site-out, alert performance, calibration, bootstrap CIs | Structured stakeholder validation with DHMT and MoH staff |
| 6. Application | 5.7.6 | Risk scores, alert levels, equity indicators, sex-disaggregated projections, actions | Three co-design rounds; low-bandwidth and offline mode; DHIS2 dashboard integration |
| 7. Deployment | 5.7.7 | Open-source code; hosted web prototype | Deployment in six sites; training; handover to MoH and DHMTs |

### How a forecast is made
1. At the latest month, the model takes each area's recent climate, health and health-system
   indicators and predicts the expected number of cases one, two and three months ahead.
2. A negative binomial distribution around that expectation gives the 80% and 95% ranges.
3. **Surge threshold:** {SURGE_FACTOR:.2f} x the area's median count for the same calendar month.
4. **Risk score** = probability (0 to 100) that cases exceed the threshold.
5. **Alert:** green below {ALERT_THRESHOLDS['amber']:.0%}, amber from {ALERT_THRESHOLDS['amber']:.0%},
   red from {ALERT_THRESHOLDS['red']:.0%}. These cut-offs are placeholders, to be calibrated against
   historical surge events and agreed with district teams.
6. County / district totals are the sum of sub-unit simulations, so their ranges are honest.

### Known limitations of this prototype
- **All data are simulated.** Patterns are built in on purpose, so the models will look better
  than they will on real data.
- The seasonal rainfall outlook is simulated as the true future anomaly plus noise (correlation about 0.7).
- Prediction intervals do not yet include uncertainty in the model parameters.
- Vulnerability-group relative risks and preparedness actions are placeholders.
""")

st.markdown("""
### Questions for the team and stakeholders
1. Which outcomes matter most to each district? Are any missing (e.g. cholera, dengue, heat illness)?
2. What should a "surge" mean in practice: a fixed percentage above normal, or a capacity limit?
3. Which lead time is most useful for action: one, two or three months?
4. Who will see the alerts, and through which channel (DHIS2 dashboard, SMS, WhatsApp, email)?
5. Which preparedness actions are realistic at each alert level?
""")
