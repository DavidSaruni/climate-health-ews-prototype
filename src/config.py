"""Project-wide settings: study sites, outcomes, alert thresholds and preparedness actions.

Everything a stakeholder might want to change after feedback lives here, so it can be
edited without touching the modelling code.
"""

APP_TITLE = "Climate-Health Early Warning Prototype"
APP_VERSION = "0.1.0 (prototype, round 1)"

# Study sites (proposal section 5.2). Sub-units are used for the equity view.
SITES = {
    "Turkana": {
        "country": "Kenya", "level": "County",
        "profile": "Arid; drought and extreme heat; pastoralist livelihoods",
        "sub_units": ["Turkana Central", "Turkana North", "Turkana South", "Loima"],
        "climate": {"rain_mean": 30, "rain_season": [4, 5, 11], "temp_mean": 30.5, "temp_amp": 2.0,
                    "flood_prone": 0.2, "drought_prone": 0.9},
        "vulnerability": 0.82,
    },
    "Mombasa": {
        "country": "Kenya", "level": "County",
        "profile": "Coastal urban; flooding and sea-level rise; informal settlements",
        "sub_units": ["Kisauni", "Likoni", "Changamwe", "Mvita"],
        "climate": {"rain_mean": 95, "rain_season": [4, 5, 10, 11], "temp_mean": 27.5, "temp_amp": 1.8,
                    "flood_prone": 0.8, "drought_prone": 0.2},
        "vulnerability": 0.55,
    },
    "Kisumu": {
        "country": "Kenya", "level": "County",
        "profile": "Lake Victoria basin; recurrent flooding; high malaria burden",
        "sub_units": ["Kisumu East", "Nyando", "Nyakach", "Muhoroni"],
        "climate": {"rain_mean": 115, "rain_season": [3, 4, 5, 10, 11], "temp_mean": 23.5, "temp_amp": 1.5,
                    "flood_prone": 0.85, "drought_prone": 0.2},
        "vulnerability": 0.60,
    },
    "Nsanje": {
        "country": "Malawi", "level": "District",
        "profile": "Lower Shire valley; riverine flooding; displacement",
        "sub_units": ["Nsanje Boma", "Bangula", "Makhanga", "Trinity"],
        "climate": {"rain_mean": 70, "rain_season": [12, 1, 2, 3], "temp_mean": 27.0, "temp_amp": 3.5,
                    "flood_prone": 0.9, "drought_prone": 0.4},
        "vulnerability": 0.78,
    },
    "Chikwawa": {
        "country": "Malawi", "level": "District",
        "profile": "Flood and heat stress; limited referral capacity",
        "sub_units": ["Chikwawa Boma", "Ngabu", "Nchalo", "Mapelera"],
        "climate": {"rain_mean": 65, "rain_season": [12, 1, 2, 3], "temp_mean": 27.5, "temp_amp": 3.8,
                    "flood_prone": 0.75, "drought_prone": 0.5},
        "vulnerability": 0.74,
    },
    "Balaka": {
        "country": "Malawi", "level": "District",
        "profile": "Drought; food insecurity; high outmigration",
        "sub_units": ["Balaka Boma", "Phalula", "Ulongwe", "Kwitanda"],
        "climate": {"rain_mean": 60, "rain_season": [12, 1, 2, 3], "temp_mean": 24.5, "temp_amp": 3.5,
                    "flood_prone": 0.3, "drought_prone": 0.85},
        "vulnerability": 0.70,
    },
}

# Climate-sensitive outcomes modelled (proposal sections 5.1 and 5.4.3).
OUTCOMES = {
    "malaria": {"label": "Malaria-like febrile illness", "base": 900},
    "diarrhoea": {"label": "Diarrhoeal disease", "base": 420},
    "respiratory": {"label": "Respiratory illness", "base": 600},
    "malnutrition": {"label": "Malnutrition admissions", "base": 60},
    "anc": {"label": "ANC and maternal service demand", "base": 350},
}

HORIZONS = [1, 2, 3]  # months ahead

# Validation design (proposal section 5.7.5)
TEMPORAL_HOLDOUT_MONTHS = 24

# A "surge" is a month where cases exceed the area's same-calendar-month historical
# median by this factor. Alerts are calibrated against these historical surges.
SURGE_FACTOR = 1.25

# Alert levels from the probability of a surge. Placeholder cut-offs: to be agreed with DHMTs.
ALERT_THRESHOLDS = {"amber": 0.35, "red": 0.65}
ALERT_COLOURS = {"Green": "#2e9e5b", "Amber": "#e8a317", "Red": "#d64545"}

# Placeholder preparedness actions. To be replaced with actions from the qualitative
# analysis of effective response practices (proposal section 5.7.6).
ACTIONS = {
    "Green": [
        "Continue routine surveillance and weekly DHIS2 reporting.",
        "Review stock levels of tracer medicines against seasonal needs.",
        "Share the monthly climate outlook with facility in-charges.",
    ],
    "Amber": [
        "Alert sub-county disease surveillance officers and facility in-charges.",
        "Pre-position tracer commodities (ACTs, RDTs, ORS/zinc, RUTF) at high-risk facilities.",
        "Activate community health promoters for risk communication.",
        "Confirm referral transport and fuel availability.",
    ],
    "Red": [
        "Convene the DHMT/CHMT emergency coordination meeting within 48 hours.",
        "Activate surge staffing and extended outreach in the highest-risk sub-units.",
        "Request emergency commodity resupply from KEMSA / CMST.",
        "Set up mobile clinics for displaced or cut-off populations.",
        "Protect continuity of ANC and maternal services (outreach, waiting homes, referral).",
    ],
}

OUTCOME_ACTIONS = {
    "malaria": "Intensify vector control (ITN distribution, IRS) and test-and-treat at community level.",
    "diarrhoea": "Chlorinate water points, distribute water treatment tablets, prepare cholera treatment units.",
    "respiratory": "Heat and dust risk messaging; protect children under five and older people.",
    "malnutrition": "Scale up MUAC screening and outpatient therapeutic programmes.",
    "anc": "Plan for disrupted access to ANC: outreach clinics, maternity waiting homes, referral transport.",
}

VULNERABILITY_GROUPS = ["Female-headed households", "Children under five", "Older persons (60+)",
                        "Persons with disability", "Lowest wealth quintile"]

# Placeholder relative risk of each vulnerability group, by outcome. To be replaced with
# estimates from the household survey (proposal section 5.6).
GROUP_RELATIVE_RISK = {
    "malaria":      [1.2, 2.0, 1.1, 1.2, 1.5],
    "diarrhoea":    [1.3, 2.2, 1.3, 1.3, 1.7],
    "respiratory":  [1.1, 1.6, 1.8, 1.3, 1.3],
    "malnutrition": [1.4, 3.0, 1.2, 1.4, 2.0],
    "anc":          [1.3, 0.0, 0.0, 1.1, 1.2],
}

# Approximate site centroids for the overview map.
SITE_COORDS = {"Turkana": (3.12, 35.60), "Mombasa": (-4.04, 39.67), "Kisumu": (-0.09, 34.77),
               "Nsanje": (-16.92, 35.26), "Chikwawa": (-16.03, 34.80), "Balaka": (-14.98, 34.95)}
