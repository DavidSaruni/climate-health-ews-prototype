"""Smoke test: every page runs without an exception. Run with:  python -m pytest tests"""
import pytest
from streamlit.testing.v1 import AppTest

import socket


def _online():
    try:
        socket.create_connection(("power.larc.nasa.gov", 443), timeout=5).close()
        return True
    except OSError:
        return False


PAGES = ["views/overview.py", "views/district.py", "views/drivers.py", "views/validation.py",
         "views/data.py", "views/methods.py", "views/feedback.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    at.switch_page(page).run()
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("model", ["XGBoost", "Random Forest", "Quasi-Poisson (GLM)", "Seasonal naive (baseline)"])
def test_district_and_drivers_each_model(model):
    for page in ["views/district.py", "views/drivers.py"]:
        at = AppTest.from_file("../app.py", default_timeout=300)
        at.session_state["model"] = model
        at.session_state["site"] = "Nsanje"
        at.session_state["outcome"] = "diarrhoea"
        at.run()
        at.switch_page(page).run()
        assert not at.exception, (page, [e.message for e in at.exception])


@pytest.mark.skipif(not _online(), reason="needs internet (NASA POWER, geoBoundaries)")
@pytest.mark.parametrize("layer", ["All hazards (most severe)", "drought", "malaria", "Model risk (project sites)"])
def test_studio(layer):
    at = AppTest.from_file("../app.py", default_timeout=600)
    at.run()
    at.switch_page("views/studio.py").run()
    assert not at.exception, [e.message for e in at.exception]
    at.sidebar.radio[0].set_value(layer).run()
    assert not at.exception, [e.message for e in at.exception]
