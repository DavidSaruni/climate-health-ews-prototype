"""Entry point. Run locally with:  streamlit run app.py"""
import streamlit as st

from src.config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, page_icon="🌦️", layout="wide")

pages = {
    "Decision support": [
        st.Page("views/overview.py", title="Overview", icon="🗺️", default=True),
        st.Page("views/district.py", title="District view", icon="🚦"),
        st.Page("views/drivers.py", title="Risk drivers", icon="🔍"),
    ],
    "Model": [
        st.Page("views/validation.py", title="Model validation", icon="📏"),
        st.Page("views/data.py", title="Data warehouse", icon="🗄️"),
        st.Page("views/methods.py", title="Methods and roadmap", icon="📘"),
    ],
    "Co-design": [
        st.Page("views/feedback.py", title="Give feedback", icon="💬"),
    ],
}
st.navigation(pages).run()
