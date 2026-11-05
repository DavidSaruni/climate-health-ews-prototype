import pandas as pd
import streamlit as st

from src import ui
from src.data import DATA_DICTIONARY, validate_upload

wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)

st.title("Data warehouse")
st.caption("Four streams joined on one area key (sub-county / district) and one time key (month).")
ui.prototype_banner()

st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontname="Helvetica", fontsize=11];
  met [label="1. Meteorological\\nKMD / DCMS"]; hmis [label="2. Routine health\\nHMIS / DHIS2"];
  hh [label="3. Household survey"]; fac [label="4. Facility assessments"];
  wh [label="Unified warehouse\\narea x month", fillcolor="#d7e6f5"];
  fe [label="Feature engineering"]; mo [label="Models\\nNB, QP, RF, XGBoost"];
  app [label="Decision-support app", fillcolor="#d9f0df"];
  met -> wh; hmis -> wh; hh -> wh; fac -> wh; wh -> fe -> mo -> app;
}""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Areas", wh["area_id"].nunique())
c2.metric("Months", wh["month"].nunique())
c3.metric("Rows", f"{len(wh):,}")
c4.metric("Period", f"{wh.month.min():%b %Y} to {wh.month.max():%b %Y}")

tab1, tab2, tab3, tab4 = st.tabs(["Data dictionary", "Browse", "Completeness", "Upload real data"])
with tab1:
    st.dataframe(DATA_DICTIONARY, hide_index=True, width="stretch", height=520)
    st.download_button("Download data dictionary (CSV)", DATA_DICTIONARY.to_csv(index=False),
                       "data_dictionary.csv")
with tab2:
    sites = st.multiselect("Sites", sorted(wh.site.unique()), default=sorted(wh.site.unique())[:1])
    view = wh[wh.site.isin(sites)]
    st.dataframe(view, hide_index=True, width="stretch", height=420)
    st.download_button("Download warehouse (CSV), also a template for real data",
                       wh.to_csv(index=False), "warehouse.csv")
with tab3:
    miss = wh.isna().mean().rename("missing share").to_frame()
    miss["stream"] = DATA_DICTIONARY.set_index("variable")["stream"]
    st.dataframe(miss.style.format({"missing share": "{:.1%}"}), width="stretch", height=520)
with tab4:
    st.markdown("Upload a CSV with the same columns as the template (see the data dictionary). "
                "Required: `area_id, site, sub_unit, month, population, rainfall_mm, temp_c, "
                "heat_days, flood_event, drought_event` and the five outcome columns. Other "
                "columns are filled with neutral defaults. The file stays in your browser "
                "session only and is not saved on the server.")
    f = st.file_uploader("Warehouse CSV", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        problems = validate_upload(df)
        if problems:
            for p in problems:
                st.error(p)
        else:
            st.session_state["uploaded_wh"] = df
            st.success(f"Loaded {len(df):,} rows. All pages now use this file.")
    if st.session_state.get("uploaded_wh") is not None and st.button("Switch back to synthetic data"):
        st.session_state["uploaded_wh"] = None
        st.rerun()
