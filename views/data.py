import pandas as pd
import streamlit as st

from src import codebook, importers, samples, ui
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

tab0, tabc, tab1, tab2, tab3, tab4 = st.tabs(["Import field data (Kobo, DHIS2)", "Codebook", "Data dictionary",
                                              "Browse", "Completeness", "Upload warehouse CSV"])

# ------------------------------------------------------------------ import Kobo / DHIS2
with tab0:
    st.markdown("Build the warehouse from the **household** and **facility** Kobo exports (XLSX with all "
                "sheets, so the monthly HMIS repeat is included) and, optionally, a **DHIS2** export. "
                "Climate is added from NASA POWER for each sub-county / TA. Files stay in your session only.")
    with st.expander("Download sample files (simulated, same layout as the real exports)"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Prepare household sample"):
            st.session_state["sample_hh"] = samples.household_export()
        if "sample_hh" in st.session_state:
            c1.download_button("household_sample.xlsx", st.session_state["sample_hh"], "household_sample.xlsx")
        if c2.button("Prepare facility sample"):
            st.session_state["sample_fac"] = samples.facility_export()
        if "sample_fac" in st.session_state:
            c2.download_button("facility_sample.xlsx", st.session_state["sample_fac"], "facility_sample.xlsx")
        c3.download_button("dhis2_sample.csv", samples.dhis2_export(), "dhis2_sample.csv")
        st.caption("In KoboToolbox: Data > Downloads > XLS, 'XML values and headers', include groups in headers.")

    u1, u2, u3 = st.columns(3)
    f_hh = u1.file_uploader("Household survey export", type=["xlsx", "csv"], key="f_hh")
    f_fac = u2.file_uploader("Facility survey export", type=["xlsx", "csv"], key="f_fac")
    f_dh = u3.file_uploader("DHIS2 export (optional)", type=["csv"], key="f_dh")

    hh = hh_area = rr = fac = fac_area = hmis = None
    if f_hh is not None:
        sheets = importers.read_kobo(f_hh, f_hh.name)
        hh = importers.household_scores(sheets["main"])
        hh_area, rr = importers.household_area_summary(hh)
        st.success(f"Household survey: {len(hh):,} households, {hh['area_id'].notna().mean():.0%} matched to an "
                   f"area, {hh_area['area_id'].nunique()} areas.")
    if f_fac is not None:
        fsheets = importers.read_kobo(f_fac, f_fac.name)
        fac = importers.facility_scores(fsheets["main"])
        fac_area = importers.facility_area_summary(fac)
        hmis = importers.hmis_from_kobo(fsheets, fac)
        st.success(f"Facility survey: {len(fac):,} facilities in {fac_area['area_id'].nunique()} areas; monthly HMIS "
                   f"rows: {len(hmis):,}.")
    if f_dh is not None:
        long = importers.read_dhis2(f_dh)
        elements = sorted(long["element"].unique())
        default = importers.dhis2_default_mapping(elements)
        targets = ["(ignore)", "malaria", "diarrhoea", "respiratory", "malnutrition", "anc", "opd_total",
                   "cholera"]
        st.markdown("**Map DHIS2 data elements to model outcomes**")
        mp = st.data_editor(pd.DataFrame({"DHIS2 data element": elements,
                                          "maps to": [default.get(e, "(ignore)") for e in elements]}),
                            column_config={"maps to": st.column_config.SelectboxColumn(options=targets)},
                            hide_index=True, width="stretch", key="dhis2_map")
        mapping = {r["DHIS2 data element"]: r["maps to"] for _, r in mp.iterrows() if r["maps to"] != "(ignore)"}
        d2 = importers.hmis_from_dhis2(long, mapping, fac)
        if d2.attrs.get("unmatched_orgunits"):
            st.warning("Org units not matched to a sub-county / TA (unknown, or a name used in two districts "
                       "without a parent column): " + ", ".join(d2.attrs["unmatched_orgunits"][:15]))
        st.success(f"DHIS2: {len(d2):,} area-months, columns: {', '.join(c for c in d2.columns if c not in ('area_id', 'site', 'month'))}.")
        if hmis is None or hmis.empty:
            hmis = d2
        else:  # DHIS2 values fill or extend the facility-form series
            hmis = hmis.set_index(["area_id", "site", "month"]).combine_first(
                d2.set_index(["area_id", "site", "month"])).reset_index()

    if rr is not None:
        with st.expander("Household survey: relative risk of illness by vulnerability group"):
            st.dataframe(rr.round(2), hide_index=True, width="stretch")
    if fac_area is not None:
        with st.expander("Facility survey: readiness by area"):
            st.dataframe(fac_area.round(2), hide_index=True, width="stretch")

    if hmis is not None and len(hmis):
        if st.button("Build warehouse and use it in all pages", type="primary"):
            with st.spinner("Joining streams and fetching NASA POWER climate for each area..."):
                new_wh, hist, report = importers.build_real_warehouse(hmis, hh_area, fac_area)
            st.session_state["uploaded_wh"] = new_wh
            st.session_state["uploaded_clim"] = hist
            st.session_state["uploaded_label"] = "Field data (Kobo/DHIS2) + NASA POWER"
            st.session_state["group_rr"] = rr
            for line in report:
                st.write("- " + line)
            st.success("Done. Overview, District view and Validation now use this warehouse.")
    elif f_hh is not None or f_fac is not None:
        st.info("A monthly health series is needed: the facility export with its B_monthly sheet, or a DHIS2 export.")

# ------------------------------------------------------------------ codebook
with tabc:
    cb = codebook.build_codebook()
    st.markdown(f"**{len(cb)} questions** from both Kobo forms, tagged with their role in the model. "
                "Full formulas: docs/CODEBOOK.md in the repository.")
    f1, f2, f3 = st.columns(3)
    form = f1.multiselect("Form", ["household", "facility"], default=["household", "facility"])
    use = f2.multiselect("Model use", sorted(cb["model_use"].unique()))
    q = f3.text_input("Search name or label")
    view = cb[cb["form"].isin(form)]
    if use:
        view = view[view["model_use"].isin(use)]
    if q:
        view = view[view["name"].str.contains(q, case=False) | view["label"].str.contains(q, case=False)]
    st.dataframe(view[["form", "name", "label", "type", "model_use", "feeds", "transform", "choices"]],
                 hide_index=True, width="stretch", height=420)
    st.download_button("Download codebook (CSV)", cb.to_csv(index=False), "codebook.csv")
    st.markdown("**Gaps between the forms and what the model needs**")
    st.dataframe(codebook.GAPS, hide_index=True, width="stretch")
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
