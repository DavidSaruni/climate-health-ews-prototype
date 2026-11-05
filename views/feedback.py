"""Feedback for the co-design rounds.

If `github_token` and `github_repo` are set in Streamlit secrets, each submission is filed as
a GitHub issue (lasting record, works on Streamlit Community Cloud). Otherwise it is appended
to data/feedback.csv, which is lost when a hosted app restarts.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src import ui
from src.config import APP_VERSION, OUTCOMES, SITES

wh, feat, key, source = ui.get_data()
ui.sidebar_footer(source)
FILE = Path(__file__).resolve().parent.parent / "data" / "feedback.csv"


def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def file_issue(entry: dict) -> str | None:
    token, repo = _secret("github_token"), _secret("github_repo")
    if not (token and repo):
        return None
    body = "\n".join(f"**{k}:** {v}" for k, v in entry.items() if k != "comments")
    body += f"\n\n---\n{entry['comments']}"
    r = requests.post(f"https://api.github.com/repos/{repo}/issues",
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                      json={"title": f"[Feedback] {entry['page']}: {entry['comments'][:60]}",
                            "body": body, "labels": ["feedback", entry["round"]]}, timeout=15)
    r.raise_for_status()
    return r.json().get("html_url")


st.title("Give feedback")
st.markdown("This prototype is round 1 of three co-design rounds. Tell us what is useful, "
            "confusing or missing. Every comment is reviewed before the next version.")

with st.form("feedback", clear_on_submit=True):
    c1, c2 = st.columns(2)
    name = c1.text_input("Name (optional)")
    role = c2.selectbox("Role", ["County / district health manager", "Disease surveillance officer",
                                 "Health records / data manager", "Ministry of Health", "Researcher",
                                 "Amref / IDRC", "Other"])
    c3, c4 = st.columns(2)
    site = c3.selectbox("Site you work in", ["Not site-specific", *SITES])
    page = c4.selectbox("Which part is this about?", ["Overview", "District view", "Risk drivers",
                                                     "Model validation", "Data warehouse", "Methods",
                                                     "General"])
    c5, c6 = st.columns(2)
    useful = c5.slider("How useful would this be for planning? (1 = not, 5 = very)", 1, 5, 3)
    clear = c6.slider("How easy is it to understand? (1 = hard, 5 = easy)", 1, 5, 3)
    outcomes = st.multiselect("Which outcomes matter most to you?", [v["label"] for v in OUTCOMES.values()])
    comments = st.text_area("Comments, suggestions, or events where the alert did or did not "
                            "match what you experienced", height=150)
    submitted = st.form_submit_button("Submit feedback", type="primary")

if submitted:
    if not comments.strip():
        st.error("Please add a comment.")
    else:
        entry = {"time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "round": "round-1",
                 "version": APP_VERSION, "name": name, "role": role, "site": site, "page": page,
                 "useful": useful, "clear": clear, "priority outcomes": "; ".join(outcomes),
                 "comments": comments.strip()}
        url = None
        try:
            url = file_issue(entry)
        except Exception as e:
            st.warning(f"Could not file to GitHub ({e}); saved locally instead.")
        FILE.parent.mkdir(exist_ok=True)
        pd.DataFrame([entry]).to_csv(FILE, mode="a", header=not FILE.exists(), index=False)
        st.success("Thank you. Your feedback has been recorded." + (f" [View]({url})" if url else ""))

if FILE.exists():
    with st.expander("Feedback received on this server"):
        fb = pd.read_csv(FILE).drop(columns=["name"], errors="ignore")  # keep names private
        st.dataframe(fb, hide_index=True, width="stretch")
        st.download_button("Download feedback (CSV)", fb.to_csv(index=False), "feedback.csv")
