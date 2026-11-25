"""Download the Kobo (Enketo) form definitions and save one row per question.

Usage:  python tools/extract_kobo_forms.py
Writes docs/forms/<form>.csv with: name, path, group, label, type, choices (code=label | ...),
relevant (skip logic), required. Re-run whenever the forms change in KoboToolbox.
"""
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

FORMS = {"household": "mWqIvFHu", "facility": "eAFwXcJ1"}
OUT = Path(__file__).resolve().parent.parent / "docs" / "forms"


def clean(t):
    return re.sub(r"\s+", " ", t or "").strip()


def choice_lists(model_xml: str) -> dict:
    """Choice lists live as secondary instances in the XForm model."""
    soup = BeautifulSoup(model_xml, "xml")
    out = {}
    for inst in soup.find_all("instance"):
        if not inst.get("id"):
            continue
        items = []
        for it in inst.find_all("item"):
            d = {c.name: clean(c.get_text()) for c in it.find_all(recursive=False)}
            items.append(d)
        out[inst["id"]] = items
    return out


def extract(form_id):
    r = requests.post(f"https://ee.kobotoolbox.org/transform/xform/{form_id}", timeout=60,
                      headers={"User-Agent": "Mozilla/5.0", "Referer": f"https://ee.kobotoolbox.org/x/{form_id}"})
    r.raise_for_status()
    payload = r.json()
    soup = BeautifulSoup(payload["form"], "lxml")
    lists = choice_lists(payload["model"])
    rows, seen = [], set()
    for q in soup.select(".question, .note"):
        inputs = q.select("input[name], select[name], textarea[name], input[data-name]")
        if not inputs:
            continue
        el = inputs[0]
        path = el.get("name") or el.get("data-name")
        if path in seen:
            continue
        seen.add(path)
        parts = [p for p in path.split("/") if p]
        lab = q.select_one(".question-label")
        typ = el.get("data-type-xml") or el.get("type") or el.name
        if el.name == "select":
            typ = "select_multiple" if el.has_attr("multiple") else "select_one"
        elif el.get("type") == "radio":
            typ = "select_one"
        elif el.get("type") == "checkbox":
            typ = "select_multiple"
        choices = []
        for opt in q.select("label"):
            i = opt.select_one("input[value]")
            ol = opt.select_one(".option-label")
            if i is not None and ol is not None:
                choices.append(f"{i['value']}={clean(ol.get_text())}")
        if el.name == "select":
            choices = [f"{o.get('value')}={clean(o.get_text())}" for o in el.select("option") if o.get("value")]
        tmpl = q.select_one("[data-items-path]")
        list_name, filt = "", ""
        if tmpl is not None:
            m = re.match(r"instance\('([^']+)'\)/root/item(?:\[(.*)\])?", tmpl["data-items-path"])
            if m:
                list_name, filt = m.group(1), clean(m.group(2) or "")
                if len(lists.get(list_name, [])) <= 60:
                    choices = [f"{c['name']}={c['label']}" for c in lists.get(list_name, [])]
        rows.append({
            "name": parts[-1], "path": "/".join(parts[1:]), "group": "/".join(parts[1:-1]),
            "label": clean(lab.get_text(" ")) if lab else "", "type": typ,
            "choices": " | ".join(dict.fromkeys(choices)),
            "choice_list": list_name, "choice_filter": filt,
            "relevant": el.get("data-relevant", ""), "required": el.has_attr("data-required"),
        })
    return pd.DataFrame(rows), lists


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fid in FORMS.items():
        df, lists = extract(fid)
        df.to_csv(OUT / f"{name}.csv", index=False)
        # cascading location lists (country -> county/district -> sub-county/TA)
        for lst in ("district", "subcounty_TA"):
            if lst in lists:
                pd.DataFrame(lists[lst]).to_csv(OUT / f"{name}_{lst}.csv", index=False)
        print(name, len(df), "questions ->", OUT / f"{name}.csv")
