"""Administrative boundaries for any country (geoBoundaries, open licence) and map layers.

geoBoundaries: https://www.geoboundaries.org (William & Mary geoLab), CC BY 4.0 / public domain.
Imagery layers: Esri World Imagery and NASA GIBS (https://nasa-gibs.github.io/gibs-api-docs/).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import requests

GB_API = "https://www.geoboundaries.org/api/current/gbOpen"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
HEADERS = {"User-Agent": "climate-health-ews-prototype/0.2 (research)"}


def _cached_json(url: str, name: str, ttl_days: float = 30):
    path = CACHE / name
    if path.exists() and time.time() - path.stat().st_mtime < ttl_days * 86400:
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    data = r.json()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def list_countries() -> list[dict]:
    """[{iso, name}] for every country with boundaries."""
    data = _cached_json(f"{GB_API}/ALL/ADM0/", "gb_countries.json")
    out = {d["boundaryISO"]: d["boundaryName"] for d in data}
    return sorted(({"iso": k, "name": v} for k, v in out.items()), key=lambda d: d["name"])


def level_info(iso: str, level: str) -> dict | None:
    try:
        return _cached_json(f"{GB_API}/{iso}/{level}/", f"gb_{iso}_{level}_meta.json")
    except Exception:
        return None


def available_levels(iso: str) -> dict[str, int]:
    out = {}
    for lv in ("ADM1", "ADM2", "ADM3"):
        meta = level_info(iso, lv)
        if meta:
            out[lv] = int(meta.get("admUnitCount") or 0)
    return out


def default_level(levels: dict[str, int]) -> str:
    """Lowest level with at least 10 units (e.g. Kenya ADM1 = 47 counties, Malawi ADM2 = 28 districts)."""
    for lv in ("ADM1", "ADM2", "ADM3"):
        if levels.get(lv, 0) >= 10:
            return lv
    return next(iter(levels), "ADM1")


def _round_coords(obj, nd=3):
    if isinstance(obj, list):
        if obj and isinstance(obj[0], (int, float)):
            return [round(v, nd) for v in obj[:2]]
        return [_round_coords(o, nd) for o in obj]
    return obj


BUNDLED = Path(__file__).resolve().parent.parent / "data" / "boundaries"
HDX_API = "https://data.humdata.org/api/3/action/package_show?id=cod-ab-{iso}"


def _simplify_ring(ring, tol=0.003):
    """Douglas-Peucker on one ring (degrees)."""
    pts = np.asarray(ring, float)[:, :2]
    if len(pts) < 8:
        return pts.tolist()
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        d = pts[a + 1:b] - pts[a]
        norm = np.hypot(*seg) or 1e-12
        dist = np.abs(seg[0] * d[:, 1] - seg[1] * d[:, 0]) / norm
        i = int(np.argmax(dist))
        if dist[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    out = pts[keep]
    return out.tolist() if len(out) >= 4 else pts[[0, len(pts) // 3, 2 * len(pts) // 3, 0]].tolist()


def simplify_geometry(geom, tol=0.003):
    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [_simplify_ring(r, tol) for r in geom["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[_simplify_ring(r, tol) for r in poly] for poly in geom["coordinates"]]}


def hdx_boundaries(iso: str, level: str) -> dict | None:
    """OCHA common operational boundaries (COD-AB) from HDX, CC BY-IGO. Used when geoBoundaries is
    incomplete. Downloads once (can be tens of MB), then simplified and cached."""
    import io
    import zipfile
    n = level[-1]
    bundled = BUNDLED / f"{iso}_{level}.geojson"
    if bundled.exists():
        return json.loads(bundled.read_text(encoding="utf-8"))
    cached = CACHE / f"hdx_{iso}_{level}.geojson"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    meta = requests.get(HDX_API.format(iso=iso.lower()), timeout=60).json()
    if not meta.get("success"):
        return None
    res = next((r for r in meta["result"]["resources"] if r["name"].endswith("admin_boundaries.geojson.zip")), None)
    if res is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(requests.get(res["url"], timeout=600).content))
    member = next((m for m in z.namelist() if m.endswith(f"_admin{n}.geojson")), None)
    if member is None:
        return None
    raw = json.loads(z.read(member))
    feats = [{"type": "Feature", "geometry": simplify_geometry(f["geometry"]),
              "properties": {"shapeName": f["properties"].get(f"adm{n}_name"),
                             "shapeISO": f["properties"].get(f"adm{n}_pcode", "")}}
             for f in raw["features"] if f.get("geometry")]
    gj = {"type": "FeatureCollection", "features": feats,
          "source": "OCHA COD-AB via HDX (CC BY-IGO)"}
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(gj), encoding="utf-8")
    return gj


def boundaries(iso: str, level: str) -> dict:
    """Simplified GeoJSON with properties: uid, name, centroid (lat, lon).
    geoBoundaries first; if its file has fewer units than it should (e.g. Malawi ADM2 has 16 of 28
    districts), fall back to OCHA COD-AB from HDX."""
    meta = level_info(iso, level)
    gj, source = None, "geoBoundaries (gbOpen)"
    if meta:
        gj = _cached_json(meta["simplifiedGeometryGeoJSON"], f"gb_{iso}_{level}.geojson")
        expected = int(meta.get("admUnitCount") or 0)
        if expected and len(gj["features"]) < 0.9 * expected:
            try:
                alt = hdx_boundaries(iso, level)
            except Exception:
                alt = None
            if alt and len(alt["features"]) >= 0.9 * expected:
                gj, source = alt, alt.get("source", "OCHA COD-AB")
    feats = []
    for i, f in enumerate(gj["features"]):
        geom = f["geometry"]
        geom = {"type": geom["type"], "coordinates": _round_coords(geom["coordinates"])}
        lat, lon = centroid(geom)
        props = f.get("properties", {})
        feats.append({"type": "Feature", "id": str(i), "geometry": geom,
                      "properties": {"uid": str(i), "name": props.get("shapeName") or f"Unit {i + 1}",
                                     "iso": props.get("shapeISO", ""), "lat": lat, "lon": lon}})
    return {"type": "FeatureCollection", "features": feats, "source": source}


def _ring_centroid(ring):
    pts = np.asarray(ring, float)
    x, y = pts[:, 0], pts[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cross.sum() / 2
    if abs(a) < 1e-12:
        return x.mean(), y.mean(), 0.0
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * a)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
    return cx, cy, abs(a)


def centroid(geom) -> tuple[float, float]:
    """Area-weighted centroid of the outer rings. Returns (lat, lon)."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    parts = [_ring_centroid(p[0]) for p in polys if p and len(p[0]) > 2]
    if not parts:
        return 0.0, 0.0
    w = np.array([p[2] for p in parts]) + 1e-12
    cx = np.average([p[0] for p in parts], weights=w)
    cy = np.average([p[1] for p in parts], weights=w)
    return float(cy), float(cx)


def bbox(gj: dict):
    xs, ys = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for cc in c:
                walk(cc)
    for f in gj["features"]:
        walk(f["geometry"]["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def zoom_for(bounds) -> float:
    w = max(bounds[2] - bounds[0], (bounds[3] - bounds[1]) * 1.6, 0.5)
    return float(np.clip(np.log2(360 / w) + 0.5, 2, 11))


def norm_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\b(county|district|sub[- ]?county|city|municipality|region|province|sta|ta)\b", " ", s)
    return re.sub(r"[^a-z]", "", s)


# ------------------------------------------------------------------ imagery layers
GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/{tms}/{{z}}/{{y}}/{{x}}.{ext}"

BASEMAPS = {
    "Satellite (Esri World Imagery)": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Imagery: Esri, Maxar, Earthstar Geographics", "maxzoom": 18, "dated": False},
    "NASA true colour, daily (MODIS Terra)": {
        "layer": "MODIS_Terra_CorrectedReflectance_TrueColor", "tms": "GoogleMapsCompatible_Level9", "ext": "jpg",
        "attribution": "NASA EOSDIS GIBS / Worldview", "maxzoom": 9, "dated": True},
    "NASA true colour, daily (VIIRS NOAA-20)": {
        "layer": "VIIRS_NOAA20_CorrectedReflectance_TrueColor", "tms": "GoogleMapsCompatible_Level9", "ext": "jpg",
        "attribution": "NASA EOSDIS GIBS / Worldview", "maxzoom": 9, "dated": True},
    "Streets (OpenStreetMap)": {"style": "open-street-map", "dated": False},
}

OVERLAYS = {
    "None": None,
    "Observed flooding, 3-day (NASA VIIRS)": {"layer": "VIIRS_Combined_Flood_3-Day", "tms": "GoogleMapsCompatible_Level9",
                                              "ext": "png", "maxzoom": 9},
    "Observed flooding, 3-day (NASA MODIS)": {"layer": "MODIS_Combined_Flood_3-Day", "tms": "GoogleMapsCompatible_Level9",
                                              "ext": "png", "maxzoom": 9},
    "Rainfall rate (NASA IMERG)": {"layer": "IMERG_Precipitation_Rate", "tms": "GoogleMapsCompatible_Level6",
                                   "ext": "png", "maxzoom": 6},
    "Land surface temperature, day (MODIS)": {"layer": "MODIS_Terra_Land_Surface_Temp_Day",
                                              "tms": "GoogleMapsCompatible_Level7", "ext": "png", "maxzoom": 7},
    "Vegetation NDVI, 8-day (MODIS)": {"layer": "MODIS_Terra_NDVI_8Day", "tms": "GoogleMapsCompatible_Level9",
                                       "ext": "png", "maxzoom": 9},
    "Root-zone soil moisture (SMAP L4)": {"layer": "SMAP_L4_Analyzed_Root_Zone_Soil_Moisture",
                                          "tms": "GoogleMapsCompatible_Level6", "ext": "png", "maxzoom": 6},
}


def gibs_url(spec: dict, day: str) -> str:
    return GIBS.format(layer=spec["layer"], date=day, tms=spec["tms"], ext=spec["ext"])
