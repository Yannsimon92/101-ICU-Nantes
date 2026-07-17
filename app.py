"""Générateur de carte kepler.gl statique — ICU Nantes v2 recentrée.

Contrairement à l'ancienne version Streamlit, ce script n'est plus un serveur
interactif : il produit une **carte kepler.gl statique** (HTML autonome, sans
serveur) à partir des sorties de `compute_icu.py` et `model_evaluation.py`.

Sorties (`--out-dir`, défaut `data/web`) :

  - `icu_map.html`    : carte kepler.gl autonome (polygones pixel 100 m,
                       extrusion 3D sur la fréquence ICU, palette YlOrRd),
                       optionnellement avec un layer temporel `delta_lst_c`
                       (filtre `time` sur la date) si `--with-timeseries`.
  - `index.html`      : page enveloppante (header + iframe de la carte + tuiles
                       de métriques + panneau SHAP + encadré de mise en garde).

Lancement :
    python app.py [--raw-dir data/raw] [--icu-dir data/icu]
                  [--table data/table.parquet] [--eval-dir data/eval]
                  [--out-dir data/web] [--with-timeseries]

Fonctionne aussi **sans aucune donnée** (mode démo synthétique) pour la vitrine
portfolio : une grille 40×50 de polygones est générée avec des valeurs de
fréquence/deltaLST gaussiennes, et un avertissement est imprimé sur la console.

`--with-timeseries` produit un fichier nettement plus volumineux (jusqu'à
~450 000 features, 1 polygone par pixel × date) et est **désactivé par défaut**.
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np

# rasterio est requis pour le mode réel (lecture GeoTIFF + reprojection) mais
# optionnel pour le mode démo (grille synthétique en lat/lon direct).
try:
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    warp_transform = None
    transform_bounds = None
    HAS_RASTERIO = False

# kepler.gl est une dépendance dure : ce script produit une carte kepler.gl.
from keplergl import KeplerGl

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_ICU_DIR = "data/icu"
DEFAULT_TABLE = "data/table.parquet"
DEFAULT_EVAL_DIR = "data/eval"
DEFAULT_OUT_DIR = "data/web"
POINTS_PATH = "data/open_data/points_fraicheur.geojson"

DEMO_BOUNDS = [[47.19, -1.60], [47.24, -1.52]]   # centre Nantes approx
ICU_DELTA_THRESHOLD = 2.0   # seuil ΔLST (°C) repris de compute_icu.py


# ---------------------------------------------------------------------------
# Découverte des fichiers (réutilise la logique de l'ancien app.py)
# ---------------------------------------------------------------------------

LST_RE = re.compile(r"Y_lst_(?:100m_)?(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")
DELTA_RE = re.compile(r"delta_lst_(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")
FREQ_RE = re.compile(r"icu_frequency_(?P<zone>.+)\.tif$")


def discover(raw_dir, icu_dir):
    """Renvoie {zone: {"raw": {date: path}, "delta": {date: path},
                       "freq": path, "dates": [dates]}}."""
    out = {}

    def _entry(z):
        return out.setdefault(
            z, {"raw": {}, "delta": {}, "freq": None, "dates": []})

    if os.path.isdir(raw_dir):
        for path in glob.glob(os.path.join(raw_dir, "Y_lst_*.tif")):
            m = LST_RE.match(os.path.basename(path))
            if not m:
                continue
            z, d = m.group("zone"), m.group("date")
            e = _entry(z)
            e["raw"][d] = path
            if d not in e["dates"]:
                e["dates"].append(d)
    if os.path.isdir(icu_dir):
        for path in glob.glob(os.path.join(icu_dir, "delta_lst_*.tif")):
            m = DELTA_RE.match(os.path.basename(path))
            if not m:
                continue
            z, d = m.group("zone"), m.group("date")
            e = _entry(z)
            e["delta"][d] = path
            if d not in e["dates"]:
                e["dates"].append(d)
        for path in glob.glob(os.path.join(icu_dir, "icu_frequency_*.tif")):
            m = FREQ_RE.match(os.path.basename(path))
            if not m:
                continue
            _entry(m.group("zone"))["freq"] = path

    for z in out:
        out[z]["dates"] = sorted(set(out[z]["dates"]))
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Construction vectorisée des polygones pixel (row,col) -> anneau lng/lat
# ---------------------------------------------------------------------------

def _round_ring(ring):
    return [[round(c[0], 6), round(c[1], 6)] for c in ring]


def polygons_from_affine(cols, rows, raster_transform, src_crs):
    """Construit les anneaux (lng/lat EPSG:4326) des polygones carrés d'origine
    (col=row pixel top-left), de façon vectorisée via rasterio.warp.transform.

    `cols`, `rows` : array-like d'entiers (col=x_px, row=y_px).
    Renvoie une liste d'anneaus (coords [lng,lat] des 5 sommets, 6 décimales).
    """
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    a, b, c = raster_transform.a, raster_transform.b, raster_transform.c
    d, e, f = raster_transform.d, raster_transform.e, raster_transform.f

    def _xy(col, row):
        return a * col + b * row + c, d * col + e * row + f

    x_tl, y_tl = _xy(cols, rows)
    x_tr, y_tr = _xy(cols + 1, rows)
    x_br, y_br = _xy(cols + 1, rows + 1)
    x_bl, y_bl = _xy(cols, rows + 1)

    all_x = np.concatenate([x_tl, x_tr, x_br, x_bl]).tolist()
    all_y = np.concatenate([y_tl, y_tr, y_br, y_bl]).tolist()
    lng, lat = warp_transform(src_crs, "EPSG:4326", all_x, all_y)
    lng = np.asarray(lng)
    lat = np.asarray(lat)

    n = cols.shape[0]
    s = slice(0, n)
    s2 = slice(n, 2 * n)
    s3 = slice(2 * n, 3 * n)
    s4 = slice(3 * n, 4 * n)

    rings = []
    for i in range(n):
        ring = [
            (lng[i], lat[i]),
            (lng[s2][i], lat[s2][i]),
            (lng[s3][i], lat[s3][i]),
            (lng[s4][i], lat[s4][i]),
            (lng[i], lat[i]),
        ]
        rings.append(_round_ring(ring))
    return rings


def polygons_from_grid(nrows, ncols, lat0, lon0, dlat, dlon):
    """Polygones d'une grille régulière déjà en lat/lon (mode démo)."""
    rings = []
    for r in range(nrows):
        for c in range(ncols):
            lat_a = lat0 + r * dlat
            lat_b = lat0 + (r + 1) * dlat
            lon_a = lon0 + c * dlon
            lon_b = lon0 + (c + 1) * dlon
            ring = [
                (lon_a, lat_a), (lon_b, lat_a),
                (lon_b, lat_b), (lon_a, lat_b), (lon_a, lat_a),
            ]
            rings.append(_round_ring(ring))
    return rings


def assemble_features(rings, props):
    """Assemble une FeatureCollection GeoJSON (polygones).

    `rings` : liste d'anneaus [[lng,lat],...].
    `props` : dict nom -> array-like de longueur len(rings). Valeurs arrondies
    à 3 décimales si numériques (autres que chaînes).
    """
    feats = []
    n = len(rings)
    keys = list(props.keys())
    for i in range(n):
        feat = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [rings[i]]},
            "properties": {},
        }
        for k in keys:
            v = props[k][i]
            if isinstance(v, (float, np.floating)):
                feat["properties"][k] = round(float(v), 3)
            elif isinstance(v, (int, np.integer)):
                feat["properties"][k] = int(v)
            else:
                feat["properties"][k] = v
        feats.append(feat)
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# Mode réel : grille snapshot depuis table + raster fréquence
# ---------------------------------------------------------------------------

SNAPSHOT_COLS = ["lst_c", "delta_lst_c", "ndvi", "ndwi", "ndbi",
                "canopee", "bati", "tissu"]


def build_real_snapshot(zone, info, table_path, raw_dir, icu_dir,
                        with_timeseries=False):
    """Construit les FeatureCollections icu_grid (et icu_grid_timeseries si
    demandé) + un dict de métriques pour une zone détectée.

    Retourne (fc_snapshot, fc_timeseries_or_None, metrics, latest_date, rings).
    `rings` est la liste des anneaus réutilisée pour la série temporelle.
    """
    import pandas as pd

    # Un raster de référence pour récupérer transform + crs : on prend la
    # première date brute disponible, sinon le raster de fréquence.
    ref_path = None
    if info["raw"]:
        ref_path = info["raw"][sorted(info["raw"])[0]]
    elif info["freq"]:
        ref_path = info["freq"]
    if ref_path is None:
        raise RuntimeError(f"Pas de raster de référence pour la zone '{zone}'.")

    with rasterio.open(ref_path) as src:
        transform = src.transform
        src_crs = src.crs
        ref_shape = (src.height, src.width)

    # Fréquence ICU multi-dates (toujours utile même sans table).
    freq = None
    if info["freq"]:
        with rasterio.open(info["freq"]) as src:
            freq = src.read(1, masked=True).filled(np.nan).astype(np.float32)

    df = None
    if table_path and os.path.exists(table_path):
        try:
            df = pd.read_parquet(table_path)
        except Exception as exc:
            print(f"  Lecture de {table_path} impossible ({exc}); "
                  f"on se rabat sur les rasters seuls.")
            df = None

    if df is not None and "zone" in df.columns:
        df = df[df["zone"] == zone].copy()
    if df is not None and df.empty:
        df = None

    if df is not None and "date" in df.columns and not df.empty:
        latest_date = sorted(df["date"].astype(str).unique())[-1]
        snap = df[df["date"].astype(str) == latest_date].copy()
    else:
        latest_date = sorted(info["dates"])[-1] if info["dates"] else None
        snap = None

    if snap is not None and not snap.empty and "y_px" in snap.columns \
            and "x_px" in snap.columns:
        cols_px = snap["x_px"].to_numpy(dtype=int)
        rows_px = snap["y_px"].to_numpy(dtype=int)
        rings = polygons_from_affine(cols_px, rows_px, transform, src_crs)
        props = {}
        for c in SNAPSHOT_COLS:
            if c not in snap.columns:
                continue
            props[c] = (snap[c].to_numpy(dtype=np.float32) if c != "tissu"
                        else snap[c].to_numpy())
        if freq is not None:
            freq_vals = freq[rows_px, cols_px]
            props["icu_frequency_pct"] = freq_vals
        else:
            props["icu_frequency_pct"] = np.full(len(rows_px), np.nan,
                                                  dtype=np.float32)
    else:
        # Pas de table : on reconstruit la grille depuis les pixels valides
        # du raster de fréquence (ou du delta de la dernière date).
        delta = None
        if latest_date and info["delta"].get(latest_date):
            with rasterio.open(info["delta"][latest_date]) as src:
                delta = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        ref = freq if freq is not None else delta
        if ref is None:
            raise RuntimeError(f"Aucune donnée exploitable pour '{zone}'.")
        valid = np.isfinite(ref)
        rows_px, cols_px = np.where(valid)
        rings = polygons_from_affine(cols_px, rows_px, transform, src_crs)
        props = {}
        props["icu_frequency_pct"] = (freq[rows_px, cols_px]
                                       if freq is not None
                                       else np.zeros(len(rows_px),
                                                     dtype=np.float32))
        props["delta_lst_c"] = (delta[rows_px, cols_px]
                                 if delta is not None
                                 else np.zeros(len(rows_px),
                                               dtype=np.float32))
        for c in ("lst_c", "ndvi", "ndwi", "ndbi", "canopee", "bati", "tissu"):
            props[c] = np.full(len(rows_px), np.nan, dtype=np.float32)

    fc_snapshot = assemble_features(rings, props)
    metrics = compute_metrics(props, zone, latest_date)

    fc_ts = None
    if with_timeseries:
        fc_ts = build_timeseries(zone, df, rings, rows_px, cols_px, latest_date)
        if fc_ts is None:
            print(f"  Série temporelle indisponible pour '{zone}' "
                  f"(table nécessaire).")

    return fc_snapshot, fc_ts, metrics, latest_date, rings


def build_timeseries(zone, df, rings, rows_px, cols_px, latest_date):
    """Construit la FeatureCollection temporelle (1 polygone par pixel×date),
    propriétés minimales date (ISO) + delta_lst_c."""
    if df is None or "date" not in df.columns or "delta_lst_c" not in df.columns:
        return None
    # Index pixel -> position dans rings (basée sur snapshot = dernière date).
    # On map (y_px, x_px) -> index pour joindre rapidement chaque date.
    pix_to_idx = {(int(r), int(c)): i
                  for i, (r, c) in enumerate(zip(rows_px, cols_px))}
    dates = sorted(df["date"].astype(str).unique())
    feats = []
    for d in dates:
        sub = df[df["date"].astype(str) == d]
        # date ISO YYYY-MM-DD
        d_iso = d[:10]
        for r, c, delta in zip(sub["y_px"].to_numpy(dtype=int),
                                sub["x_px"].to_numpy(dtype=int),
                                sub["delta_lst_c"].to_numpy()):
            idx = pix_to_idx.get((int(r), int(c)))
            ring = rings[idx] if idx is not None else None
            if ring is None:
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"date": d_iso,
                                 "delta_lst_c": round(float(delta), 3)},
            })
    if not feats:
        return None
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# Mode démo : grille synthétique 40×50
# ---------------------------------------------------------------------------

def build_demo_snapshot():
    """Grille de polygones synthétique (40×50) avec fréquence/deltaLST
    gaussiens, pour la vitrine sans données."""
    rng = np.random.default_rng(0)
    nrows, ncols = 40, 50
    lat0, lon0 = DEMO_BOUNDS[0]
    lat1, lon1 = DEMO_BOUNDS[1]
    dlat = (lat1 - lat0) / nrows
    dlon = (lon1 - lon0) / ncols

    yy, xx = np.mgrid[0:nrows, 0:ncols].astype(np.float32)
    freq = 100.0 * np.exp(
        -(((xx - ncols / 2) / (ncols / 3)) ** 2
          + ((yy - nrows / 2) / (nrows / 3)) ** 2))
    freq += rng.normal(0, 5.0, (nrows, ncols)).astype(np.float32)
    freq = np.clip(freq, 0, 100)
    lst = 26 + 5 * (freq / 100.0) + rng.normal(0, 0.4, (nrows, ncols))
    delta = lst - np.median(lst)

    rings = polygons_from_grid(nrows, ncols, lat0, lon0, dlat, dlon)
    props = {}
    flat = lambda arr2: arr2.flatten()
    props["lst_c"] = flat(lst)
    props["delta_lst_c"] = flat(delta)
    props["icu_frequency_pct"] = flat(freq)
    for c in ("ndvi", "ndwi", "ndbi", "canopee", "bati", "tissu"):
        props[c] = np.full(nrows * ncols, np.nan, dtype=np.float32)

    fc_snapshot = assemble_features(rings, props)
    metrics = compute_metrics(props, "démo", None)
    return fc_snapshot, None, metrics, None, rings


def build_demo_timeseries(rings):
    """Série temporelle synthétique légère (3 dates) pour le mode démo."""
    n = len(rings)
    feats = []
    for k, d_iso in enumerate(("2022-07-10", "2023-07-15", "2024-07-08")):
        rng = np.random.default_rng(k)
        delta = rng.normal(0, 1.5, n)
        for i in range(n):
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [rings[i]]},
                "properties": {"date": d_iso,
                                 "delta_lst_c": round(float(delta[i]), 3)},
            })
    return {"type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# Métriques (tuiles HTML) — calculées sur les arrays déjà chargés
# ---------------------------------------------------------------------------

def compute_metrics(props, zone, latest_date):
    """Renvoie un dict de métriques synthétiques pour les tuiles HTML."""
    freq = props.get("icu_frequency_pct")
    delta = props.get("delta_lst_c")
    res = {"zone": zone, "date": latest_date or "-"}
    if freq is not None and np.isfinite(freq).any():
        fv = freq[np.isfinite(freq)]
        res["freq_mean"] = float(np.mean(fv))
        res["surface_icu_pct"] = float((fv > 50).mean() * 100.0)
        res["freq_max"] = float(np.nanmax(fv))
    else:
        res["freq_mean"] = float("nan")
        res["surface_icu_pct"] = float("nan")
        res["freq_max"] = float("nan")
    if delta is not None and np.isfinite(delta).any():
        dv = delta[np.isfinite(delta)]
        res["delta_median"] = float(np.median(dv))
    else:
        res["delta_median"] = float("nan")
    return res


# ---------------------------------------------------------------------------
# Config kepler.gl
# ---------------------------------------------------------------------------

YLOrrD_RANGE = {
    "name": "YlOrRd", "type": "sequential", "category": "ColorBrewer",
    "colors": ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c",
               "#fc4e2a", "#e31a1c"],
}
RDBU_R_RANGE = {
    "name": "RdBu_r", "type": "diverging", "category": "ColorBrewer",
    "colors": ["#b2182b", "#ef8a62", "#fddbc7",
               "#d1e5f0", "#67a9cf", "#2166ac"],
}


def _polygon_layer(layer_id, data_id, label, color_field, color_range,
                   height_field=None, opacity=0.8):
    layer = {
        "id": layer_id, "type": "polygon",
        "config": {
            "dataId": data_id, "label": label,
            "color": [255, 102, 97], "highlightColor": [252, 242, 26, 255],
            "opacity": opacity,
            "thickness": {"thickness": 0, "enable3d": height_field is not None},
            "isVisible": True,
            "visConfig": {
                "opacity": opacity, "thickness": 0,
                "enable3d": height_field is not None,
                "colorRange": color_range,
                "sizeRange": [0, 500],
                "heightRange": [0, 800],
                "coverage": 1.0, "elevationScale": 1.0,
                "wireframe": False,
            },
        },
        "visualChannels": {
            "color": {
                "field": {"name": color_field, "type": "real"},
                "scale": "quantize", "range": color_range,
            },
        },
    }
    if height_field is not None:
        layer["visualChannels"]["height"] = {
            "field": {"name": height_field, "type": "real"},
            "scale": "linear", "range": [0, 800],
        }
    return layer


def _point_layer(layer_id, data_id, label):
    return {
        "id": layer_id, "type": "point",
        "config": {
            "dataId": data_id, "label": label,
            "color": [38, 132, 255], "highlightColor": [252, 242, 26, 255],
            "opacity": 0.9, "thickness": {"thickness": 1, "enable3d": False},
            "isVisible": True,
            "visConfig": {
                "opacity": 0.9, "outline": False, "outlineColor": None,
                "thickness": 0.2, "strokeColor": None, "colorRange": {
                    "name": "Global Warming", "type": "sequential",
                    "category": "ColorBrewer",
                    "colors": ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb",
                               "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"],
                },
                "radius": 50, "sizeRange": [0, 10], "radiusRange": [0, 50],
                "heightRange": [0, 500], "elevationScale": 5, "enable3d": False,
            },
        },
        "visualChannels": {
            "color": {"field": None, "scale": "quantize", "range": None},
            "size": {"field": None, "scale": "quantize", "range": None},
        },
    }


def build_kepler_config(center, has_timeseries, has_points, ts_epoch=None):
    layers = [
        _polygon_layer("icu_grid_layer", "icu_grid", "Fréquence ICU (%)",
                       "icu_frequency_pct", YLOrrD_RANGE,
                       height_field="icu_frequency_pct", opacity=0.8),
    ]
    filters = []
    if has_timeseries:
        layers.append(_polygon_layer(
            "icu_ts_layer", "icu_grid_timeseries",
            "Anomalie ΔLST (°C, série temporelle)",
            "delta_lst_c", RDBU_R_RANGE, height_field="delta_lst_c",
            opacity=0.7))
        if ts_epoch:
            filters.append({
                "dataId": "icu_grid_timeseries", "id": "filter_time",
                "name": ["date"], "type": "time",
                "value": list(ts_epoch), "enlarged": True,
                "plotType": "histogram", "animationWindow": "free",
                "yAxis": None, "speed": 1, "layerId": ["icu_ts_layer"],
            })
    if has_points:
        layers.append(_point_layer("points_layer", "points_fraicheur",
                                    "Points de fraîcheur"))
    return {
        "version": "v1",
        "config": {
            "mapState": {
                "latitude": center[0], "longitude": center[1],
                "zoom": 11, "pitch": 45, "bearing": 0,
                "dragRotate": True,
            },
            "mapStyle": {"styleType": "light", "topLayerGroups": [],
                          "visibleLayers": [], "threeDBuildingColor": None},
            "visState": {
                "filters": filters, "layers": layers,
                "interactionConfig": {
                    "tooltip": {"enabled": True, "visualChannels": ["color"]},
                    "brush": {"enabled": False, "size": 0.5},
                    "geocoder": {"enabled": False},
                    "coordinate": {"enabled": False},
                },
                "layerBlending": "normal", "splitMaps": [],
            },
        },
    }


def fc_bounds(fc):
    """Bounds [min_lng, min_lat, max_lng, max_lat] d'une FeatureCollection."""
    lngs, lats = [], []
    for ft in fc["features"]:
        for pt in ft["geometry"]["coordinates"][0]:
            lngs.append(pt[0]); lats.append(pt[1])
    if not lngs:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(lngs), min(lats), max(lngs), max(lats)]


# ---------------------------------------------------------------------------
# Page HTML enveloppante
# ---------------------------------------------------------------------------

def _metric_tile(label, value):
    return (
        '<div class="metric">'
        f'<span class="metric-label">{label}</span>'
        f'<span class="metric-value">{value}</span>'
        '</div>'
    )


def _fmt(x, suffix, precision=1):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{precision}f}{suffix}"


def build_index_html(out_dir, metrics, eval_dir, map_file="icu_map.html"):
    tiles = [
        _metric_tile("Fréquence ICU moyenne", _fmt(metrics["freq_mean"], " %")),
        _metric_tile("ΔLST médian",
                     _fmt(metrics["delta_median"], " °C", 2)),
        _metric_tile("Surface ICU (> 50 % dates)",
                     _fmt(metrics["surface_icu_pct"], " %")),
        _metric_tile("Zone / date",
                     f"{metrics['zone']} — {metrics['date']}"),
    ]
    metric_html = "\n".join(tiles)

    metrics_path = os.path.join(eval_dir, "metrics.json")
    shap_path = os.path.join(eval_dir, "shap_summary.png")
    shap_section = build_shap_section(eval_dir, metrics_path, shap_path)

    disclaimer = (
        "• La carte montre une <strong>température de surface</strong> "
        "(LST Landsat), pas la température de l'air ressenti.<br>"
        "• Acquisitions <strong>matinales d'été par ciel clair</strong> "
        "(~10h50 UTC, SUHI diurne) : ce n'est <strong>pas</strong> l'îlot de "
        "chaleur nocturne visé par les politiques de fraîcheur urbaine.<br>"
        "• Résolution 100 m = échelle du <strong>quartier</strong>, pas de "
        "la rue. Les cœurs d'îlots (cours, rues étroites) ne sont pas "
        "résolus.<br>"
        "• Validation <strong>in situ</strong> non réalisée à ce stade "
        "(cf. audit scientifique) ; résultats à visée exploratoire et "
        "pédagogique."
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Îlots de Chaleur Urbains — Nantes Métropole</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0 16px 32px;
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    color: #222; background: #f7f7f9; line-height: 1.5; max-width: 1200px;
    margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 1.6rem; margin: 0.4em 0 0; }}
  .caption {{ color: #555; font-size: 0.95rem; margin: 0 0 1em; }}
  .metrics {{
    display: flex; flex-wrap: wrap; gap: 12px; margin: 1em 0;
  }}
  .metric {{
    flex: 1 1 220px; background: #fff; border: 1px solid #e3e3e3;
    border-radius: 8px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  .metric-label {{ display: block; font-size: 0.8rem; color: #666;
    text-transform: uppercase; letter-spacing: 0.04em; }}
  .metric-value {{ display: block; font-size: 1.4rem; font-weight: 600;
    margin-top: 4px; }}
  iframe {{ width: 100%; height: 750px; border: none;
    border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  section {{ margin: 2em 0; }}
  h2 {{ font-size: 1.25rem; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  .shap-row {{ display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }}
  .shap-row img {{ max-width: 640px; width: 100%; border-radius: 6px; }}
  .shap-side {{ font-size: 0.92rem; }}
  .shap-side .feat {{ margin: 2px 0; }}
  .shap-side .model {{ margin: 8px 0; font-weight: 600; }}
  .disclaimer {{
    background: #fff8e1; border: 1px solid #ffe082; border-left: 6px solid #ffb300;
    border-radius: 6px; padding: 14px 18px; font-size: 0.92rem; color: #5c4b00;
    margin: 2em 0;
  }}
  .info {{ background: #eef4fb; border: 1px solid #cfe2f3; border-radius: 6px;
    padding: 12px 16px; color: #225; font-size: 0.92rem; }}
</style>
</head>
<body>
  <h1>🌡️ Îlots de Chaleur Urbains — Nantes Métropole</h1>
  <p class="caption">Cartographie et analyse explicative des ICU à la résolution
  native Landsat (100 m). Modélisation : régression linéaire + LightGBM,
  interprétation SHAP.</p>

  <div class="metrics">
    {metric_html}
  </div>

  <section>
    <h2>Carte kepler.gl — fréquence ICU multi-dates</h2>
    <p class="caption">Extrusion 3D sur la fréquence ICU (% de dates en
    surchauffe), palette YlOrRd. Survolez une cellule 100 m pour voir ses
    propriétés (ΔLST, NDVI/NDWI/NDBI, canopée, bâti).</p>
    <iframe src="{map_file}" title="Carte kepler.gl ICU Nantes"></iframe>
  </section>

  {shap_section}

  <section>
    <h2>⚠️ Mise en garde — lire avant toute interprétation</h2>
    <div class="disclaimer">{disclaimer}</div>
  </section>
</body>
</html>
"""
    return html


def build_shap_section(eval_dir, metrics_path, shap_path):
    if not os.path.exists(metrics_path):
        return (
            '<section><h2>Analyse explicative — SHAP</h2>'
            f'<div class="info">{metrics_path} absent — l\'analyse explicative '
            "n'a pas été lancée. Pipeline : <code>build_table.py</code> puis "
            "<code>model_evaluation.py</code>.</div></section>"
        )
    with open(metrics_path, encoding="utf-8") as f:
        meta = json.load(f)
    shap_exists = os.path.exists(shap_path)
    sh = meta.get("shap", {}).get("mean_abs_shap", {})
    top = sorted(sh.items(), key=lambda kv: -kv[1])[:5]
    feat_rows = "\n".join(
        f'<div class="feat"><b>{k}</b> — {v:.4f}</div>' for k, v in top)
    lgb_meta = meta.get("lightgbm", {})
    lin_meta = meta.get("linear", {})
    coeffs = lin_meta.get("coeffs_°C_per_unit", {})
    coeffs_txt = ", ".join(f"{k}={v:+.2f}" for k, v in coeffs.items())

    img_block = ""
    if shap_exists:
        img_block = (
            '<img src="../eval/shap_summary.png" '
            'alt="Importance SHAP (mean |shap|) — LightGBM" '
            'title="Importance SHAP (mean |shap|) — LightGBM">'
        )
    else:
        img_block = (
            '<div class="info"><code>shap_summary.png</code> absent — relancer '
            "<code>model_evaluation.py</code>.</div>"
        )

    lgb_r2 = lgb_meta.get("r2_global", 0)
    lgb_rmse = lgb_meta.get("rmse_global", 0)
    lin_r2 = lin_meta.get("r2_global", 0)
    lin_rmse = lin_meta.get("rmse_global", 0)
    coeffs_disp = coeffs_txt if coeffs_txt else "n/a"
    thirds = f"""
  <section>
    <h2>Analyse explicative — SHAP</h2>
    <div class="shap-row">
      {img_block}
      <div class="shap-side">
        <div class="model">Top features (mean |SHAP|)</div>
        {feat_rows if feat_rows else '<div class="feat">n/a</div>'}
        <div class="model">LightGBM : R² = {lgb_r2:.3f}, RMSE = {lgb_rmse:.3f} °C</div>
        <div class="model">Linéaire : R² = {lin_r2:.3f}, RMSE = {lin_rmse:.3f} °C</div>
        <div class="caption">Coefficients linéaires (°C par unité) : {coeffs_disp}</div>
      </div>
    </div>
  </section>"""
    return thirds


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--icu-dir", default=DEFAULT_ICU_DIR)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--with-timeseries", action="store_true",
                    help="génère aussi le layer temporel icu_grid_timeseries "
                         "(fichier plus volumineux, jusqu'à ~450k features).")
    return ap.parse_args(argv)


def run(raw_dir=DEFAULT_RAW_DIR, icu_dir=DEFAULT_ICU_DIR,
        table=DEFAULT_TABLE, eval_dir=DEFAULT_EVAL_DIR,
        out_dir=DEFAULT_OUT_DIR, with_timeseries=False, points_path=POINTS_PATH):
    """Point d'entrée réutilisable (appelable depuis les tests)."""
    os.makedirs(out_dir, exist_ok=True)

    scenes = discover(raw_dir, icu_dir) if HAS_RASTERIO else {}
    real_zones = [z for z, info in scenes.items()
                  if info["freq"] or info["delta"] or info["raw"]]

    datasets = {}
    metrics = {}
    all_fc_snapshot = {"type": "FeatureCollection", "features": []}
    fc_ts = None
    ts_epoch = None

    if real_zones:
        for z in real_zones:
            info = scenes[z]
            fc, fc_ts_z, m, latest_date, rings = build_real_snapshot(
                z, info, table, raw_dir, icu_dir,
                with_timeseries=with_timeseries)
            all_fc_snapshot["features"].extend(fc["features"])
            metrics = m  # dernière zone (typiquement une seule)
            if fc_ts_z is not None:
                fc_ts = fc_ts_z
        datasets["icu_grid"] = all_fc_snapshot
        if fc_ts is not None:
            datasets["icu_grid_timeseries"] = fc_ts
            # epoch min/max pour le filtre time kepler
            dates = sorted({ft["properties"]["date"]
                            for ft in fc_ts["features"]})
            if dates:
                def _epoch(d):
                    return int(datetime.fromisoformat(d).timestamp() * 1000)
                ts_epoch = (_epoch(dates[0]), _epoch(dates[-1]))
    else:
        print("Aucune donnée détectée : passage en mode démo (grille "
              "synthétique 40×50). Lancez le pipeline "
              "(gee_extraction → compute_icu → build_table) pour de vraies "
              "données.")
        fc, _, m, _, rings = build_demo_snapshot()
        datasets["icu_grid"] = fc
        metrics = m
        if with_timeseries:
            fc_ts = build_demo_timeseries(rings)
            datasets["icu_grid_timeseries"] = fc_ts
            ts_epoch = (int(datetime.fromisoformat("2022-07-10").timestamp() * 1000),
                        int(datetime.fromisoformat("2024-07-08").timestamp() * 1000))

    has_ts = "icu_grid_timeseries" in datasets
    if points_path and os.path.exists(points_path):
        try:
            with open(points_path, encoding="utf-8") as f:
                datasets["points_fraicheur"] = json.load(f)
        except Exception as exc:
            print(f"  Points de fraîcheur non chargés : {exc}")

    has_points = "points_fraicheur" in datasets
    bounds = fc_bounds(datasets["icu_grid"])
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    config = build_kepler_config(center, has_ts, has_points, ts_epoch=ts_epoch)

    map_1 = KeplerGl(height=750, data=datasets, config=config)
    map_path = os.path.join(out_dir, "icu_map.html")
    map_1.save_to_html(file_name=map_path, read_only=False)

    # Page enveloppante (référence les métriques + SHAP + disclaimer).
    index_html = build_index_html(out_dir, metrics, eval_dir, map_file="icu_map.html")
    # Décale la référence SHAP si out_dir n'est pas data/web : on garde le
    # chemin relatif ../eval/ (pertinent pour l'usage standard data/web).
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    return map_path, index_path


def main(argv=None):
    args = parse_args(argv)
    map_path, index_path = run(
        raw_dir=args.raw_dir, icu_dir=args.icu_dir, table=args.table,
        eval_dir=args.eval_dir, out_dir=args.out_dir,
        with_timeseries=args.with_timeseries)
    print(f"\nCarte kepler.gl générée : {map_path}")
    print(f"Page enveloppante     : {index_path}")
    print("Ouvrir ce fichier dans un navigateur : "
          f"file://{os.path.abspath(index_path)}")


if __name__ == "__main__":
    main()