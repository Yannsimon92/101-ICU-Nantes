"""Application Streamlit — ICU Nantes v2 recentrée.

Visualise les îlots de chaleur urbains à la résolution native Landsat (100 m)
sur Nantes Métropole, à partir des sorties de `compute_icu.py` et
`model_evaluation.py` :

  - LST Landsat brut (°C) par date,
  - Anomalie ΔLST = LST − médiane spatiale (°C) par date,
  - **Fréquence ICU multi-dates** (% de dates où le pixel est en surchauffe —
    le livrable robuste, beaucoup plus stable qu'une date unique),
  - Panneau SHAP : importance des variables et dependence plots,
  - Marqueurs « points de fraîcheur » Open Data superposés (si disponibles),
  - Encadré de mise en garde : température *de surface* (pas de l'air),
    matinées d'été par ciel clair (SUHI diurne ≠ UHI nocturne visé par les
    politiques de fraîcheur), résolution 100 m = échelle du quartier pas de la
    rue.

Lancement :
    streamlit run app.py

Fonctionne aussi sans aucune donnée (mode démo synthétique) pour la vitrine
portfolio.
"""

import glob
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

try:
    import rasterio
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

RAW_DIR = "data/raw"
ICU_DIR = "data/icu"
EVAL_DIR = "data/eval"
POINTS_PATH = "data/open_data/points_fraicheur.geojson"

DEMO_BOUNDS = [[47.19, -1.60], [47.24, -1.52]]   # centre Nantes approx

st.set_page_config(page_title="ICU Nantes", page_icon="🌡️", layout="wide")


# ---------------------------------------------------------------------------
# Découverte des fichiers
# ---------------------------------------------------------------------------

LST_RE = re.compile(r"Y_lst_(?:100m_)?(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")
DELTA_RE = re.compile(r"delta_lst_(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")
FREQ_RE = re.compile(r"icu_frequency_(?P<zone>.+)\.tif$")


def discover():
    """Renvoie {zone: {"dates": [dates], "raw": {date: path}, "delta": {date: path},
                       "freq": path, "tissu_counts": {...}}}."""
    out = {}
    for path in glob.glob(os.path.join(RAW_DIR, "Y_lst_*.tif")):
        m = LST_RE.match(os.path.basename(path))
        if not m:
            continue
        z, d = m.group("zone"), m.group("date")
        out.setdefault(z, {"raw": {}, "delta": {}, "freq": None, "dates": []})
        out[z]["raw"][d] = path
        out[z]["dates"].append(d)
    for path in glob.glob(os.path.join(ICU_DIR, "delta_lst_*.tif")):
        m = DELTA_RE.match(os.path.basename(path))
        if not m:
            continue
        z, d = m.group("zone"), m.group("date")
        out.setdefault(z, {"raw": {}, "delta": {}, "freq": None, "dates": []})
        out[z]["delta"][d] = path
        if d not in out[z]["dates"]:
            out[z]["dates"].append(d)
    for path in glob.glob(os.path.join(ICU_DIR, "icu_frequency_*.tif")):
        m = FREQ_RE.match(os.path.basename(path))
        if not m:
            continue
        z = m.group("zone")
        out.setdefault(z, {"raw": {}, "delta": {}, "freq": None, "dates": []})
        out[z]["freq"] = path
    for z in out:
        out[z]["dates"] = sorted(set(out[z]["dates"]))
    return dict(sorted(out.items()))


@st.cache_data(show_spinner="Lecture du raster…")
def load_raster(path):
    """Retourne (array 2D avec NaN, bounds folium [[S,W],[N,E]])."""
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return data, [[south, west], [north, east]]


@st.cache_data
def demo_scene(seed=0):
    """Scène LST synthétique : gradient urbain + noyaux chauds, pour la vitrine."""
    rng = np.random.default_rng(seed)
    h, w = 256, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lst = 26 + 4 * np.exp(-(((xx - w / 2) / (w / 3)) ** 2 + ((yy - h / 2) / (h / 3)) ** 2))
    for _ in range(6):
        cx, cy, amp = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(2, 5)
        lst += amp * np.exp(-(((xx - cx) / 25) ** 2 + ((yy - cy) / 25) ** 2))
    lst += rng.normal(0, 0.3, (h, w)).astype(np.float32)
    return lst, DEMO_BOUNDS


# ---------------------------------------------------------------------------
# Rendu carte
# ---------------------------------------------------------------------------

def colorize(data, cmap_name, vmin, vmax):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = matplotlib.colormaps[cmap_name](norm(np.nan_to_num(data, nan=vmin)))
    rgba[..., 3] = np.where(np.isfinite(data), 1.0, 0.0)
    return rgba


def build_map(data, bounds, cmap_name, vmin, vmax, opacity, caption,
              points_path=None):
    import branca.colormap as bcm

    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    fmap = folium.Map(location=center, tiles="CartoDB positron", zoom_start=13)
    folium.raster_layers.ImageOverlay(
        image=colorize(data, cmap_name, vmin, vmax),
        bounds=bounds, opacity=opacity, origin="upper",
    ).add_to(fmap)

    steps = matplotlib.colormaps[cmap_name](np.linspace(0, 1, 12))
    legend = bcm.LinearColormap(
        [matplotlib.colors.to_hex(c) for c in steps],
        vmin=vmin, vmax=vmax, caption=caption,
    )
    legend.add_to(fmap)

    if points_path and os.path.exists(points_path):
        import json as _json
        try:
            with open(points_path, encoding="utf-8") as f:
                gj = _json.load(f)
            feats = gj.get("features", [])
            for ft in feats:
                coords = ft.get("geometry", {}).get("coordinates") or [0, 0]
                name = (ft.get("properties") or {}).get("nom") or "Point fraîcheur"
                folium.Marker(
                    location=[coords[1], coords[0]], popup=name,
                    icon=folium.Icon(color="blue", icon="tree-deciduous", prefix="fa"),
                ).add_to(fmap)
        except Exception as exc:
            st.warning(f"Points de fraîcheur non chargés : {exc}")

    fmap.fit_bounds(bounds)
    return fmap


def plot_histogram(values, threshold, unit):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.hist(values, bins=60, color="#4C78A8", alpha=0.85, edgecolor="none")
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f" seuil ICU {threshold:.1f}{unit}",
            color="#333333", fontsize=9, va="top")
    ax.set_xlabel(f"Température ({unit})", fontsize=9, color="#555555")
    ax.set_ylabel("Pixels", fontsize=9, color="#555555")
    ax.tick_params(labelsize=8, colors="#777777")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CCCCCC")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("🌡️ Îlots de Chaleur Urbains — Nantes Métropole")
st.caption("Cartographie et analyse explicative des ICU à la résolution native "
           "Landsat (100 m). Modélisation : régression linéaire + LightGBM, "
           "interprétation SHAP.")

scenes = discover() if HAS_RASTERIO else {}

with st.sidebar:
    st.header("Données")
    if scenes:
        zone = st.selectbox("Zone", list(scenes))
        info = scenes[zone]
        layer_options = []
        if info["freq"]:
            layer_options.append("Fréquence ICU multi-dates")
        if info["delta"]:
            layer_options.append("Anomalie ΔLST (par date)")
        if info["raw"]:
            layer_options.append("LST Landsat brut (°C)")
        layer = st.radio("Couche affichée", layer_options)
        date = None
        if layer.startswith("Anomalie") or layer.startswith("LST"):
            dates = [d for d in info["dates"]
                     if (d in info["delta"] if layer.startswith("Anomalie")
                         else d in info["raw"])]
            if dates:
                date = st.selectbox("Date", dates)
            elif layer.startswith("Anomalie"):
                st.warning("Pas de carte d'anomalie pour cette zone "
                           "(lancer compute_icu.py).")
        demo = False
    else:
        if not HAS_RASTERIO:
            st.warning("`rasterio` absent : `pip install rasterio` pour de vrais GeoTIFF.")
        else:
            st.info(f"Aucun raster dans `{RAW_DIR}/` ou `{ICU_DIR}/`.\n"
                    f"Pipeline : `gee_extraction.py` → `compute_icu.py`")
        demo = st.toggle("Mode démo (synthétique)", value=True)
        layer = "Fréquence ICU multi-dates"
        zone, date = "démo", None

    st.header("Affichage")
    opacity = st.slider("Opacité de la couche", 0.2, 1.0, 0.7, 0.05)
    icu_delta = st.slider("Seuil ICU (ΔLST, °C)", 0.5, 6.0, 2.0, 0.5,
                          help="Un pixel est ICU si ΔLST dépasse ce seuil.")

if not scenes and not demo:
    st.stop()

# --- Sélection de la couche ---
if scenes:
    if layer.startswith("Fréquence"):
        path = info["freq"]
    elif layer.startswith("Anomalie"):
        path = info["delta"].get(date)
    else:  # LST brut
        path = info["raw"].get(date)
    if not path:
        st.error("Couche demandée indisponible pour cette zone/date.")
        st.stop()
    data, bounds = load_raster(path)
else:
    data, bounds = demo_scene()
    # Construire une pseudo-fréquence (3 dates synthétiques -> % fois en surchauffe)
    med = np.nanmedian(data)
    over = (data > med + icu_delta).astype(np.float32)
    if layer.startswith("Fréquence"):
        data = 100.0 * over
    elif layer.startswith("Anomalie"):
        data = data - med

valid = data[np.isfinite(data)]
if valid.size == 0:
    st.error("Raster vide (100 % nodata).")
    st.stop()

# --- Choix colormap / caption ---
if layer.startswith("Fréquence"):
    display = data
    cmap = "YlOrRd"
    vmin, vmax = 0.0, float(max(50.0, np.nanpercentile(valid, 99)))
    caption = "Fréquence ICU (% de dates en surchauffe)"
    threshold_abs = None
elif layer.startswith("Anomalie"):
    display = data
    span = float(np.nanpercentile(np.abs(display), 99))
    cmap, vmin, vmax = "RdBu_r", -span, span
    caption = "Écart à la médiane terrestre de la scène (°C)"
    threshold_abs = icu_delta
else:  # LST brut
    display = data
    vmin = float(np.nanpercentile(valid, 1))
    vmax = float(np.nanpercentile(valid, 99))
    cmap = "inferno"
    caption = "Température de surface (°C)"
    med = float(np.nanmedian(valid))
    threshold_abs = med + icu_delta

# --- Tuiles de synthèse ---
c1, c2, c3, c4 = st.columns(4)
if layer.startswith("Fréquence"):
    c1.metric("Fréquence moyenne", f"{np.nanmean(valid):.1f} %")
    c2.metric("Pixels > 50 % dates", f"{(valid > 50).mean() * 100:.1f} %")
    c3.metric("Maximum", f"{np.nanmax(valid):.0f} %")
    c4.metric("Zone", zone)
elif layer.startswith("Anomalie"):
    c1.metric("ΔLST médian", f"{np.nanmedian(valid):+.1f} °C")
    c2.metric("ΔLST p95", f"{np.nanpercentile(valid, 95):+.1f} °C")
    c3.metric(f"Surface ICU (Δ>{icu_delta:.1f} °C)",
              f"{(valid > icu_delta).mean() * 100:.1f} %")
    c4.metric("Scène", f"{zone} {date}")
else:
    c1.metric("LST médiane", f"{np.nanmedian(valid):.1f} °C")
    c2.metric("Maximum", f"{np.nanmax(valid):.1f} °C")
    c3.metric(f"Surface ICU (> méd. +{icu_delta:.1f} °C)",
              f"{(valid > threshold_abs).mean() * 100:.1f} %")
    c4.metric("Scène", f"{zone} {date}")

# --- Carte ---
fmap = build_map(display, bounds, cmap, vmin, vmax, opacity, caption,
                 points_path=POINTS_PATH)
st_folium(fmap, use_container_width=True, height=550, returned_objects=[])

# --- Distribution (sauf fréquence, peu pertinente) ---
if not layer.startswith("Fréquence"):
    st.subheader("Distribution")
    st.pyplot(plot_histogram(valid, threshold_abs, "°C"), use_container_width=True)
    plt.close("all")

# --- Panneau analyse SHAP (si metrics.json dispo) ---
st.divider()
st.subheader("Analyse explicative — SHAP")
metrics_path = os.path.join(EVAL_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, encoding="utf-8") as f:
        meta = json.load(f)
    shap_summary = os.path.join(EVAL_DIR, "shap_summary.png")
    if os.path.exists(shap_summary):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(shap_summary, caption="Importance SHAP (mean |shap|) — LightGBM")
        with col2:
            sh = meta.get("shap", {}).get("mean_abs_shap", {})
            st.markdown("**Top features (mean |SHAP|)**")
            for k, v in sorted(sh.items(), key=lambda kv: -kv[1])[:5]:
                st.metric(k, f"{v:.4f}")
            lgb_meta = meta.get("lightgbm", {})
            st.markdown(f"**LightGBM** R² = {lgb_meta.get('r2_global', 0):.3f}, "
                        f"RMSE = {lgb_meta.get('rmse_global', 0):.3f} °C")
            lin_meta = meta.get("linear", {})
            st.markdown(f"**Linéaire** R² = {lin_meta.get('r2_global', 0):.3f}, "
                        f"RMSE = {lin_meta.get('rmse_global', 0):.3f} °C")
            st.caption("Coefficients linéaires (°C par unité) : "
                       + ", ".join(f"{k}={v:+.2f}"
                                    for k, v in lin_meta.get("coeffs_°C_per_unit", {}).items()))
    else:
        st.info(f"`shap_summary.png` absent — relancer `model_evaluation.py`.")
else:
    st.info(f"`{EVAL_DIR}/metrics.json` absent — l'analyse explicative n'a pas été "
            f"lancée. Pipeline : `build_table.py` puis `model_evaluation.py`.")

# --- Encadré de mise en garde ---
st.divider()
with st.container():
    st.markdown("**⚠️ Mise en garde — lire avant toute interprétation**")
    st.caption(
        "• La carte montre une **température de surface** (LST Landsat), pas la "
        "température de l'air ressenti.\n"
        "• Acquisitions **matinales d'été par ciel clair** (~10h50 UTC, "
        " SUHI diurne) : ce n'est **pas** l'îlot de chaleur nocturne visé par "
        "les politiques de fraîcheur urbaine.\n"
        "• Résolution 100 m = échelle du **quartier**, pas de la rue. Les "
        "cœurs d'îlots (cours, rues étroites) ne sont pas résolus.\n"
        "• Validation **in situ** non réalisée à ce stade (cf. audit scientifique) ; "
        "résultats à visée exploratoire et pédagogique."
    )