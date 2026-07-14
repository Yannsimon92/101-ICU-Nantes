"""Application Streamlit de visualisation des ICU de Nantes.

Superpose les cartes LST (prédiction 10 m, Landsat 100 m, anomalie ICU) sur un
fond CartoDB Positron, avec statistiques et distribution des températures.

Lancement :
    streamlit run app.py

Sources affichées :
    data/predictions/pred_lst_<zone>_<date>.tif  (sortie de predict.py)
    data/raw/Y_lst_<zone>_<date>.tif             (export GEE, vérité 100 m)
Sans fichier disponible, un mode démo sur données synthétiques est proposé.
"""

import glob
import os
import re

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

PRED_DIR = "data/predictions"
RAW_DIR = "data/raw"
# Emprise approx. du centre de Nantes pour le mode démo (sud-ouest, nord-est)
DEMO_BOUNDS = [[47.19, -1.60], [47.24, -1.52]]

st.set_page_config(page_title="ICU Nantes", page_icon="🌡️", layout="wide")


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def list_scenes():
    """{clé zone_date: {"pred": path|None, "raw": path|None}}"""
    scenes = {}
    for path in glob.glob(os.path.join(PRED_DIR, "pred_lst_*.tif")):
        key = re.sub(r"^pred_lst_|\.tif$", "", os.path.basename(path))
        scenes.setdefault(key, {})["pred"] = path
    for path in glob.glob(os.path.join(RAW_DIR, "Y_lst_*.tif")):
        key = re.sub(r"^Y_lst_|\.tif$", "", os.path.basename(path))
        scenes.setdefault(key, {})["raw"] = path
    return dict(sorted(scenes.items()))


@st.cache_data(show_spinner="Lecture du raster…")
def load_raster(path):
    """Retourne (array 2D avec NaN, bounds folium [[S,W],[N,E]])."""
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return data, [[south, west], [north, east]]


@st.cache_data
def demo_raster(seed=0):
    """LST synthétique : gradient urbain + noyaux chauds, pour tester l'app sans données."""
    rng = np.random.default_rng(seed)
    h, w = 512, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lst = 24 + 4 * np.exp(-(((xx - w / 2) / (w / 3)) ** 2 + ((yy - h / 2) / (h / 3)) ** 2))
    for _ in range(8):  # îlots de chaleur localisés
        cx, cy, amp = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(2, 5)
        lst += amp * np.exp(-(((xx - cx) / 40) ** 2 + ((yy - cy) / 40) ** 2))
    lst += rng.normal(0, 0.3, (h, w)).astype(np.float32)
    return lst, DEMO_BOUNDS


# ---------------------------------------------------------------------------
# Rendu carte
# ---------------------------------------------------------------------------

def colorize(data, cmap_name, vmin, vmax):
    """Array 2D -> image RGBA (NaN transparents)."""
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    rgba = matplotlib.colormaps[cmap_name](norm(np.nan_to_num(data, nan=vmin)))
    rgba[..., 3] = np.where(np.isfinite(data), 1.0, 0.0)
    return rgba


def build_map(data, bounds, cmap_name, vmin, vmax, opacity, caption):
    import branca.colormap as bcm

    center = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    fmap = folium.Map(location=center, tiles="CartoDB positron", zoom_start=13)
    folium.raster_layers.ImageOverlay(
        image=colorize(data, cmap_name, vmin, vmax),
        bounds=bounds,
        opacity=opacity,
        origin="upper",
    ).add_to(fmap)

    steps = matplotlib.colormaps[cmap_name](np.linspace(0, 1, 12))
    legend = bcm.LinearColormap(
        [matplotlib.colors.to_hex(c) for c in steps], vmin=vmin, vmax=vmax, caption=caption
    )
    legend.add_to(fmap)
    fmap.fit_bounds(bounds)
    return fmap


def plot_histogram(values, threshold, unit):
    """Distribution mono-série : pas de légende, grille discrète, seuil étiqueté."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
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
st.caption("Super-résolution guidée de la température de surface : "
           "Landsat 8/9 (100 m) + Sentinel-2 & morphologie urbaine (10 m)")

scenes = list_scenes() if HAS_RASTERIO else {}

with st.sidebar:
    st.header("Données")
    if scenes:
        scene_key = st.selectbox("Scène (zone_date)", list(scenes))
        available = scenes[scene_key]
        layer_options = []
        if available.get("pred"):
            layer_options += ["Prédiction 10 m", "Anomalie ICU (Δ moyenne)"]
        if available.get("raw"):
            layer_options += ["Landsat brut 100 m"]
        layer = st.radio("Couche affichée", layer_options)
        demo = False
    else:
        if not HAS_RASTERIO:
            st.warning("`rasterio` n'est pas installé : `pip install rasterio` "
                       "pour afficher de vrais GeoTIFF.")
        else:
            st.info(f"Aucun raster trouvé dans `{PRED_DIR}/` ni `{RAW_DIR}/`. "
                    "Pipeline : gee_extraction.py → prepare_patches.py → "
                    "train.py → predict.py")
        demo = st.toggle("Mode démo (données synthétiques)", value=True)
        layer = st.radio("Couche affichée", ["Prédiction 10 m", "Anomalie ICU (Δ moyenne)"])
        scene_key = "démo synthétique"

    st.header("Affichage")
    opacity = st.slider("Opacité de la couche", 0.2, 1.0, 0.7, 0.05)
    icu_delta = st.slider("Seuil ICU : écart à la moyenne (°C)", 0.5, 6.0, 2.0, 0.5,
                          help="Un pixel est classé ICU si sa température dépasse "
                               "la moyenne de la zone de ce delta.")

if not scenes and not demo:
    st.stop()

# --- Chargement de la couche sélectionnée ---
if scenes:
    if layer == "Landsat brut 100 m":
        data, bounds = load_raster(scenes[scene_key]["raw"])
    else:
        data, bounds = load_raster(scenes[scene_key]["pred"])
else:
    data, bounds = demo_raster()

valid = data[np.isfinite(data)]
if valid.size == 0:
    st.error("Raster vide (100 % nodata).")
    st.stop()

mean_t = float(valid.mean())
threshold_abs = mean_t + icu_delta
icu_fraction = float((valid > threshold_abs).mean())

if layer.startswith("Anomalie"):
    display = data - mean_t
    span = float(np.nanpercentile(np.abs(display), 99))
    # Divergent : bleu (plus frais) -> gris neutre à 0 -> rouge (plus chaud)
    cmap, vmin, vmax = "RdBu_r", -span, span
    caption = "Écart à la température moyenne de la zone (°C)"
else:
    display = data
    vmin = float(np.nanpercentile(valid, 1))
    vmax = float(np.nanpercentile(valid, 99))
    cmap = "inferno"  # séquentiel perceptuellement uniforme, sûr pour daltonisme
    caption = "Température de surface (°C)"

# --- Tuiles de synthèse ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Température moyenne", f"{mean_t:.1f} °C")
c2.metric("Maximum", f"{valid.max():.1f} °C")
c3.metric(f"Surface en ICU (> moy. +{icu_delta:.1f} °C)", f"{icu_fraction * 100:.1f} %")
c4.metric("Scène", scene_key)

# --- Carte ---
fmap = build_map(display, bounds, cmap, vmin, vmax, opacity, caption)
st_folium(fmap, use_container_width=True, height=550, returned_objects=[])

# --- Distribution ---
st.subheader("Distribution des températures")
st.pyplot(plot_histogram(valid, threshold_abs, "°C"), use_container_width=True)
plt.close("all")

st.caption("Fond de carte : CartoDB Positron. Les pixels au-delà du seuil ICU "
           "correspondent aux zones prioritaires de rafraîchissement urbain "
           "(à croiser avec l'Open Data « points de fraîcheur » de Nantes Métropole).")
