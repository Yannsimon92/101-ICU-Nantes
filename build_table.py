"""Construit le DataFrame tabulaire (1 ligne = 1 pixel × 1 date) pour l'analyse
explicative des ICU Nantes (J2.1, roadmap recentrée).

Entrée  : data/raw/ contenant les paires GEE 100 m :
              X_s2_100m_<zone>_<date>.tif   (bandes : NDVI, NDWI, [+ NDBI])
              Y_lst_100m_<zone>_<date>.tif  (bande : LST °C)
          + optionnellement les couches morphologiques Open Data Nantes
          Métropole rasterisées en Lambert-93 (canopée, bâti) — n'importe quelle
          grille/emprise couvrant la zone, elles sont reprojetées à la volée sur
          la grille 100 m de chaque scène via WarpedVRT (rééchantillonnage moyen).

Sortie  : un Parquet (data/table.parquet) avec une ligne par pixel×date :
            zone, date, bloc_id (pour split spatial), x_px, y_px,
            lst_c, delta_lst_c (anomalie spatiale vs médiane terrestre),
            ndvi, ndwi, ndbi (NaN si absent), canopee, bati, tissu

Hygiène :
  - Filtrage des pixels d'eau (NDWI > seuil), non inclus dans la médiane ni
    dans le tableau final (l'eau est froide par nature, polluerait la régression).
  - Masque des LST NaN/invalides.
  - `bloc_id` = (y_px // BLOC_PX, x_px // BLOC_PX) pour split spatial par blocs
    (~2×2 km par défaut à 100 m, soit BLOC_PX=20).

Exemple :
    python build_table.py --raw-dir data/raw --out data/table.parquet \
        --canopee data/morpho/canopee.tif --bati data/morpho/bati.tif
"""

import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

Y_PATTERN = re.compile(r"Y_lst_(?:100m_)?(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")
X_PATTERN = re.compile(r"X_s2_(?:100m_)?(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")

BLOC_PX_DEFAULT = 20          # 20 pixels * 100 m = 2 km de côté pour le bloc
WATER_NDWI_DEFAULT = 0.30     # au-dessus = eau (Loire, Erdre)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_pairs(raw_dir):
    y_files, x_files = {}, {}
    for f in sorted(os.listdir(raw_dir)):
        m_y = Y_PATTERN.match(f)
        m_x = X_PATTERN.match(f)
        if m_y:
            y_files[(m_y.group("zone"), m_y.group("date"))] = os.path.join(raw_dir, f)
        elif m_x:
            x_files[(m_x.group("zone"), m_x.group("date"))] = os.path.join(raw_dir, f)
    return [(z, d, y_files[(z, d)], x_files.get((z, d)))
            for (z, d) in sorted(y_files)]


def read_band_aligned(path, band_name, ref):
    """Lit une bande nommée alignée sur le raster `ref` (transform, crs)."""
    with rasterio.open(path) as src:
        names = src.descriptions or [None] * src.count
        for i, n in enumerate(names, start=1):
            if n and n.upper() == band_name.upper():
                with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                               width=ref.width, height=ref.height,
                               resampling=Resampling.bilinear) as vrt:
                    return vrt.read(i, masked=True).filled(np.nan).astype(np.float32)
    return None


def read_layer_aligned(path, ref, resampling):
    """Lit un raster mono-bande reprojeté sur la grille de `ref`."""
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                       width=ref.width, height=ref.height,
                       resampling=resampling) as vrt:
            return vrt.read(1, masked=True).filled(np.nan).astype(np.float32)


# ---------------------------------------------------------------------------
# Tissu urbain (audit Sci §6, recyclé)
# ---------------------------------------------------------------------------

def classify_tissu(canopy, bati, ndwi):
    """Classification simple par seuils — 5 classes, interpétables.
    canopy/bati supposés en [0..1] (fraction de surface), ndwi en [-1..1]."""
    t = np.full(canopy.shape, "autre", dtype=object)
    t[ndwi > 0.30] = "eau"
    t[(bati >= 0.60) & (canopy < 0.10) & (ndwi <= 0.30)] = "dense"
    t[(bati >= 0.40) & (canopy < 0.20) & (ndwi <= 0.30) & (t == "autre")] = "industriel"
    t[(bati >= 0.10) & (canopy >= 0.10) & (canopy < 0.30)
      & (bati < 0.40) & (ndwi <= 0.30) & (t == "autre")] = "pavillonnaire"
    t[(canopy >= 0.30) & (bati < 0.10) & (ndwi <= 0.30) & (t == "autre")] = "végétal"
    return t


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def scene_to_records(zone, date, y_path, x_path, canopee_path, bati_path,
                     bloc_px, ndwi_water):
    """Construit les enregistrements (liste de dicts) pour une scène."""
    with rasterio.open(y_path) as ysrc:
        lst = ysrc.read(1, masked=True).filled(np.nan).astype(np.float32)
        profile = ysrc.profile
        H, W = lst.shape

        canopee = (read_layer_aligned(canopee_path, ysrc, Resampling.average)
                   if canopee_path else np.zeros((H, W), dtype=np.float32))
        bati = (read_layer_aligned(bati_path, ysrc, Resampling.average)
                if bati_path else np.zeros((H, W), dtype=np.float32))

        ndvi = ndbi = None
        if x_path is not None:
            ndvi = read_band_aligned(x_path, "NDVI", ysrc)
            ndwi = read_band_aligned(x_path, "NDWI", ysrc)
            ndbi = read_band_aligned(x_path, "NDBI", ysrc)
        if ndvi is None: ndvi = np.full((H, W), np.nan, dtype=np.float32)
        if ndwi is None: ndwi = np.full((H, W), np.nan, dtype=np.float32)
        if ndbi is None: ndbi = np.full((H, W), np.nan, dtype=np.float32)

    # Anomalie spatiale : ΔLST = LST − médiane des pixels terrestres valides
    valid_lst = np.isfinite(lst)
    land = valid_lst & (ndwi <= ndwi_water)
    if land.sum() < 50:
        return []
    ref = float(np.nanmedian(lst[land]))
    delta = lst - ref
    delta[~valid_lst] = np.nan

    tissu = classify_tissu(canopee, bati, ndwi)
    keep = valid_lst & (tissu != "eau") & (tissu != "autre")
    keep_idx = np.where(keep)
    if keep_idx[0].size == 0:
        return []

    bloc_r = keep_idx[0] // bloc_px
    bloc_c = keep_idx[1] // bloc_px
    bloc_id = [f"{zone}-{r}-{c}" for (r, c) in zip(bloc_r, bloc_c)]

    df = pd.DataFrame({
        "zone": zone, "date": date,
        "bloc_id": bloc_id,
        "y_px": keep_idx[0], "x_px": keep_idx[1],
        "lst_c": lst[keep].astype(np.float32),
        "delta_lst_c": delta[keep].astype(np.float32),
        "ndvi": ndvi[keep].astype(np.float32),
        "ndwi": ndwi[keep].astype(np.float32),
        "ndbi": ndbi[keep].astype(np.float32),
        "canopee": canopee[keep].astype(np.float32),
        "bati": bati[keep].astype(np.float32),
        "tissu": tissu[keep],
        "scene_ref_c": ref,
    })
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw",
                    help="dossier d'exports GEE (défaut: data/raw)")
    ap.add_argument("--out", default="data/table.parquet",
                    help="fichier de sortie (parquet recommandé ; défaut: data/table.parquet)")
    ap.add_argument("--canopee", default=None,
                    help="GeoTIFF canopée Open Data (fraction 0..1)")
    ap.add_argument("--bati", default=None,
                    help="GeoTIFF bâti Open Data (fraction 0..1)")
    ap.add_argument("--bloc-px", type=int, default=BLOC_PX_DEFAULT,
                    help=f"taille de bloc en pixels pour split spatial (défaut: {BLOC_PX_DEFAULT} = 2 km)")
    ap.add_argument("--ndwi-water", type=float, default=WATER_NDWI_DEFAULT,
                    help=f"seuil NDWI eau (défaut: {WATER_NDWI_DEFAULT})")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        sys.exit(f"Dossier introuvable : {args.raw_dir}")
    pairs = find_pairs(args.raw_dir)
    if not pairs:
        sys.exit(f"Aucune paire X/Y 100 m trouvée dans {args.raw_dir}. "
                 "Lancer d'abord gee_extraction.py --resolution 100.")

    if not args.canopee or not args.bati:
        print("  ATTENTION : canopée et/ou bâti absents -> colonnes à 0, tissu "
              "non classable au-delà de l'eau. L'analyse explicative perdra les "
              "facteurs morphologiques (cœur de l'étude).")

    frames = []
    for zone, date, y_path, x_path in pairs:
        print(f"  Scene {zone} {date} ...")
        rec = scene_to_records(zone, date, y_path, x_path,
                               args.canopee, args.bati,
                               args.bloc_px, args.ndwi_water)
        if isinstance(rec, pd.DataFrame) and not rec.empty:
            frames.append(rec)
        else:
            print(f"    -> aucune ligne valide, ignorée")
    if not frames:
        sys.exit("Aucune donnée exploitable.")

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.out.endswith(".parquet"):
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"\nTerminé : {len(df):,} lignes (pixels×dates), "
          f"{df['bloc_id'].nunique()} blocs uniques, "
          f"{df['date'].nunique()} dates, "
          f"{df['zone'].nunique()} zone(s).")
    print(f"  Composition tissu : "
          + ", ".join(f"{t}={int(n):,}"
                      for t, n in df["tissu"].value_counts().items()))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()