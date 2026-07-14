"""Prépare le dataset d'entraînement à partir des exports GEE téléchargés.

Entrée  : data/raw/ contenant les paires produites par gee_extraction.py
          (X_s2_<zone>_<date>.tif à 5 bandes, Y_lst_<zone>_<date>.tif à 1 bande),
          plus, en option, les couches morphologiques Open Data Nantes Métropole
          rasterisées en Lambert-93 (canopée, densité bâti) — n'importe quelle
          grille/emprise couvrant la zone, elles sont reprojetées à la volée.

Sortie  : out_dir/
            X/ -> patchs 256x256 à 7 canaux [B4, B3, B2, NDVI, NDWI, Canopée, Bâti]
            Y/ -> patchs 256x256 à 1 canal (LST °C), mêmes noms de fichiers
            full/ -> X_full_<zone>_<date>.tif : stack 7 canaux pleine zone
                     (utilisé par predict.py pour l'inférence)

Exemple :
    python prepare_patches.py --raw-dir data/raw --out-dir data/patches \
        --canopee data/morpho/canopee_10m.tif --bati data/morpho/bati_10m.tif
"""

import argparse
import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform

X_BAND_NAMES = ["B4", "B3", "B2", "NDVI", "NDWI", "Canopee", "Bati"]


def read_aligned_layer(path, ref):
    """Lit un raster quelconque reprojeté/rééchantillonné sur la grille du raster `ref`."""
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                       width=ref.width, height=ref.height,
                       resampling=Resampling.bilinear) as vrt:
            return vrt.read(1).astype(np.float32)


def build_full_stack(x_path, canopee_path, bati_path):
    """Stack 7 canaux pleine zone + profil rasterio du raster de référence."""
    with rasterio.open(x_path) as src:
        profile = src.profile.copy()
        s2 = src.read(masked=True).filled(np.nan).astype(np.float32)  # (5, H, W)
        h, w = src.height, src.width

        extra = []
        for name, path in (("canopée", canopee_path), ("bâti", bati_path)):
            if path:
                extra.append(read_aligned_layer(path, src))
            else:
                print(f"  ATTENTION : couche {name} absente -> canal rempli de zéros")
                extra.append(np.zeros((h, w), dtype=np.float32))

    stack = np.concatenate([s2, np.stack(extra)], axis=0)  # (7, H, W)
    profile.update(count=len(X_BAND_NAMES), dtype="float32", nodata=np.nan)
    return stack, profile


def write_tif(path, array, profile, transform=None, band_names=None):
    prof = profile.copy()
    if array.ndim == 2:
        array = array[None]
    prof.update(count=array.shape[0], height=array.shape[1], width=array.shape[2],
                dtype="float32", nodata=np.nan)
    if transform is not None:
        prof.update(transform=transform)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(np.float32))
        if band_names:
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/patches")
    ap.add_argument("--canopee", default=None, help="GeoTIFF canopée (Open Data rasterisé)")
    ap.add_argument("--bati", default=None, help="GeoTIFF densité/hauteur du bâti")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--min-valid", type=float, default=0.8,
                    help="fraction minimale de pixels valides pour garder un patch (défaut: 0.8)")
    args = ap.parse_args()

    for sub in ("X", "Y", "full"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    pairs = []
    for f in sorted(os.listdir(args.raw_dir)):
        m = re.match(r"X_s2_(.+)\.tif$", f)
        if m and os.path.exists(os.path.join(args.raw_dir, f"Y_lst_{m.group(1)}.tif")):
            pairs.append(m.group(1))
    if not pairs:
        raise SystemExit(f"Aucune paire X_s2_*/Y_lst_* trouvée dans {args.raw_dir}")

    ps = args.patch_size
    total = 0
    for key in pairs:
        print(f"Paire '{key}' :")
        x_path = os.path.join(args.raw_dir, f"X_s2_{key}.tif")
        y_path = os.path.join(args.raw_dir, f"Y_lst_{key}.tif")

        x_stack, profile = build_full_stack(x_path, args.canopee, args.bati)
        with rasterio.open(y_path) as src:
            y = src.read(1, masked=True).filled(np.nan).astype(np.float32)

        full_path = os.path.join(args.out_dir, "full", f"X_full_{key}.tif")
        write_tif(full_path, x_stack, profile, band_names=X_BAND_NAMES)

        h, w = y.shape
        kept = 0
        for row in range(0, h - ps + 1, ps):
            for col in range(0, w - ps + 1, ps):
                x_patch = x_stack[:, row:row + ps, col:col + ps]
                y_patch = y[row:row + ps, col:col + ps]
                valid = min(np.isfinite(x_patch).mean(), np.isfinite(y_patch).mean())
                if valid < args.min_valid:
                    continue
                name = f"{key}_r{row}_c{col}.tif"
                tfm = window_transform(Window(col, row, ps, ps), profile["transform"])
                write_tif(os.path.join(args.out_dir, "X", name), x_patch, profile,
                          transform=tfm, band_names=X_BAND_NAMES)
                write_tif(os.path.join(args.out_dir, "Y", name), y_patch, profile,
                          transform=tfm)
                kept += 1
        total += kept
        print(f"  {kept} patch(s) {ps}x{ps} conservés, stack pleine zone -> {full_path}")

    print(f"\nTerminé : {total} paires de patchs dans {args.out_dir}/X et /Y")


if __name__ == "__main__":
    main()
