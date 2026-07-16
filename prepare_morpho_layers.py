"""Préparation des couches morphologiques (canopée, bâti) — projet ICU Nantes.

À partir du GeoJSON Open Data « Occupation du sol 2022 niveau 3 » de Nantes
Métropole (`data/morpho/source/ocs_2022_niveau3.geojson`, WGS84, champ
propriété `c_niveau` = code entier type CORINE Land Cover niveau 3), on
rasterise deux couches fractionnelles continues sur la **même grille 100 m**
que les exports Landsat/Sentinel-2 de `gee_extraction.py` (Lambert-93, ancrage
haut-gauche sur (xmin, ymax)) :

    data/morpho/canopee.tif   fraction 0..1 de canopée par cellule 100 m
    data/morpho/bati.tif      fraction 0..1 de bâti     par cellule 100 m

Méthode : sur-échantillonnage — on rasterise les polygones à une résolution
fine (cell_m / supersample, soit 10 m par défaut), puis on moyenne par blocs
(supersample x supersample) pour obtenir la fraction de couverture continue
dans chaque cellule 100 m. La fraction est définie partout (0 = absence, pas
donnée manquante) : pas de nodata.

Entrées : GeoJSON source (déjà présent localement, jamais téléchargé).
Sorties : canopee.tif et bati.tif (float32, EPSG:2154) dans --out-dir.

Exemple :
    python prepare_morpho_layers.py
    python prepare_morpho_layers.py --cell-m 100 --supersample 10
    python prepare_morpho_layers.py --bbox 335000 6671000 375000 6705000
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import numpy as np
import geopandas as gpd
import rasterio
import rasterio.transform
import rasterio.features
import pandas as pd

CANOPY_CODES = {311, 224}   # 311 = Bois et forêts, 224 = Sylviculture et peupleraies
BATI_CODES = {111, 112, 113, 114, 115, 121, 122, 123, 124}
# 111-115 = tissu résidentiel (centre-ville, hameau, habitat collectif/pavillonnaire/mixte)
# 121-124 = zones d'activités / commerces / services urbains / zones portuaires


def compute_fraction_layer(gdf, codes, height, width, supersample, fine_transform,
                          cell_m, xmin, ymax):
    """Rasterise les polygones de `gdf` dont `c_niveau` est dans `codes` à la
    résolution fine, puis aggrège en fraction 0..1 sur la grille 100 m.
    Renvoie (fraction float32 (height, width), n_polygones)."""
    f = gdf[pd.notna(gdf["c_niveau"])]
    code_int = pd.to_numeric(f["c_niveau"], errors="coerce")
    keep = code_int.notna() & code_int.astype(int).isin(codes)
    sub = f[keep]

    fine_h = height * supersample
    fine_w = width * supersample

    if len(sub) == 0:
        return np.zeros((height, width), dtype=np.float32), 0

    shapes = [(geom, 1) for geom in sub.geometry if geom is not None]
    if not shapes:
        return np.zeros((height, width), dtype=np.float32), 0

    arr = rasterio.features.rasterize(
        shapes,
        out_shape=(fine_h, fine_w),
        transform=fine_transform,
        fill=0,
        dtype="uint8",
    )
    assert arr.shape == (fine_h, fine_w), (
        f"forme raster fine inattendue: {arr.shape} != ({fine_h}, {fine_w})")

    arr = arr.reshape(height, supersample, width, supersample)
    frac = arr.mean(axis=(1, 3)).astype(np.float32)
    return frac, len(sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source",
                    default="data/morpho/source/ocs_2022_niveau3.geojson",
                    help="GeoJSON Open Data OCS 2022 niveau 3 (défaut: "
                         "data/morpho/source/ocs_2022_niveau3.geojson)")
    ap.add_argument("--out-dir", default="data/morpho",
                    help="dossier de sortie (défaut: data/morpho)")
    ap.add_argument("--bbox", nargs=4, type=float,
                    default=[335000, 6671000, 375000, 6705000],
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="emprise Lambert-93 (défaut: nantes_metropole)")
    ap.add_argument("--cell-m", type=int, default=100,
                    help="taille de cellule de sortie en mètres (défaut: 100)")
    ap.add_argument("--supersample", type=int, default=10,
                    help="facteur de sur-échantillonnage pour le calcul de "
                         "fraction (défaut: 10 -> résolution fine 10 m)")
    args = ap.parse_args()

    if not os.path.exists(args.source):
        sys.exit(f"Fichier source introuvable : {args.source}")

    xmin, ymin, xmax, ymax = args.bbox
    cell_m = args.cell_m
    supersample = args.supersample
    width = int((xmax - xmin) // cell_m)
    height = int((ymax - ymin) // cell_m)
    transform = rasterio.transform.from_origin(xmin, ymax, cell_m, cell_m)

    fine_m = cell_m / supersample
    fine_transform = rasterio.transform.from_origin(xmin, ymax, fine_m, fine_m)

    print(f"Chargement du GeoJSON : {args.source}")
    gdf = gpd.read_file(args.source)
    print(f"  {len(gdf)} feature(s) chargée(s) en CRS {gdf.crs}")
    gdf = gdf.to_crs("EPSG:2154")
    print("  Reprojeté en EPSG:2154")

    print(f"Grille de sortie : {width}x{height} cellules de {cell_m} m "
          f"(suréchantillonnage x{supersample} -> {fine_m} m)")

    os.makedirs(args.out_dir, exist_ok=True)

    groups = [
        ("canopee", CANOPY_CODES, "canopee.tif"),
        ("bati", BATI_CODES, "bati.tif"),
    ]

    summary = {}
    for name, codes, fname in groups:
        print(f"\nRasterisation groupe '{name}' ({len(codes)} codes)...")
        frac, n_pol = compute_fraction_layer(
            gdf, codes, height, width, supersample, fine_transform,
            cell_m, xmin, ymax)
        out_path = os.path.join(args.out_dir, fname)
        with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:2154",
                transform=transform) as dst:
            dst.write(frac.astype(np.float32), 1)
        print(f"  {n_pol} polygone(s) utilisé(s) -> {out_path}")
        print(f"  fraction moyenne sur la grille : {frac.mean():.4f}")
        summary[name] = (n_pol, float(frac.mean()))

    print("\nRésumé final :")
    for name, (n_pol, mean_frac) in summary.items():
        print(f"  {name:8s} : {n_pol:6d} polygone(s)  |  fraction moyenne = "
              f"{mean_frac:.4f}")
    print("\nTerminé.")


if __name__ == "__main__":
    main()