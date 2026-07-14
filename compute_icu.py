"""Calcul des îlots de chaleur urbains (ICU) — projet ICU Nantes recentré.

Pour chaque date d'export Landsat 100 m (`Y_lst_100m_<zone>_<date>.tif` produit
par `gee_extraction.py --resolution 100`), on calcule :

  - l'**anomalie spatiale** ΔLST = LST − médiane spatiale (sur pixels terrestres
    valides). Référence = scène entière, ce qui élimine l'effet de la météo du
    jour et isole la structure thermique urbaine (zones plus chaudes que leur
    environnement métropolitain à ce moment — définition standard d'un ICU).
  - un **masque binaire ICU** (ΔLST > seuil, défaut +2 °C).

Puis on agrège les dates en une **carte de fréquence multi-dates** : pour
chaque pixel, fraction (en %) des dates où il est en surchauffe. C'est le
**livrable final** : bien plus robuste qu'une date unique (un parking peut être
froid un jour pluvieux, mais rester un ICU structurel sur 10 dates), et c'est
une vraie valeur ajoutée par rapport à une simple collection Landsat brute.

Optionnel : le masque d'eau (NDWI > 0.3 sur le X_s2_100m_*.tif compagnon)
exclut la Loire/l'Erdre des statistiques — les eaux sont froides par nature et
pollueraient la médiane terrestre.

Sorties (data/icu/) :
    delta_lst_<zone>_<date>.tif       anomalie spatiale (°C, peut être négative)
    icu_mask_<zone>_<date>.tif        masque binaire ICU (1 = en surchauffe)
    icu_frequency_<zone>.tif          fréquence multi-dates (0..100 %)

Vérification visuelle rapide (J1.4) : un résumé par scène affiche la médiane,
les 5e/95e percentiles, et la part de pixels ICU ; pour Nantes on doit voir la
Loire et l'Erdre froides (ΔLST < 0), les zones industrielles et grands parkings
chaudes (ΔLST > +2 °C).

Exemple :
    python compute_icu.py --raw-dir data/raw --out-dir data/icu --zone nantes_metropole
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
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


Y_PATTERN = re.compile(r"Y_lst_(?:100m_)?(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")
X_PATTERN = re.compile(r"X_s2_(?:100m_)?(?P<zone>.+?)_(?P<date>\d{4}-\d{2}-\d{2})\.tif$")


def find_pairs(raw_dir):
    """Liste (zone, date, y_path, x_path_or_None) pour tous les exports."""
    y_files, x_files = {}, {}
    for f in sorted(os.listdir(raw_dir)):
        m_y = Y_PATTERN.match(f)
        m_x = X_PATTERN.match(f)
        if m_y:
            y_files[(m_y.group("zone"), m_y.group("date"))] = os.path.join(raw_dir, f)
        elif m_x:
            x_files[(m_x.group("zone"), m_x.group("date"))] = os.path.join(raw_dir, f)
    pairs = []
    for key, y_path in y_files.items():
        x_path = x_files.get(key)
        pairs.append((key[0], key[1], y_path, x_path))
    return pairs


def read_raster(path):
    """Lit un GeoTIFF mono-bande en respectant les nodata -> NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        profile = src.profile
    return arr, profile


def read_band_aligned(path, band_name, ref_profile):
    """Lit une bande nommée d'un GeoTIFF multi-bandes, alignée sur `ref_profile`.
    Renvoie un tableau 2D (float32) ou None si la bande est absente."""
    with rasterio.open(path) as src:
        names = src.descriptions or [None] * src.count
        for i, n in enumerate(names, start=1):
            if n and n.upper() == band_name.upper():
                with WarpedVRT(src, crs=ref_profile["crs"],
                               transform=ref_profile["transform"],
                               width=ref_profile["width"],
                               height=ref_profile["height"],
                               resampling=Resampling.bilinear) as vrt:
                    return vrt.read(i, masked=True).filled(np.nan).astype(np.float32)
    return None


def water_mask(x_path, ref_profile, ndwi_threshold=0.3):
    """Masque d'eau (True = eau) sur la grille de référence, ou None si pas de X."""
    if x_path is None:
        return None
    ndwi = read_band_aligned(x_path, "NDWI", ref_profile)
    if ndwi is None:
        return None
    return ndwi > ndwi_threshold


def write_tif(path, arr, profile, dtype="float32"):
    prof = profile.copy()
    if arr.ndim == 2:
        arr = arr[None]
    prof.update(count=arr.shape[0], height=arr.shape[1], width=arr.shape[2],
                dtype=dtype, nodata=np.nan if dtype == "float32" else None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype))


def compute_anomaly(lst, valid_land):
    """ΔLST = LST − médiane spatiale des pixels terrestres valides (°C)."""
    ref = np.nanmedian(lst[valid_land])
    delta = lst - ref
    delta[~np.isfinite(lst)] = np.nan
    return delta, float(ref)


def summarize(zone, date, lst, delta, valid_land, water, threshold):
    n_valid = int(valid_land.sum())
    n_water = int(water.sum()) if water is not None else 0
    n_icu = int(np.nansum(delta[valid_land] > threshold))
    pct_icu = 100 * n_icu / n_valid if n_valid else float("nan")
    p05, p50, p95 = (np.nanpercentile(delta[valid_land], q) for q in (5, 50, 95))
    print(f"  [{zone} {date}] ref(méd)={np.nanmedian(lst[valid_land]):.1f}°C  "
          f"ΔLST p5={p05:+.1f}  p50={p50:+.1f}  p95={p95:+.1f}  "
          f"ICU={pct_icu:5.1f}% (n={n_icu}/{n_valid}; eau exclue: {n_water})")


def process_zone(zone, pairs, out_dir, threshold):
    zone_pairs = [(d, yp, xp) for (z, d, yp, xp) in pairs if z == zone]
    if not zone_pairs:
        print(f"  Aucune scène pour la zone '{zone}'.")
        return None

    os.makedirs(out_dir, exist_ok=True)
    accum_mask = None      # somme des masques ICU valides
    accum_valid = None     # somme des dates valides (non-nan après masque)
    template_profile = None
    water = None
    n_dates = 0

    for date, y_path, x_path in zone_pairs:
        lst, profile = read_raster(y_path)
        if template_profile is None:
            template_profile = profile
            water = water_mask(x_path, profile) if x_path else None
        finite = np.isfinite(lst)
        valid_land = finite.copy()
        if water is not None:
            valid_land &= ~water
        if valid_land.sum() < 50:
            print(f"  [{zone} {date}] trop peu de pixels terrestres valides, ignorée.")
            continue

        delta, ref = compute_anomaly(lst, valid_land)
        summarize(zone, date, lst, delta, valid_land, water, threshold)

        mask_icu = np.where(valid_land, (delta > threshold).astype(np.float32), np.nan)

        write_tif(os.path.join(out_dir, f"delta_lst_{zone}_{date}.tif"),
                  delta, profile)
        write_tif(os.path.join(out_dir, f"icu_mask_{zone}_{date}.tif"),
                  mask_icu, profile)

        valid_for_freq = np.where(valid_land, 1.0, np.nan)
        if accum_mask is None:
            accum_mask = np.where(valid_land, mask_icu, 0.0)
            accum_valid = valid_for_freq
        else:
            accum_mask = np.where(valid_land, accum_mask + mask_icu, accum_mask)
            accum_valid = np.where(valid_land, accum_valid + valid_for_freq,
                                   accum_valid)
        n_dates += 1

    if n_dates == 0:
        return None

    # Fréquence multi-dates (% de dates avec ICU) — normalisée par le nombre
    # de dates valides par pixel (gère les pixels masqués ponctuellement).
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = 100.0 * accum_mask / accum_valid
    freq = np.where(np.isfinite(accum_valid) & (accum_valid > 0), freq, np.nan)

    write_tif(os.path.join(out_dir, f"icu_frequency_{zone}.tif"),
              freq, template_profile)
    print(f"  -> Carte de fréquence ICU '{zone}' : {n_dates} date(s), "
          f"{np.nanmean(freq):.1f}% de pixels en surchauffe en moyenne.")
    return n_dates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw",
                    help="dossier d'entrée (exports GEE ; défaut: data/raw)")
    ap.add_argument("--out-dir", default="data/icu",
                    help="dossier de sortie (défaut: data/icu)")
    ap.add_argument("--zone", default=None,
                    help="limiter à une zone (défaut: toutes les zones trouvées)")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="seuil ΔLST pour ICU en °C (défaut: +2)")
    ap.add_argument("--no-water-mask", action="store_true",
                    help="désactiver le masque d'eau (NDWI > 0.3 sur X_s2)")
    ap.add_argument("--ndwi-threshold", type=float, default=0.3,
                    help="seuil NDWI pour l'eau (défaut: 0.3)")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        sys.exit(f"Dossier introuvable : {args.raw_dir}")

    pairs = find_pairs(args.raw_dir)
    if not pairs:
        sys.exit(f"Aucun Y_lst_*.tif trouvé dans {args.raw_dir}. "
                 "Lancer d'abord gee_extraction.py --resolution 100.")

    zones = sorted({z for (z, d, yp, xp) in pairs})
    if args.zone:
        if args.zone not in zones:
            sys.exit(f"Zone '{args.zone}' absente. Dispos : {zones}")
        zones = [args.zone]

    total = 0
    for zone in zones:
        print(f"\n=== Zone '{zone}' ===")
        n = process_zone(zone, pairs, args.out_dir, args.threshold)
        if n:
            total += n

    print(f"\nTerminé : {total} date(s) traitée(s), cartes -> {args.out_dir}/")


if __name__ == "__main__":
    main()