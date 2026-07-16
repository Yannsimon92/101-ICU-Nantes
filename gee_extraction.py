"""Extraction Google Earth Engine — Projet ICU Nantes (v2 recentrée).

Deux modes d'export sélectionnés par --resolution :

  --resolution 100 (défaut, projet recentré) :
    Pour chaque date Landsat estivale claire au-dessus de la zone, exporte deux
    GeoTIFF alignés sur la **même grille 100 m** (Lambert-93, ancrée sur
    l'emprise) :
      - Y_lst_100m_<zone>_<date>.tif : LST Landsat 8/9 (°C) à 100 m natif.
      - X_s2_100m_<zone>_<date>.tif : features Sentinel-2 agrégées à 100 m par
        moyenne par cellule (reduceResolution) : NDVI, NDWI (+ NDBI si
        --with-ndbi, nécessite le SWIR B11).
    Les couches morphologiques (canopée, bâti) de l'Open Data Nantes Métropole
    sont fusionnées **localement** par `prepare_patches.py` / `build_table.py`
    sur la même grille 100 m (on garde l'Open Data en lecture, jamais uploadé
    sur GEE).

  --resolution 10 (extension U-Net, voir /extension) :
    Exporte les 5 bandes S2 (B4,B3,B2,NDVI,NDWI) à 10 m et la LST à 100 m
    rééchantillonnée en plus-proche-voisin sur la même grille 10 m (v1 : tape de
    marching antérieure pour le U-Net guidé).

Hygiène (audit Sci §8 et roadmap J1.1) :
  - --years 2022 2025 par défaut (2026 incomplet, nuisible à la reproductibilité).
  - Filtre nuages **par emprise** : couverture nuageuse évaluée par reduceRegion
    sur QA_PIXEL (Landsat C2 L2) dans l'emprise, et non via la propriété scène
    (qui masque les dates où les nuages sont restés hors de la zone et inverse-
    ment). S2 garde son masque SCL par pixel + filtre scène sur
    CLOUDY_PIXEL_PERCENTAGE en pré-filtre grossier.
    - **Manifeste JSON par zone** : system:index des scènes Landsat utilisées,
      bbox snappée, arguments CLI, IDs de tâches d'export, dates — écrit dans
      `manifests/manifest_<zone>_<AAAAMMJJ-HHMMSS>.json`.
    - **Mosaïque temporelle** : pour les dates dont l'AOI est à cheval sur
      plusieurs scènes/paths Landsat, une mosaïque est construite avec une
      fenêtre de +/-`mosaic_window_days` (priorité au jour cible) ; la fraction
      de couverture de l'emprise est vérifiée avant export (`min_aoi_coverage`).
      Les champs `aoi_coverage_fraction` et `landsat_mosaic_scene_ids` sont
      ajoutés au manifeste.

Prérequis :
    pip install earthengine-api
    earthengine authenticate

Exemples :
    python gee_extraction.py --list-zones
    python gee_extraction.py --zone nantes_centre --dry-run
    python gee_extraction.py --zone nantes_metropole --resolution 100 --project mon-projet
    python gee_extraction.py --zone all --resolution 100 --with-ndbi --project mon-projet
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import ee

CRS = "EPSG:2154"          # Lambert-93, comme l'Open Data Nantes Métropole
SCALE = 10                 # résolution fine (mode 10 m)
HUNDRED_M = 100            # résolution native Landsat thermal (mode 100 m)
PATCH_M = 256 * SCALE      # 2560 m : emprises 10 m alignées sur des patchs 256 px

QA_CLOUD_BITS = 0b11110    # QA_PIXEL bits 1..4 : cirrus dilaté, cirrus, nuage, ombre
MAX_PIXELS_REDUCE = 1e10


# Emprises en Lambert-93 (xmin, ymin, xmax, ymax), approximatives — à ajuster
# librement, elles sont re-snappées sur la grille au chargement.
ZONES = {
    "nantes_centre":    (353000, 6687500, 358000, 6692500),
    "ile_de_nantes":    (353500, 6684500, 358500, 6688000),
    "nantes_nord":      (352000, 6692000, 358000, 6697000),
    "saint_herblain":   (346500, 6687000, 352500, 6692000),
    "reze_sud_loire":   (353000, 6680000, 358500, 6685000),
    # L'ensemble de la métropole : gros export (~4000 x 3400 px à 10 m).
    "nantes_metropole": (335000, 6671000, 375000, 6705000),
}


def snap_bbox(bbox, cell):
    """Étend l'emprise aux multiples de `cell` pour un ancrage propre."""
    xmin, ymin, xmax, ymax = bbox
    xmin = (xmin // cell) * cell
    ymin = (ymin // cell) * cell
    xmax = ((xmax + cell - 1) // cell) * cell
    ymax = ((ymax + cell - 1) // cell) * cell
    return int(xmin), int(ymin), int(xmax), int(ymax)


def region_of(bbox):
    return ee.Geometry.Rectangle(list(bbox), proj=CRS, geodesic=False)


# ---------------------------------------------------------------------------
# Landsat 8/9 Collection 2 Level 2
# ---------------------------------------------------------------------------

def landsat_collection(region, year_range, month_range, max_scene_cloud):
    """Collection Landsat 8/9 C2 L2 filtrée par dates/mois et un pré-filtre
    grossier scène-entier (CLOUD_COVER) ; le filtre fin par emprise est appliqué
    ensuite via `add_cloud_cover_local` + filter sur `cloud_cover_local`."""
    merged = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")))
    return (merged
            .filterBounds(region)
            .filter(ee.Filter.calendarRange(year_range[0], year_range[1], "year"))
            .filter(ee.Filter.calendarRange(month_range[0], month_range[1], "month"))
            .filter(ee.Filter.lt("CLOUD_COVER", max_scene_cloud)))


def to_lst(img):
    """ST_B10 -> °C (facteurs officiels Collection 2), masqué des nuages/ombres."""
    qa = img.select("QA_PIXEL")
    clear = qa.bitwiseAnd(QA_CLOUD_BITS).eq(0)
    lst = (img.select("ST_B10")
           .multiply(0.00341802).add(149.0)
           .subtract(273.15)
           .rename("LST"))
    return ee.Image(lst.updateMask(clear)
                    .copyProperties(img, ["system:time_start", "system:index"]))


def add_cloud_cover_local(img, region):
    """Ajoute la propriété `cloud_cover_local` (%) = 1 - fraction de pixels clairs
    de QA_PIXEL dans l'emprise. Approximation de la mesure terrain, indépendante
    de la propriété scène-entière CLOUD_COVER."""
    qa = img.select("QA_PIXEL")
    clear = qa.bitwiseAnd(QA_CLOUD_BITS).eq(0)            # 1 = clair, 0 = nuageux
    clear_frac = (clear.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=100,
        maxPixels=MAX_PIXELS_REDUCE,
        bestEffort=True,
    ).get("QA_PIXEL"))
    cloud_pct = ee.Number(1).subtract(ee.Number(clear_frac)).multiply(100)
    return img.set("cloud_cover_local", cloud_pct)


def landsat_lst_clear(region, year_range, month_range,
                      max_scene_cloud, max_cloud_local):
    """Collection Landsat filtrée par emprise + scènes -> LST (°C) masquée."""
    coll = landsat_collection(region, year_range, month_range, max_scene_cloud)
    coll = coll.map(lambda img: add_cloud_cover_local(img, region))
    coll = coll.filter(ee.Filter.lte("cloud_cover_local", max_cloud_local))
    return coll.map(to_lst)


def build_ymosaic(lst_coll, day, window_days, region, cell_m):
    """Mosaique les scenes LST (deja filtrees nuages par scene) dans une
    fenetre +/- window_days autour de `day`, triees par proximite au jour
    cible (mosaic() garde le pixel de la premiere image non masquee de la
    collection -> les scenes du jour cible sont prioritaires, les scenes
    voisines ne comblent que les trous de couverture).

    Renvoie (mosaic_image_float32, coverage_fraction, scene_ids_utilises)."""
    start = day.advance(-window_days, "day")
    end = day.advance(window_days + 1, "day")
    window_coll = lst_coll.filterDate(start, end)

    def _add_day_diff(img):
        diff = ee.Number(img.get("system:time_start")).subtract(
            day.millis()).abs()
        return img.set("_day_diff", diff)

    window_coll = window_coll.map(_add_day_diff).sort("_day_diff")
    mosaic_img = window_coll.mosaic().toFloat()

    coverage = mosaic_img.mask().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=cell_m,
        maxPixels=MAX_PIXELS_REDUCE,
        bestEffort=True,
    ).get("LST")
    coverage = ee.Number(coverage).getInfo() if coverage is not None else 0.0

    scene_ids = window_coll.aggregate_array("system:index").getInfo() or []
    return mosaic_img, float(coverage or 0.0), scene_ids


# ---------------------------------------------------------------------------
# Sentinel-2 SR harmonisé -> composite médian masqué (SCL)
# ---------------------------------------------------------------------------

def s2_collection(region, center_date, window_days, max_scene_cloud):
    def mask_and_scale(img):
        scl = img.select("SCL")
        # SCL : 3 = ombre, 8/9 = nuages, 10 = cirrus
        clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
        return img.updateMask(clear).divide(10000)
    start = center_date.advance(-window_days, "day")
    end = center_date.advance(window_days + 1, "day")
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud))
            .map(mask_and_scale))


def build_x_stack_10m(s2_median):
    """Stack 5 bandes à 10 m : B4, B3, B2, NDVI, NDWI (mode U-Net hérité)."""
    ndvi = s2_median.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2_median.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return (s2_median.select(["B4", "B3", "B2"])
            .addBands([ndvi, ndwi])
            .toFloat())


def build_x_stack_100m(s2_median, with_ndbi, native_proj):
    """Stack d'indices Sentinel-2 agrégés à 100 m par moyenne par cellule
    (reduceResolution) : NDVI, NDWI (+ NDBI si demandé).

    L'agrégation se fait sur le médian 10 m déjà masqué SCL ; on suppose une
    couverture suffisante de pixels clairs dans chaque cellule 100 m (l'écart
    de couverture S2/Landsat reste < 5 j, biais faible sur les indices).

    `native_proj` doit être la projection native 10 m de la collection S2
    AVANT compositing (passée par l'appelant) : le composite `s2.median()` ne
    porte pas de projection par défaut exploitée par reduceResolution(), d'où
    l'appel explicite à setDefaultProjection() ci-dessous."""
    ndvi = s2_median.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2_median.normalizedDifference(["B3", "B8"]).rename("NDWI")
    bands = [ndvi, ndwi]
    if with_ndbi:
        ndbi = s2_median.normalizedDifference(["B11", "B8"]).rename("NDBI")
        bands.append(ndbi)
    # reduceResolution() aggrège les pixels 10 m vers 100 m par moyenne.
    return (ee.Image(bands).setDefaultProjection(native_proj).reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=1024).toFloat())


# ---------------------------------------------------------------------------
# Exports + manifeste
# ---------------------------------------------------------------------------

def task_id(task):
    """Récupère l'ID de tâche GEE de façon résiliente (attribut .id ou .status())."""
    tid = getattr(task, "id", None)
    if not tid:
        try:
            tid = task.status().get("id")
        except Exception:
            tid = None
    return tid


def export_image(image, name, bbox, folder, scale):
    xmin, _, _, ymax = bbox
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        fileNamePrefix=name,
        crs=CRS,
        # Grille ancrée sur (xmin, ymax) : tous les exports (X et Y) d'une zone
        # partagent exactement les mêmes pixels -> alignement garanti.
        crsTransform=[scale, 0, xmin, 0, -scale, ymax],
        region=region_of(bbox),
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task_id(task)


def write_manifest(zone, bbox_snapped, args, exports):
    """Écrit manifests/manifest_<zone>_<ts>.json pour reproductibilité."""
    os.makedirs("manifests", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = os.path.join("manifests", f"manifest_{zone}_{ts}.json")
    manifest = {
        "zone": zone,
        "bbox_lambert93_snapped": list(bbox_snapped),
        "resolution_m": args.resolution,
        "exported_at_utc": ts,
        "cli_args": {
            "years": list(args.years),
            "months": list(args.months),
            "max_cloud_landsat_scene": args.max_cloud_landsat,
            "max_cloud_landsat_local": args.max_cloud_local,
            "max_cloud_s2": args.max_cloud_s2,
            "s2_window_days": args.s2_window,
            "with_ndbi": args.with_ndbi,
            "folder": args.folder,
        },
        "exports": exports,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifeste -> {path}")
    return path


def extract_zone(zone, bbox, args):
    cell_m = args.resolution
    snap_cell = PATCH_M if cell_m == 10 else cell_m
    bbox = snap_bbox(bbox, snap_cell)
    region = region_of(bbox)
    w_px = (bbox[2] - bbox[0]) // cell_m
    h_px = (bbox[3] - bbox[1]) // cell_m
    print(f"\n=== Zone '{zone}' : {bbox} ({w_px}x{h_px} px @ {cell_m} m) ===")

    lst_coll = landsat_lst_clear(
        region, args.years, args.months,
        args.max_cloud_landsat, args.max_cloud_local)

    # On récupère dates + scènes (petite collection déjà filtrée -> getInfo OK).
    timestamps = lst_coll.aggregate_array("system:time_start").getInfo() or []
    scene_ids = lst_coll.aggregate_array("system:index").getInfo() or []
    scenes_by_date = {}
    for ts_ms, sid in zip(timestamps, scene_ids):
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        scenes_by_date.setdefault(d, []).append(sid)
    dates = sorted(scenes_by_date)

    if not dates:
        print("  Aucune scène Landsat ne satisfait les filtres (été, nuages "
              f"< {args.max_cloud_local} % sur l'emprise). Élargir --years ou "
              f"--max-cloud-local.")
        return

    print(f"  {len(dates)} date(s) Landsat candidates : {', '.join(dates)}")

    resample = "bilinear"  # LST, optical : reprojeter proprement en bilinéaire
    exports = []
    for date_str in dates:
        day = ee.Date(date_str)
        s2 = s2_collection(region, day, args.s2_window, args.max_cloud_s2)
        n_s2 = s2.size().getInfo()
        if n_s2 == 0:
            print(f"  [{date_str}] ignorée : aucune Sentinel-2 exploitable à "
                  f"±{args.s2_window} j")
            continue

        print(f"  [{date_str}] {n_s2} images S2 dans la fenêtre"
              + (" — dry-run, pas d'export" if args.dry_run else ""))

        entry = {
            "date": date_str,
            "landsat_scene_ids": scenes_by_date[date_str],
            "n_s2_in_window": n_s2,
            "files": [],
        }

        y_img, coverage, mosaic_scene_ids = build_ymosaic(
            lst_coll, day, args.mosaic_window_days, region, cell_m)
        if coverage < args.min_aoi_coverage:
            print(f"  [{date_str}] ignorée : couverture emprise "
                  f"{coverage:.1%} < seuil {args.min_aoi_coverage:.0%} "
                  f"meme apres mosaique +/-{args.mosaic_window_days} j")
            continue

        entry["aoi_coverage_fraction"] = round(coverage, 4)
        entry["landsat_mosaic_scene_ids"] = mosaic_scene_ids

        if args.dry_run:
            exports.append(entry)
            continue

        if cell_m == 10:
            x_img = build_x_stack_10m(s2.median())
            tag_x = "X_s2"
            resample_x = True
        else:
            x_img = build_x_stack_100m(s2.median(), args.with_ndbi,
                                       s2.first().select("B4").projection())
            tag_x = "X_s2_100m"
            resample_x = True   # l'optique reste en bilinéaire pour la reprojection

        file_x = f"{tag_x}_{zone}_{date_str}"
        file_y = f"Y_lst_{('100m_' if cell_m == 100 else '')}{zone}_{date_str}"

        if resample_x:
            x_img = x_img.resample("bilinear")

        tid_x = export_image(x_img, file_x, bbox, args.folder, cell_m)
        tid_y = export_image(y_img, file_y, bbox, args.folder, cell_m)
        entry["files"] = [
            {"name": f"{file_x}.tif", "task_id": tid_x},
            {"name": f"{file_y}.tif", "task_id": tid_y},
        ]
        exports.append(entry)

    if not args.dry_run and exports:
        print(f"  {sum(len(e['files']) for e in exports)} export(s) lancés vers "
              f"Drive/{args.folder} (suivi : https://code.earthengine.google.com/tasks)")

    write_manifest(zone, bbox, args, exports)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", default="nantes_centre",
                    help=f"une zone parmi {list(ZONES)} ou 'all' (défaut: nantes_centre)")
    ap.add_argument("--resolution", type=int, default=100, choices=(10, 100),
                    help="résolution d'export : 10 (mode U-Net /extension) ou "
                         "100 (mode recentré, défaut)")
    ap.add_argument("--years", nargs=2, type=int, default=[2022, 2025],
                    metavar=("DEBUT", "FIN"),
                    help="années incluses (défaut: 2022 2025 — 2026 exclu pour "
                         "reproductibilité)")
    ap.add_argument("--months", nargs=2, type=int, default=[6, 8],
                    metavar=("DEBUT", "FIN"), help="mois inclus (défaut: 6 8)")
    ap.add_argument("--max-cloud-landsat", type=float, default=30.0,
                    help="pré-filtre CLOUD_COVER scène (%%) avant le filtrage fin "
                         "par emprise (défaut: 30)")
    ap.add_argument("--max-cloud-local", type=float, default=5.0,
                    help="couverture nuageuse max sur l'emprise (QA_PIXEL, %% ; "
                         "défaut: 5)")
    ap.add_argument("--max-cloud-s2", type=float, default=20.0,
                    help="CLOUDY_PIXEL_PERCENTAGE max des composites S2 (défaut: 20)")
    ap.add_argument("--s2-window", type=int, default=5,
                    help="fenêtre S2 en jours autour de la date Landsat (défaut: 5)")
    ap.add_argument("--mosaic-window-days", type=int, default=3,
                    help="fenetre +/- jours pour chercher des scenes Landsat "
                         "complementaires afin de combler la couverture de "
                         "l'emprise (mosaic) (defaut: 3)")
    ap.add_argument("--min-aoi-coverage", type=float, default=0.98,
                    help="fraction minimale de l'emprise qui doit avoir des "
                         "pixels LST valides pour garder une date (defaut: 0.98)")
    ap.add_argument("--with-ndbi", action="store_true",
                    help="(mode 100 m) ajoute le NDBI via le SWIR B11")
    ap.add_argument("--folder", default="ICU_Nantes",
                    help="dossier Google Drive de destination (défaut: ICU_Nantes)")
    ap.add_argument("--project", default=None,
                    help="projet Google Cloud pour ee.Initialize()")
    ap.add_argument("--dry-run", action="store_true",
                    help="liste les paires Landsat/S2 sans lancer d'export")
    ap.add_argument("--list-zones", action="store_true",
                    help="liste les zones et quitte")
    args = ap.parse_args()

    if args.list_zones:
        for name, bbox in ZONES.items():
            cell = PATCH_M if args.resolution == 10 else args.resolution
            print(f"{name:20s} {snap_bbox(bbox, cell)}")
        return

    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        sys.exit(f"Échec ee.Initialize ({exc}).\n"
                 "Lancer d'abord : earthengine authenticate "
                 "(et fournir --project si nécessaire).")

    if args.zone == "all":
        zones = ZONES
    elif args.zone in ZONES:
        zones = {args.zone: ZONES[args.zone]}
    else:
        sys.exit(f"Zone inconnue '{args.zone}'. Choix : {list(ZONES)} ou 'all'.")

    for name, bbox in zones.items():
        extract_zone(name, bbox, args)

    print("\nTerminé." if not args.dry_run else "\nDry-run terminé.")


if __name__ == "__main__":
    main()