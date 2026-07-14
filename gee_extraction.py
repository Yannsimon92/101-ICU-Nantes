"""Extraction Google Earth Engine pour le projet ICU Nantes.

Pour chaque scène Landsat estivale quasi sans nuages trouvée sur la zone,
le script exporte vers Google Drive deux GeoTIFF **parfaitement alignés**
(même CRS EPSG:2154, même grille 10 m via crsTransform, même emprise) :

  - X_s2_<zone>_<date>.tif : stack Sentinel-2 à 10 m, 5 bandes
        [B4 (Rouge), B3 (Vert), B2 (Bleu), NDVI, NDWI]
    Les 2 canaux morphologiques (canopée, bâti) viennent de l'Open Data
    Nantes Métropole et sont fusionnés localement par prepare_patches.py
    pour obtenir le tensor X à 7 canaux attendu par le modèle.
  - Y_lst_<zone>_<date>.tif : LST Landsat 8/9 (°C) à 100 m natif,
    rééchantillonnée sur la grille 10 m en plus-proche-voisin (chaque pixel
    10 m porte la valeur de son parent 100 m ; l'average-pooling 10x10 de la
    consistency loss retrouve donc exactement la mesure Landsat).

Appariement temporel : pour chaque date Landsat, un composite médian
Sentinel-2 est construit sur une fenêtre de ±N jours (défaut 5), après
masquage des nuages/ombres via la bande SCL.

Prérequis :
    pip install earthengine-api
    earthengine authenticate

Exemples :
    python gee_extraction.py --list-zones
    python gee_extraction.py --zone nantes_centre --dry-run
    python gee_extraction.py --zone nantes_centre --project mon-projet-gee
    python gee_extraction.py --zone all            # batch Nantes Métropole
"""

import argparse
import sys

import ee

CRS = "EPSG:2154"          # Lambert-93, comme l'Open Data Nantes Métropole
SCALE = 10                 # résolution cible en mètres
PATCH_M = 256 * SCALE      # 2560 m : les emprises sont des multiples de patchs 256 px

# Emprises en Lambert-93 (xmin, ymin, xmax, ymax), approximatives — à ajuster
# librement, elles sont re-snappées sur la grille des patchs au chargement.
ZONES = {
    "nantes_centre":    (353000, 6687500, 358000, 6692500),
    "ile_de_nantes":    (353500, 6684500, 358500, 6688000),
    "nantes_nord":      (352000, 6692000, 358000, 6697000),
    "saint_herblain":   (346500, 6687000, 352500, 6692000),
    "reze_sud_loire":   (353000, 6680000, 358500, 6685000),
    # L'ensemble de la métropole : gros export (~4000 x 3400 px par bande)
    "nantes_metropole": (335000, 6671000, 375000, 6705000),
}


def snap_bbox(bbox, cell=PATCH_M):
    """Étend l'emprise aux multiples de `cell` pour que les patchs 256x256 tuilent exactement."""
    xmin, ymin, xmax, ymax = bbox
    xmin = (xmin // cell) * cell
    ymin = (ymin // cell) * cell
    xmax = ((xmax + cell - 1) // cell) * cell
    ymax = ((ymax + cell - 1) // cell) * cell
    return int(xmin), int(ymin), int(xmax), int(ymax)


def region_of(bbox):
    return ee.Geometry.Rectangle(list(bbox), proj=CRS, geodesic=False)


# ---------------------------------------------------------------------------
# Landsat 8/9 Collection 2 Level 2 -> LST (°C) masquée des nuages
# ---------------------------------------------------------------------------

def landsat_lst_collection(region, year_range, month_range, max_scene_cloud):
    def to_lst(img):
        qa = img.select("QA_PIXEL")
        # Bits QA_PIXEL : 1 = nuage dilaté, 2 = cirrus, 3 = nuage, 4 = ombre de nuage
        clear = qa.bitwiseAnd(0b11110).eq(0)
        # Facteurs officiels Collection 2 : ST_B10 * 0.00341802 + 149.0 (Kelvin)
        lst = (img.select("ST_B10")
               .multiply(0.00341802).add(149.0)
               .subtract(273.15)
               .rename("LST"))
        return ee.Image(lst.updateMask(clear)
                        .copyProperties(img, ["system:time_start", "CLOUD_COVER"]))

    merged = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")))
    return (merged
            .filterBounds(region)
            .filter(ee.Filter.calendarRange(year_range[0], year_range[1], "year"))
            .filter(ee.Filter.calendarRange(month_range[0], month_range[1], "month"))
            .filter(ee.Filter.lt("CLOUD_COVER", max_scene_cloud))
            .map(to_lst))


# ---------------------------------------------------------------------------
# Sentinel-2 SR harmonisé -> composite médian masqué des nuages (bande SCL)
# ---------------------------------------------------------------------------

def s2_collection(region, center_date, window_days, max_scene_cloud):
    def mask_and_scale(img):
        scl = img.select("SCL")
        # SCL : 3 = ombre de nuage, 8/9 = nuages, 10 = cirrus
        clear = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
        return img.updateMask(clear).divide(10000)

    start = center_date.advance(-window_days, "day")
    end = center_date.advance(window_days + 1, "day")
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud))
            .map(mask_and_scale))


def build_x_stack(s2_median):
    """Stack 5 bandes : B4, B3, B2, NDVI, NDWI (ajouter B11/NDBI ici si besoin)."""
    ndvi = s2_median.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2_median.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return (s2_median.select(["B4", "B3", "B2"])
            .addBands([ndvi, ndwi])
            .toFloat())


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def export_image(image, name, bbox, folder, resample_bilinear):
    xmin, _, _, ymax = bbox
    if resample_bilinear:
        image = image.resample("bilinear")
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=folder,
        fileNamePrefix=name,
        crs=CRS,
        # Grille ancrée sur (xmin, ymax) : tous les exports d'une zone partagent
        # exactement les mêmes pixels -> alignement X/Y garanti.
        crsTransform=[SCALE, 0, xmin, 0, -SCALE, ymax],
        region=region_of(bbox),
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def extract_zone(zone, bbox, args):
    bbox = snap_bbox(bbox)
    region = region_of(bbox)
    w_px = (bbox[2] - bbox[0]) // SCALE
    h_px = (bbox[3] - bbox[1]) // SCALE
    print(f"\n=== Zone '{zone}' : {bbox} ({w_px}x{h_px} px @ {SCALE} m) ===")

    lst_coll = landsat_lst_collection(region, args.years, args.months,
                                      args.max_cloud_landsat)
    dates = (lst_coll.aggregate_array("system:time_start")
             .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
             .distinct().sort().getInfo())

    if not dates:
        print("  Aucune scène Landsat ne satisfait les filtres (été, nuages "
              f"< {args.max_cloud_landsat} %). Élargir --years ou --max-cloud-landsat.")
        return []

    print(f"  {len(dates)} date(s) Landsat candidates : {', '.join(dates)}")
    tasks = []

    for date_str in dates:
        day = ee.Date(date_str)
        s2 = s2_collection(region, day, args.s2_window, args.max_cloud_s2)
        n_s2 = s2.size().getInfo()
        if n_s2 == 0:
            print(f"  [{date_str}] ignorée : aucune image Sentinel-2 exploitable "
                  f"à ±{args.s2_window} j")
            continue

        print(f"  [{date_str}] appariement OK ({n_s2} images S2 dans la fenêtre)"
              + (" — dry-run, pas d'export" if args.dry_run else ""))
        if args.dry_run:
            continue

        # Moyenne des scènes Landsat du jour (gère le recouvrement de traces)
        y_img = lst_coll.filterDate(day, day.advance(1, "day")).mean().toFloat()
        x_img = build_x_stack(s2.median())

        # X : bilinéaire (reprojection UTM -> L93 plus propre pour l'optique).
        # Y : plus-proche-voisin (défaut GEE) pour préserver les blocs 100 m.
        tasks.append(export_image(x_img, f"X_s2_{zone}_{date_str}", bbox,
                                  args.folder, resample_bilinear=True))
        tasks.append(export_image(y_img, f"Y_lst_{zone}_{date_str}", bbox,
                                  args.folder, resample_bilinear=False))

    if tasks:
        print(f"  {len(tasks)} export(s) lancés vers Drive/{args.folder} "
              "(suivi : https://code.earthengine.google.com/tasks)")
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", default="nantes_centre",
                    help=f"une zone parmi {list(ZONES)} ou 'all' (défaut: nantes_centre)")
    ap.add_argument("--years", nargs=2, type=int, default=[2022, 2026],
                    metavar=("DEBUT", "FIN"), help="années incluses (défaut: 2022 2026)")
    ap.add_argument("--months", nargs=2, type=int, default=[6, 8],
                    metavar=("DEBUT", "FIN"), help="mois inclus (défaut: 6 8 = juin-août)")
    ap.add_argument("--max-cloud-landsat", type=float, default=5.0,
                    help="couverture nuageuse max de la scène Landsat en %% (défaut: 5)")
    ap.add_argument("--max-cloud-s2", type=float, default=20.0,
                    help="couverture nuageuse max des scènes S2 du composite (défaut: 20)")
    ap.add_argument("--s2-window", type=int, default=5,
                    help="fenêtre d'appariement S2 en jours autour de la date Landsat (défaut: 5)")
    ap.add_argument("--folder", default="ICU_Nantes",
                    help="dossier Google Drive de destination (défaut: ICU_Nantes)")
    ap.add_argument("--project", default=None,
                    help="projet Google Cloud pour ee.Initialize()")
    ap.add_argument("--dry-run", action="store_true",
                    help="liste les paires Landsat/S2 sans lancer d'export")
    ap.add_argument("--list-zones", action="store_true", help="liste les zones et quitte")
    args = ap.parse_args()

    if args.list_zones:
        for name, bbox in ZONES.items():
            print(f"{name:20s} {snap_bbox(bbox)}")
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

    all_tasks = []
    for name, bbox in zones.items():
        all_tasks += extract_zone(name, bbox, args)

    print(f"\nTotal : {len(all_tasks)} export(s) lancés."
          if not args.dry_run else "\nDry-run terminé.")


if __name__ == "__main__":
    main()
