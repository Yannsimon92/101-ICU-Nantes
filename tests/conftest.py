"""Fixtures synthétiques pour les tests du pipeline ICU Nantes.

Génère des exports GEE fictifs (Y_lst_100m, X_s2_100m) + canopée/bâti
rasterisés, en Lambert-93 — assez petits pour des tests rapides.
"""
import numpy as np
import rasterio
import pytest
from rasterio.transform import from_origin

H = W = 60
SCALE = 100
XMIN, YMAX = 353000, 6692500


def _profile(count=1, dtype="float32"):
    return dict(
        crs="EPSG:2154",
        transform=from_origin(XMIN, YMAX, SCALE, SCALE),
        width=W, height=H,
        count=count, dtype=dtype, nodata=float("nan"),
    )


def _write_band(path, arr, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(profile["dtype"]), 1)


def _make_scene(zone, date, hot, cold, ref_c, seed):
    rng = np.random.default_rng(seed)
    base = float(ref_c) + rng.normal(0, 0.4, size=(H, W)).astype(np.float32)
    hr, hc = hot
    base[hr:hr + 8, hc:hc + 8] += 4.0       # zone industrielle / parking
    cr, cc = cold
    base[cr:cr + 8, cc:cc + 8] -= 5.0       # eau / parc
    ndvi = rng.uniform(0.0, 0.6, (H, W)).astype(np.float32)
    ndwi = rng.uniform(-0.4, 0.1, (H, W)).astype(np.float32)
    ndwi[cr:cr + 8, cc:cc + 8] = 0.5       # eau taguée NDWI
    return base, ndvi, ndwi


def _make_morpho(hot, cold, seed):
    rng = np.random.default_rng(seed)
    canopy = rng.uniform(0.05, 0.35, (H, W)).astype(np.float32)
    bati = rng.uniform(0.05, 0.30, (H, W)).astype(np.float32)
    hr, hc = hot
    bati[hr:hr + 8, hc:hc + 8] = 0.7
    canopy[hr:hr + 8, hc:hc + 8] = 0.0      # industriel = dense/bâti
    cr, cc = cold
    canopy[cr:cr + 8, cc:cc + 8] = 0.5
    bati[cr:cr + 8, cc:cc + 8] = 0.0        # végétal
    return canopy, bati


@pytest.fixture
def synthetic_zone(tmp_path):
    """Crée raw/ + morpho/ pour 1 zone (nantes_centre), 2 dates."""
    raw = tmp_path / "raw"; raw.mkdir()
    morpho = tmp_path / "morpho"; morpho.mkdir()
    zone = "nantes_centre"
    hot, cold = (15, 15), (40, 40)
    can, bat = _make_morpho(hot, cold, seed=11)
    _write_band(str(morpho / f"canopee_{zone}.tif"), can, _profile())
    _write_band(str(morpho / f"bati_{zone}.tif"), bat, _profile())
    for di, ds in enumerate(["2024-07-15", "2024-08-12"]):
        b, ndvi, ndwi = _make_scene(zone, ds, hot, cold, ref_c=30.0, seed=21 + di)
        _write_band(str(raw / f"Y_lst_100m_{zone}_{ds}.tif"), b, _profile())
        prof_x = _profile(count=2)
        with rasterio.open(str(raw / f"X_s2_100m_{zone}_{ds}.tif"), "w", **prof_x) as dst:
            dst.write(ndvi.astype("float32"), 1); dst.set_band_description(1, "NDVI")
            dst.write(ndwi.astype("float32"), 2); dst.set_band_description(2, "NDWI")
    return {"raw": str(raw), "morpho": str(morpho), "zone": zone,
            "hot_xy": hot, "cold_xy": cold, "H": H, "W": W}
