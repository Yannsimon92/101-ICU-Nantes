"""Tests compute_icu.py — anomalies, masques binaire ICU, carte de fréquence."""
import numpy as np
import rasterio

from compute_icu import find_pairs, compute_anomaly


def test_find_pairs_detects_100m(synthetic_zone):
    pairs = find_pairs(synthetic_zone["raw"])
    assert len(pairs) == 2
    zones = {p[0] for p in pairs}
    assert zones == {synthetic_zone["zone"]}
    for _, _, yp, xp in pairs:
        assert yp is not None
        assert xp is not None


def test_compute_anomaly_excludes_water_and_is_zero_centered(synthetic_zone):
    pairs = find_pairs(synthetic_zone["raw"])
    zone, date, y_path, x_path = pairs[0]
    with rasterio.open(y_path) as src:
        lst = src.read(1, masked=True).filled(np.nan).astype(np.float32)
    from compute_icu import read_band_aligned
    with rasterio.open(y_path) as src:
        ndwi = read_band_aligned(x_path, "NDWI", src.profile) if x_path else None
    valid_land = np.isfinite(lst) & (ndwi <= 0.30)
    delta, ref = compute_anomaly(lst, valid_land)
    assert np.isfinite(ref)
    # Médiane terrestre ~ zéro (par construction de l'anomalie)
    assert abs(np.nanmedian(delta[valid_land])) < 1e-6
    # L'eau n'a pas d'anomalie NaN (mais on l'exclut ensuite du masque ICU)
    hot = synthetic_zone["hot_xy"]; hr, hc = hot
    assert np.nanmean(delta[hr:hr + 8, hc:hc + 8]) > 3.0    # hot spot > +3 °C


def test_pipeline_run_produces_frequency(synthetic_zone, tmp_path):
    """Run compute_icu.py end-to-end sur la fixture."""
    import compute_icu as ci
    out_dir = str(tmp_path / "icu")
    pairs = ci.find_pairs(synthetic_zone["raw"])
    n = ci.process_zone(synthetic_zone["zone"], pairs, out_dir, threshold=2.0)
    assert n == 2
    freq_path = f"{out_dir}/icu_frequency_{synthetic_zone['zone']}.tif"
    with rasterio.open(freq_path) as s:
        freq = s.read(1, masked=True).filled(np.nan)
    # Le hot spot doit être ICU sur toutes les dates (fréquence = 100 %)
    hr, hc = synthetic_zone["hot_xy"]
    assert np.nanmean(freq[hr:hr + 8, hc:hc + 8]) >= 95.0
    # L'eau doit être NaN (exclue à chaque date -> accum_valid = 0 -> masquée)
    cr, cc = synthetic_zone["cold_xy"]
    assert np.all(np.isnan(freq[cr:cr + 8, cc:cc + 8]))
