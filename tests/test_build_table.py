"""Tests build_table.py — construction du DataFrame pixel×date."""
import os
import subprocess
import sys

import pandas as pd
import pytest

from build_table import find_pairs, classify_tissu, scene_to_records


def test_find_pairs(synthetic_zone):
    pairs = find_pairs(synthetic_zone["raw"])
    assert len(pairs) == 2
    for zone, date, yp, xp in pairs:
        assert zone == synthetic_zone["zone"]
        assert date.startswith("2024-")
        assert os.path.exists(yp) and os.path.exists(xp)


def test_classify_tissu_thresholds():
    import numpy as np
    H = W = 6
    can = np.full((H, W), 0.4)       # par défaut végétal
    bat = np.full((H, W), 0.05)      # bas -> végétal (canopy >= 0.3, bat < 0.10)
    ndwi = np.zeros((H, W))
    # Cluster dense (bâti élevé, canopée 0)
    can[0, 0] = 0.0; bat[0, 0] = 0.7
    # Cluster industriel (bâti élevé mais > canopy 0.2)
    can[1, 1] = 0.1; bat[1, 1] = 0.5
    # Eau
    ndwi[2, 2] = 0.5
    # Pavillonnaire (bâti moyen, canopée moyenne)
    can[3, 3] = 0.2; bat[3, 3] = 0.2
    t = classify_tissu(can, bat, ndwi)
    assert t[0, 0] == "dense"
    assert t[1, 1] == "industriel"
    assert t[2, 2] == "eau"
    assert t[3, 3] == "pavillonnaire"
    assert t[5, 5] == "végétal"


def test_scene_to_records_filters_water_and_classifies(synthetic_zone):
    zone = synthetic_zone["zone"]
    pairs = find_pairs(synthetic_zone["raw"])
    zone_pairs = [(d, yp, xp) for (z, d, yp, xp) in pairs if z == zone]
    date, y_path, x_path = zone_pairs[0]
    can_path = f"{synthetic_zone['morpho']}/canopee_{zone}.tif"
    bat_path = f"{synthetic_zone['morpho']}/bati_{zone}.tif"
    df = scene_to_records(zone, date, y_path, x_path, can_path, bat_path,
                          bloc_px=5, ndwi_water=0.30)
    assert isinstance(df, pd.DataFrame) and not df.empty
    # Pas d'eau dans la table (filtre NDWI > 0.3)
    assert (df["tissu"] == "eau").sum() == 0
    # Tissus attendus présents (au moins végétal + un chaud)
    tissus = set(df["tissu"].unique())
    assert "végétal" in tissus
    assert any(t in tissus for t in ("dense", "industriel"))
    # Toutes colonnes attendues présentes
    for col in ("zone", "date", "bloc_id", "lst_c", "delta_lst_c",
                "ndvi", "ndwi", "canopee", "bati", "tissu"):
        assert col in df.columns, f"missing {col}"


def test_build_table_main_writes_parquet(synthetic_zone, tmp_path):
    out = tmp_path / "table.parquet"
    cmd = [
        sys.executable, "build_table.py",
        "--raw-dir", synthetic_zone["raw"],
        "--out", str(out),
        "--canopee", f"{synthetic_zone['morpho']}/canopee_{synthetic_zone['zone']}.tif",
        "--bati", f"{synthetic_zone['morpho']}/bati_{synthetic_zone['zone']}.tif",
        "--bloc-px", "5",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    df = pd.read_parquet(out)
    assert len(df) > 0
    assert df["zone"].nunique() == 1
    assert df["date"].nunique() == 2
    assert "delta_lst_c" in df.columns
