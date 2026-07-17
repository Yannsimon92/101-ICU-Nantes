"""Smoke test de app.py (générateur de carte kepler.gl) en mode démo,
sans données réelles."""
import os

from app import run


def test_app_demo_mode_generates_html(tmp_path):
    HERE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(HERE))   # cwd = racine projet

    # Dossiers/fichiers volontairement absents pour forcer le mode démo,
    # indépendamment des données réelles éventuellement présentes en local.
    empty = tmp_path / "no_data"
    map_path, index_path = run(
        raw_dir=str(empty / "raw"),
        icu_dir=str(empty / "icu"),
        table=str(empty / "table.parquet"),
        eval_dir=str(empty / "eval"),
        out_dir=str(tmp_path / "web"),
        points_path=str(empty / "points_fraicheur.geojson"),
    )

    assert os.path.exists(map_path) and os.path.getsize(map_path) > 0
    assert os.path.exists(index_path) and os.path.getsize(index_path) > 0
