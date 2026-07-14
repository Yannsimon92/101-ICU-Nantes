"""Tests model_evaluation.py — split spatial sans fuite, fit + métriques."""
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import pytest

from model_evaluation import split_train_val_test, stratified_metrics


def test_split_no_bloc_overlap():
    """Le test clé (audit Sci §1) : zéro bloc_id partagé entre train/val/test."""
    blocs = [f"z0-{r}-{c}" for r in range(8) for c in range(8) for _ in range(20)]
    df = pd.DataFrame({
        "bloc_id": blocs,
        "delta_lst_c": np.random.default_rng(0).normal(0, 1, len(blocs)),
        "tissu": "pavillonnaire",
        "ndvi": 0.3, "ndwi": -0.1, "canopee": 0.2, "bati": 0.2,
    })
    train, val, test = split_train_val_test(df)
    s_tr = set(train["bloc_id"]); s_va = set(val["bloc_id"]); s_te = set(test["bloc_id"])
    assert len(s_tr & s_va) == 0
    assert len(s_tr & s_te) == 0
    assert len(s_va & s_te) == 0
    # Le test set contient une fraction réaliste
    assert 0.10 < len(test) / len(df) < 0.30


def test_split_determinism_same_seed():
    blocs = [f"b{i}" for i in range(20) for _ in range(10)]
    df = pd.DataFrame({"bloc_id": blocs,
                       "delta_lst_c": np.zeros(len(blocs)),
                       "tissu": "x", "ndvi": 0.0, "ndwi": 0.0,
                       "canopee": 0.0, "bati": 0.0})
    a1 = split_train_val_test(df); a2 = split_train_val_test(df)
    assert list(a1[0].index) == list(a2[0].index)
    assert list(a1[2].index) == list(a2[2].index)


def test_stratified_metrics_per_tissu():
    df = pd.DataFrame({"tissu": ["dense"] * 5 + ["végétal"] * 5})
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 0.6, 0.7, 0.8, 0.9])
    y_pred = y_true.copy()
    m = stratified_metrics(df, y_true, y_pred)
    assert m.loc["dense", "rmse"] == 0.0
    assert m.loc["végétal", "rmse"] == 0.0
    assert m.loc["global", "n"] == 10


def test_model_evaluation_main_end_to_end(synthetic_zone, tmp_path):
    """Run end-to-end: build_table -> parquet -> model_evaluation -> metrics.json."""
    table = tmp_path / "table.parquet"
    eval_dir = tmp_path / "eval"
    # 1. build_table
    r1 = subprocess.run([
        sys.executable, "build_table.py",
        "--raw-dir", synthetic_zone["raw"], "--out", str(table),
        "--canopee", f"{synthetic_zone['morpho']}/canopee_{synthetic_zone['zone']}.tif",
        "--bati", f"{synthetic_zone['morpho']}/bati_{synthetic_zone['zone']}.tif",
        "--bloc-px", "5",
    ], capture_output=True, text=True, timeout=120)
    assert r1.returncode == 0, r1.stderr
    # 2. model_evaluation
    r2 = subprocess.run([
        sys.executable, "model_evaluation.py",
        "--table", str(table), "--out-dir", str(eval_dir),
    ], capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr
    # 3. Vérifie les livrables
    metrics_path = eval_dir / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, encoding="utf-8") as f:
        m = __import__("json").load(f)
    assert "linear" in m and "lightgbm" in m and "shap" in m
    assert "metrics_rmse_stratif" in m
    # Le split gelé a été écrit
    assert os.path.exists("data/split/blocs_split.json")
