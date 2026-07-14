"""Smoke test de app.py en mode démo via Streamlit AppTest (sans données réelles)."""
import os

from streamlit.testing.v1 import AppTest


def test_app_demo_mode_no_exceptions():
    HERE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(HERE))   # cwd = racine projet pour trouver data/*
    ab = AppTest.from_file("app.py").run(timeout=120)
    assert len(ab.exception) == 0, [repr(e) for e in ab.exception]
