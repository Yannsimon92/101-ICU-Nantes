# ICU Nantes — Super-résolution des Îlots de Chaleur Urbains

Détection et cartographie à 10 m des îlots de chaleur urbains (ICU) de Nantes
Métropole par super-résolution guidée : la température de surface Landsat 8/9
(100 m) est contrainte par Sentinel-2 et la morphologie urbaine (10 m).

Spécification complète : [icu_nantes_deep_learning_spec.md](icu_nantes_deep_learning_spec.md).
Notebooks pédagogiques : `01_comprendre_les_donnees_ICU.ipynb` (données),
`02_comprendre_le_modele_UNet.ipynb` (architecture — version de référence du modèle).

## Pipeline

```
gee_extraction.py ──> Google Drive ──> data/raw/ ──> prepare_patches.py ──> data/patches/
                                                            │                    │
                                                            │ (full/)            │ (X/, Y/)
                                                            v                    v
   app.py <── data/predictions/ <── predict.py <── checkpoint <────────────── train.py
```

1. **Extraction** (`gee_extraction.py`) — exporte, pour chaque scène Landsat
   estivale sans nuages (2022-2026, juin-août, < 5 %), un couple de GeoTIFF
   alignés pixel-perfect en Lambert-93 à 10 m : stack Sentinel-2
   (RGB + NDVI + NDWI) et LST Landsat. `--zone all` lance le batch sur toutes
   les zones de Nantes Métropole définies dans `ZONES`.

   ```bash
   pip install earthengine-api && earthengine authenticate
   python gee_extraction.py --zone nantes_centre --dry-run   # lister les paires
   python gee_extraction.py --zone all --project mon-projet  # batch métropole
   ```

2. **Préparation** (`prepare_patches.py`) — télécharger les exports Drive dans
   `data/raw/`, puis tuiler en patchs 256×256 et fusionner les couches Open
   Data (canopée, bâti) pour obtenir les 7 canaux du modèle :

   ```bash
   python prepare_patches.py --raw-dir data/raw --out-dir data/patches \
       --canopee data/morpho/canopee_10m.tif --bati data/morpho/bati_10m.tif
   ```

3. **Entraînement** (`train.py`) — U-Net de super-résolution guidée
   (`model.py`, 3 étages symétriques), loss L1 + cohérence 100 m, mixed precision :

   ```bash
   python train.py --data-dir data/patches --epochs 50
   ```

4. **Inférence** (`predict.py`) — fenêtre glissante avec fusion de Hann sur la
   zone complète, sortie GeoTIFF LST 10 m :

   ```bash
   python predict.py "data/patches/full/X_full_*.tif" --checkpoint <best.ckpt>
   ```

5. **Visualisation** (`app.py`) — carte folium (fond CartoDB Positron) avec
   couches Prédiction 10 m / Landsat 100 m / Anomalie ICU, seuil ICU réglable,
   statistiques et distribution. Fonctionne aussi sans données (mode démo) :

   ```bash
   streamlit run app.py
   ```

## Note sur l'architecture du modèle

La version de référence du U-Net est celle du **notebook 02** et de
`model.py` : 3 étages de descente/remontée symétriques. Une version antérieure
de la spec (§4.2) ne comportait que 2 descentes sans pooling avant le
bottleneck, ce qui désalignait les skip-connections (`torch.cat` plantait) —
corrigée depuis, les trois sources sont cohérentes.

## Environnement Python

- Python 3.14 (test OK) — voir `requirements.lock` pour le gel complet des versions
- Venv projet : voir `icu_roadmap_3jours_projet_recentre.md` (roadmap active) et `icu_roadmap_v2.md` (extension U-Net)

```bash
python -m venv .venv/icu
# Windows (Git Bash) :  .venv/icu/Scripts/python.exe -m pip install -r requirements.txt
# Linux / macOS :       source .venv/icu/bin/activate && pip install -r requirements.txt
```
