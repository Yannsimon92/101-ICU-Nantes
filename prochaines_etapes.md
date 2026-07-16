# Prochaines étapes — ICU Nantes v2 recentrée

*État au 2026-07-14. Code complet et testé (12 pytest verts), projet poussé sur
https://github.com/Yannsimon92/101-ICU-Nantes. Reste à exécuter sur données
réelles puis à valider.*

## 1. Données — blocages utilisateur

### 1.1 Authentification Google Earth Engine ✅ fait (2026-07-16)
```bash
# Une fois le venv active (le paquet earthengine-api installe un script CLI
# "earthengine", pas un module -m earthengine) :
earthengine authenticate --auth_mode=notebook
# Le mode par defaut (gcloud) a ete bloque par Google ("Cette application est
# bloquee") sur le compte perso -> --auth_mode=notebook route via le client
# OAuth dedie Earth Engine et fonctionne (code d'autorisation colle a la main).
```
Reste a verifier/obtenir : un projet Google Cloud avec l'API Earth Engine
activee (le `--project <id>` a passer aux commandes ci-dessous).

### 1.2 Valider les emprises Lambert-93 🟠
Les coordonnées des `ZONES` dans `gee_extraction.py` sont **approximatives**
(signalé par `audit_coherence.md` §7). Avant un gros export métropole, vérifier
sur une carte (par ex. ouvrir un `geojson` des emprises dans QGIS / geojson.io)
que `nantes_metropole` couvre bien la métropole et que `reze_sud_loire` /
`saint_herblain` sont bien placées.

### 1.3 Couches Open Data Nantes Métropole 🟠 (nécessaire pour SHAP)
Rasteriser en Lambert-93 (n'importe quelle grille — reprojection à la volée via
`WarpedVRT`) :
- canopée (fraction 0..1 par cellule) → `data/morpho/canopee.tif`
- bâti (fraction 0..1) → `data/morpho/bati.tif`

Sources Open Data Nantes Métropole :BDTopo/OCS-GE ou la couche "ilots de
fraîcheur" / "végétation" selon ce qui est disponible. Sans ces rasters, les
colonnes `canopee`/`bati` de la table seront à 0 et **la classification de
tissu + l'analyse SHAP perdent leur cœur métier** (l'effet canopée/bâti).

### 1.4 Points de fraîcheur (J2.6) 🟡 optionnel
Télécharger le jeu Open Data Nantes Métropole des points de fraîcheur (parcs,
fontaines, brumisateurs…) → `data/open_data/points_fraicheur.geojson`
(features avec `geometry` Point en EPSG:4326). Le script `model_evaluation.py
--points data/open_data/points_fraicheur.geojson` s'en occupe ensuite.

---

## 2. Lancer le pipeline sur données réelles

### 2.1 Dry-run (combien de dates exploitables ?)
```bash
.venv/icu/Scripts/python.exe gee_extraction.py \
    --zone nantes_metropole --resolution 100 --with-ndbi \
    --project <ton-project> --dry-run
```
Vérifier le nombre de dates candidates (été 2022–2025, nuages < 5 % sur
l'emprise). Si < 5 dates, desserrer `--max-cloud-local 10` ou `--months 5 9`.

### 2.2 Export réel GEE → Google Drive
```bash
.venv/icu/Scripts/python.exe gee_extraction.py \
    --zone nantes_metropole --resolution 100 --with-ndbi \
    --project <ton-project>
```
Suivi : https://code.earthengine.google.com/tasks — télécharger les `.tif` depuis
Drive/ICU_Nantes vers `data/raw/`. Le manifeste est écrit dans
`manifests/manifest_nantes_metropole_<ts>.json`.

### 2.3 Cartes ICU
```bash
.venv/icu/Scripts/python.exe compute_icu.py \
    --raw-dir data/raw --out-dir data/icu --zone nantes_metropole
```
**Sanity check J1.4** (le script l'affiche dans stdout) : la Loire et l'Erdre
doivent ressortir froides (ΔLST < 0), les zones industrielles / grands parkings
chaudes (ΔLST > +2 °C). Si l'inverse -> bug d'alignement, à régler avant J2.

### 2.4 Table + modèles + SHAP
```bash
.venv/icu/Scripts/python.exe build_table.py \
    --raw-dir data/raw --out data/table.parquet \
    --canopee data/morpho/canopee.tif --bati data/morpho/bati.tif

.venv/icu/Scripts/python.exe model_evaluation.py \
    --table data/table.parquet --out-dir data/eval \
    --points data/open_data/points_fraicheur.geojson
```
Livrables : `data/eval/metrics.json` + `metrics_rmse_stratif.csv` +
`shap_summary.png` + `shap_dependence_*.png` + `lightgbm.pkl`.

### 2.5 Visualisation
```bash
.venv/icu/Scripts/python.exe -m streamlit run app.py
```
Vérifier visuellement : couches LST / ΔLST / fréquence ICU, panneau SHAP,
encadré de mise en garde. Ajuster `--threshold` (seuil ICU) si la carte de
fréquence paraît trop/largement peuplée.

---

## 3. Améliorations code (post-données réelles)

### 3.1 `validate_points_fraicheur` est une approximation 🟡
Actuellement la fonction compare le ΔLST moyen tissu-végétal vs tissu-chaud
(tabulaire). Quand les vraies coordonnées X/Y des points de fraîcheur seront
disponibles, intersecter proprement chaque point avec la LST raster et le
tissu environnant (~200 m = 2 px à 100 m). Fonction à reprendre dans
`model_evaluation.py`.

### 3.2 Stratification temporelle du split 🟡
Le split actuel (`GroupShuffleSplit` sur `bloc_id`) est purement spatial. Pour
durcir le protocole (audit Sci §1) : combiner avec un hold-out temporel (ex.
été 2025 en test, 2022–2024 en train). ~15 lignes à ajouter dans
`split_train_val_test` (grouper par `bloc_id` ET date).

### 3.3 Notebook `03_evaluation_et_baselines.ipynb` 🟢 bonus portfolio
Mêmes style pédagogique que `01_comprendre_les_donnees_ICU.ipynb` : pourquoi les
baselines, lecture du tableau comparatif, visualisations avant/après. Pas
critique, mais bon signal portfolio.

---

## 4. Décisions projet en suspens

Reprises de `extension/icu_roadmap_v2.md` §« Décisions à prendre » :

1. **NDBI** : garder `--with-ndbi` activé par défaut sur les exports réels ?
   Décision à prendre après avoir vu l'importance SHAP du NDBI sur le premier
   run.
2. **Fenêtre temporelle publiée** : figer 2022–2025 (reproductible,
   recommandé). Garder 2026 pour une éventuelle démo live une fois l'été
   terminé.
3. **`nantes_metropole`** : zone d'export + de prédiction (recommandé pour la
   v2 recentrée). Pas de split temporel/zone en test dans l'immédiat — la
   séparation par blocs spatiaux suffit pour la table.

---

## 5. Récap commandes (raccourci)

```bash
# une fois pour toutes (venv active)
earthengine authenticate

# J1 — données + cartes
.venv/icu/Scripts/python.exe gee_extraction.py --zone nantes_metropole --resolution 100 --with-ndbi --project <projet>
.venv/icu/Scripts/python.exe compute_icu.py --raw-dir data/raw --out-dir data/icu

# J2 — table + modèles
.venv/icu/Scripts/python.exe build_table.py --raw-dir data/raw --out data/table.parquet \
    --canopee data/morpho/canopee.tif --bati data/morpho/bati.tif
.venv/icu/Scripts/python.exe model_evaluation.py --table data/table.parquet --out-dir data/eval \
    --points data/open_data/points_fraicheur.geojson

# J3 — visualisation
.venv/icu/Scripts/python.exe -m streamlit run app.py

# tests
.venv/icu/Scripts/python.exe -m pytest tests/ -v
```

## 6. État courant (snapshot)

- 5 commits sur `main`, poussés sur https://github.com/Yannsimon92/101-ICU-Nantes
- 12 tests pytest verts (dont split sans fuite, compute_anomaly, end-to-end)
- Code du pipeline complet : `gee_extraction.py` → `compute_icu.py` →
  `build_table.py` → `model_evaluation.py` → `app.py`
- Extension U-Net rangée dans `extension/` (non perdue, matière d'entretien)
- **Aucune donnée réelle encore extraite** — en attente de l'auth GEE de
  l'utilisateur
