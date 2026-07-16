# ICU Nantes — Îlots de Chaleur Urbains à Nantes Métropole

Cartographie et **analyse explicative** des îlots de chaleur urbains (ICU) à
Nantes Métropole, à la résolution native Landsat (100 m). La question du
projet : *où sont les ICU à Nantes, et qu'est-ce qui les explique ?*

On combine la température de surface Landsat 8/9 (LST, 100 m), des indices
Sentinel-2 (NDVI, NDWI, NDBI) agrégés à la même grille, et les couches
morphologiques Open Data de Nantes Métropole (canopée, bâti) pour produire :

- des **cartes d'anomalie ΔLST** par date (LST − médiane spatiale, qui élimine
  la météo du jour et isole la structure thermique urbaine),
- une **carte de fréquence ICU multi-dates** (% de dates où chaque pixel est
  en surchauffe) — le livrable robuste, qui révèle les ICU structurels plutôt
  qu'une météo ponctuelle,
- une **analyse explicative** : régression linéaire + LightGBM sur l'anomalie
  ΔLST, interprétée par SHAP — *« +10 % de canopée ≈ −X °C toutes choses
  égales par ailleurs »*.

> 📐 **Une extension par super-résolution profonde (U-Net guidé 100 m → 10 m)
> a été prototypée puis écartée du périmètre** : en l'absence de vérité terrain
> à 10 m, la validation du détail généré n'est pas défendable (voir
> [`extension/audit_scientifique.md`](extension/audit_scientifique.md)). Le
> prototype et l'analyse critique sont conservés dans
> [`extension/`](extension/) — matière d'entretien, pas livrable.

## Landsat & Sentinel-2 : deux satellites, deux rôles

Le projet combine deux missions d'observation de la Terre, chacune apportant
ce que l'autre n'a pas :

| | **Landsat 8/9** (NASA/USGS) | **Sentinel-2** (ESA/Copernicus) |
|---|---|---|
| Rôle dans le pipeline | **Cible (Y)** : température de surface | **Features (X)** : indices optiques |
| Produit GEE utilisé | `LANDSAT/LC08\|LC09/C02/T1_L2` (Collection 2, niveau 2) | `COPERNICUS/S2_SR_HARMONIZED` (réflectance de surface) |
| Capteur clé | Thermique infrarouge (bande `ST_B10`) | Optique multispectral (visible, proche infrarouge, SWIR) |
| Résolution native | 100 m (thermique) | 10 m (bandes visibles/NIR), 20 m (SWIR) |
| Revisite | ~16 j par satellite (~8 j Landsat 8+9 combinés) | ~5 j (2 satellites Sentinel-2A/B/C) |
| Pourquoi indispensable | Seul satellite du duo à mesurer une **température** — impossible de calculer un ICU sans lui | Résolution fine + bandes dédiées à la végétation/eau/bâti — Landsat seul n'a pas d'équivalent aussi précis pour ces indices |

**Landsat 8/9** fournit la bande `Y_lst_100m_*.tif` : `to_lst()` dans
`gee_extraction.py` convertit la bande thermique brute `ST_B10` en °C avec les
facteurs officiels Collection 2, masquée des nuages/ombres via le bit-mask
`QA_PIXEL`. Résolution native 100 m — c'est elle qui fixe la résolution de
tout le pipeline (inutile de sur-résoudre le reste à une échelle que la
température ne peut pas atteindre).

**Sentinel-2** fournit la bande `X_s2_100m_*.tif` : composite médian sur une
fenêtre de ±5 jours autour de chaque date Landsat (masqué nuages/cirrus via la
bande `SCL`), puis `build_x_stack_100m()` calcule NDVI (végétation), NDWI
(eau) et NDBI (bâti, optionnel) à partir des bandes brutes 10 m, avant de les
agréger à 100 m par moyenne (`reduceResolution`) pour les aligner sur la
grille Landsat.

Sans cette combinaison, on aurait soit une température sans explication
(Landsat seul), soit une cartographie fine de la végétation/du bâti sans
lien avec la chaleur réelle (Sentinel-2 seul). C'est la mise en correspondance
des deux, pixel par pixel sur la même grille 100 m, qui permet la régression
et l'analyse SHAP.

## Cadrage Machine Learning : X (features) et y (target)

Une fois les deux satellites alignés sur la même grille 100 m, `build_table.py`
transforme les rasters en une table tabulaire classique — **1 ligne = 1 pixel
× 1 date** — avec un `X` et un `y` au sens scikit-learn :

- **y (cible)** : `delta_lst_c`, l'**anomalie spatiale** ΔLST = LST − médiane
  spatiale de la scène — pas la LST brute. Ce choix élimine l'effet de la
  météo du jour (une canicule qui touche toute la ville ne doit pas être
  confondue avec un ICU) et isole la structure thermique propre à chaque
  pixel, comparée au reste de la métropole ce jour-là.
- **X (features)**, définies dans `model_evaluation.py`
  (`FEATURES = ["ndvi", "ndwi", "ndbi", "canopee", "bati"]`) :
  - `ndvi`, `ndwi`, `ndbi` — indices Sentinel-2 (végétation, eau, bâti),
    recalculés à chaque date.
  - `canopee`, `bati` — couches morphologiques statiques Open Data Nantes
    Métropole (fraction de canopée / bâti par cellule 100 m), identiques à
    toutes les dates : elles capturent la structure urbaine plutôt que la
    météo du jour.

Deux modèles sont entraînés sur ce couple `X → y` (voir `model_evaluation.py`) :

1. **Régression linéaire** — baseline interprétable : chaque coefficient se
   lit directement en « °C par unité de feature » (ex. NDVI +0.1 ⇒ ≈ −X °C),
   toutes choses égales par ailleurs.
2. **LightGBM** — gradient boosting, capture les non-linéarités et
   interactions entre features qu'une régression linéaire ne peut pas voir.

Le split train/val/test est **spatial** (`GroupShuffleSplit` sur `bloc_id`,
blocs de ~2×2 km) plutôt qu'aléatoire pixel par pixel, pour éviter toute fuite
d'information entre pixels voisins quasi-identiques. Les valeurs **SHAP** du
modèle LightGBM quantifient ensuite la contribution de chaque feature `X` à
l'anomalie `y` prédite — c'est le livrable explicatif du projet.

## Pipeline

```
gee_extraction.py ──> Google Drive ──> data/raw/                  (LST 100 m + NDVI/NDWI/NDBI 100 m)
                                          │
                                          ├──> compute_icu.py ──> data/icu/                    (ΔLST, masques ICU, fréquence)
                                          │                            │
                                          └──> build_table.py ──> data/table.parquet          (pixel × date)
                                                                       │
                                                                       v
                                       app.py  <── data/eval/  <── model_evaluation.py         (linreg + LightGBM + SHAP)
```

1. **Extraction** (`gee_extraction.py --resolution 100`) — pour chaque scène
   Landsat estivale claire (2022–2025, juin-août, filtre nuages **par emprise**
   via `QA_PIXEL`), exporte deux GeoTIFF alignés 100 m Lambert-93 : LST en °C
   et un stack d'indices S2 agrégés (NDVI, NDWI, + NDBI avec `--with-ndbi`).
   Manifeste JSON par export pour reproductibilité.

   ```bash
   pip install -r requirements.txt
   earthengine authenticate
   python gee_extraction.py --zone nantes_metropole --resolution 100 --with-ndbi --project mon-projet
   ```

2. **Cartes ICU** (`compute_icu.py`) — anomalies ΔLST, masques binaires ICU
   (ΔLST > +2 °C par défaut, eau NDWI>0.3 exclue), et **carte de fréquence
   multi-dates** (la plus-value vs une simple collection Landsat brute) :

   ```bash
   python compute_icu.py --raw-dir data/raw --out-dir data/icu --zone nantes_metropole
   ```

3. **Table** (`build_table.py`) — empile les rasters 100 m → Parquet
   (1 ligne = 1 pixel × 1 date), filtre l'eau, classifie le tissu
   (dense / pavillonnaire / industriel / végétal) :

   ```bash
   python build_table.py --raw-dir data/raw --out data/table.parquet \
       --canopee data/morpho/canopee.tif --bati data/morpho/bati.tif
   ```

4. **Analyse explicative** (`model_evaluation.py`) — split **spatial par blocs**
   (`GroupShuffleSplit` sur `bloc_id`, ~2×2 km ; zéro fuite train/test — le test
   qui valide le protocole est dans `tests/test_model_evaluation.py`), puis
   régression linéaire + LightGBM, métriques RMSE/MAE/R² globales **et
   stratifiées par tissu**, et SHAP (summary + dependence plots) :

   ```bash
   python model_evaluation.py --table data/table.parquet --out-dir data/eval
   # optionnel : validation terrain points de fraîcheur Open Data
   python model_evaluation.py --table data/table.parquet --out-dir data/eval \
       --points data/open_data/points_fraicheur.geojson
   ```

5. **Visualisation** (`app.py`) — carte folium (fond CartoDB Positron) avec
   couches LST / ΔLST / fréquence ICU, points de fraîcheur superposés, panneau
   SHAP, et **encadré de mise en garde** (température de surface ≠ air,
   SUHI diurne ≠ UHI nocturne, résolution 100 m = quartier pas rue).
   Fonctionne aussi sans données (mode démo) :

   ```bash
   streamlit run app.py
   ```

## Tests

```bash
pytest tests/ -v
```

12 tests : split sans fuite (zéro bloc partagé), calcul d'anomalie,
classification de tissu, table end-to-end, run complet
`compute_icu → build_table → model_evaluation`, smoke test de l'app Streamlit.

## Environnement Python

- Python 3.14 (test OK) — voir `requirements.lock` pour le gel complet
- Venv projet recommandé :

  ```bash
  python -m venv .venv/icu
  # Windows (Git Bash) :  .venv/icu/Scripts/python.exe -m pip install -r requirements.txt
  # Linux / macOS :       source .venv/icu/bin/activate && pip install -r requirements.txt
  ```

## Stack technique

- **GEE** (Landsat C2 L2, Sentinel-2 SR harmonisé) — extraction serveur
- **rasterio** — I/O GeoTIFF + reprojection à la volée (`WarpedVRT`)
- **scikit-learn** — régression linéaire + split spatial `GroupShuffleSplit`
- **LightGBM** — gradient boosting
- **SHAP** — interprétation (TreeExplainer)
- **streamlit + folium** — visualisation interactive

## Limites assumées (argument d'entretien, pas faiblesse)

- **Température de surface** (LST Landsat), pas de l'air.
- Acquisitions **matinales d'été par ciel clair** (~10 h 50 UTC) : SUHI
  diurne, **différent de l'ICU nocturne** visé par les politiques de fraîcheur.
- Résolution 100 m = échelle du **quartier**, pas de la rue.
- **Couverture Landsat inégale selon les dates** (tuilage WRS-2 : la bbox
  métropole chevauche deux traces adjacentes) : sur `nantes_metropole`, ~11
  des 20 dates couvrent l'emprise entière, les ~9 autres n'en couvrent qu'une
  tranche (une seule des deux scènes du jour a passé le filtre nuages). La
  fréquence ICU est donc plus robuste au centre-ouest qu'en périphérie
  (est, vers Carquefou/Orvault), où moins de dates contribuent par pixel.
- Validation **in situ** non réalisée (cf. audit scientifique) ; le projet est
  exploratoire et pédagogique, à valider terrain avant communication opérationnelle.

## Périmètre & feuilles de route

- Roadmap active (v2 recentrée) :
  [`icu_roadmap_3jours_projet_recentre.md`](icu_roadmap_3jours_projet_recentre.md)
- Roadmap alternative (super-résolution U-Net, écartée) :
  [`extension/icu_roadmap_v2.md`](extension/icu_roadmap_v2.md)
- Audits scientifiques (verrous critiques) :
  [`extension/audit_scientifique.md`](extension/audit_scientifique.md) +
  [`extension/audit_scientifique_resume.md`](extension/audit_scientifique_resume.md)
- Audit de cohérence du prototype U-Net :
  [`extension/audit_coherence.md`](extension/audit_coherence.md)
