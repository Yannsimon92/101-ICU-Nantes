# Roadmap 3 jours — ICU Nantes v2 « recentré »

*Périmètre : cartographie et analyse explicative des îlots de chaleur urbains
à Nantes Métropole, à la résolution native Landsat (100 m). La super-résolution
deep learning est écartée du périmètre (voir encadré final) — décision
documentée, pas un abandon.*

**Question du projet** : *où sont les îlots de chaleur à Nantes, et qu'est-ce
qui les explique ?*

**Réutilisé de la v1 (~70 % du code)** : `gee_extraction.py` (extraction
Landsat/Sentinel-2, alignement des grilles, manifestes), `prepare_patches.py`
(fusion canopée/bâti — adapté), `app.py` (visualisation — adaptée), notebook
pédagogique 01 (indices), audits (annexe méthodo).

---

## Jour 1 — Données et cartes ICU

### Matin : extraction propre

- [ ] **1.1** Reprendre `gee_extraction.py` avec les correctifs d'hygiène de
      l'audit :
      - fenêtre figée `--years 2022 2025` (2026 exclu : année incomplète,
        reproductibilité) ;
      - manifeste JSON par export (IDs de scènes, bbox, arguments) ;
      - filtre nuages **sur l'emprise** (`QA_PIXEL` moyenné via
        `reduceRegion`) et non sur la propriété scène → plus de dates
        utilisables.
- [ ] **1.2** Exporter pour Nantes Métropole entière :
      - LST **à 100 m natif** (plus besoin du rééchantillonnage 10 m !) ;
      - features agrégées à la même grille 100 m : NDVI, NDWI moyens
        (Sentinel-2 10 m → moyenne par cellule 100 m), % canopée, % bâti,
        (optionnel si temps : NDBI avec le SWIR B11 — il redevient trivial
        à intégrer sans le verrou des 7 canaux du U-Net).
- [ ] **1.3** `git init` + seeds + `requirements.lock` (30 min, une fois pour
      toutes).

### Après-midi : cartes d'anomalies

- [ ] **1.4** Script `compute_icu.py` : pour chaque date, anomalie
      `ΔLST = LST − médiane spatiale de la scène` ; carte ICU binaire
      (ΔLST > +2 °C) ; **carte de synthèse multi-dates** : fréquence
      d'appartenance à un ICU (ex. « ce pixel est en surchauffe dans 8 scènes
      sur 10 ») — beaucoup plus robuste qu'une date unique, et c'est une
      vraie plus-value par rapport à une carte Landsat brute.
- [ ] **1.5** Vérification visuelle rapide : la Loire, l'Erdre et les grands
      parcs (Procé, Grand Blottereau) doivent ressortir froids ; les zones
      industrielles et grands parkings commerciaux chauds. Si non → bug
      d'alignement, à régler avant d'aller plus loin.

**Livrable J1** : GeoTIFFs d'anomalies par date + carte de fréquence ICU +
manifeste reproductible.

---

## Jour 2 — Analyse explicative (le cœur data science)

### Matin : dataset tabulaire et modèles

- [ ] **2.1** `build_table.py` : empiler les rasters 100 m → DataFrame
      (1 ligne = 1 pixel × 1 date) : `ΔLST ~ NDVI + NDWI + canopée + bâti
      (+ NDBI) + date`. Filtrer les pixels d'eau (NDWI élevé) pour ne pas
      polluer la régression.
- [ ] **2.2** **Split spatial par blocs** (le point 🔴 n°1 de l'audit reste
      pertinent, mais devient simple ici) : blocs de ~2×2 km assignés
      entièrement à train/val/test, seedé. En tabulaire c'est ~15 lignes
      avec `GroupKFold` de scikit-learn (groupe = ID de bloc).
- [ ] **2.3** Modèles, du plus simple au plus riche :
      1. régression linéaire (baseline interprétable — les coefficients
         se lisent en « °C par point de NDVI ») ;
      2. gradient boosting (LightGBM/XGBoost) avec le même split.
      Métriques : RMSE/MAE globales **et stratifiées par classe de tissu**
      (dense / pavillonnaire / végétal — seuils sur canopée/bâti, recyclé de
      l'audit §6).

### Après-midi : interprétation

- [ ] **2.4** Importance des variables + **SHAP values** sur le gradient
      boosting : quantifier l'effet de chaque facteur (ex. « +10 % de canopée
      ≈ −X °C toutes choses égales par ailleurs »). C'est LE résultat
      actionnable du projet, celui qu'un aménageur ou un recruteur retient.
- [ ] **2.5** Dependence plots des 2-3 variables dominantes (non-linéarités :
      l'effet canopée sature-t-il ? le bâti a-t-il un seuil ?).
- [ ] **2.6** **Validation terrain points de fraîcheur** (Open Data Nantes,
      promise dans la spec v1, enfin implémentée) : delta thermique moyen des
      points de fraîcheur vs tissu environnant à ~200 m. Un seul chiffre avec
      intervalle (bootstrap), mais un chiffre *terrain*.

**Livrable J2** : notebook `analyse_explicative.ipynb` avec tableau de
métriques (linéaire vs GBM, global + stratifié), SHAP, validation points de
fraîcheur.

---

## Jour 3 — Restitution

### Matin : application

- [ ] **3.1** Adapter `app.py` (l'essentiel existe) :
      - couches : LST, anomalie ΔLST, fréquence ICU multi-dates, points de
        fraîcheur superposés ;
      - panneau « facteurs explicatifs » : les résultats SHAP en 3-4 phrases
        et un graphique d'importance ;
      - **l'encadré de mise en garde de l'audit §7** (adapté) : température
        *de surface* et non de l'air ; matinées d'été par ciel clair
        (SUHI diurne ≠ UHI nocturne) ; résolution 100 m = échelle du
        quartier, pas de la rue.
- [ ] **3.2** Test de bout en bout : extraction → ICU → analyse → app dans un
      env vierge, en suivant uniquement le README.

### Après-midi : portfolio

- [ ] **3.3** README vitrine : problème → données → méthode → **carte + tableau
      + graphique SHAP** (les 3 visuels qui portent le projet) → limites
      assumées → stack (GEE, rasterio, scikit-learn/LightGBM, SHAP,
      Streamlit).
- [ ] **3.4** Encadré « travaux d'extension étudiés » : 3-4 phrases sur la
      super-résolution U-Net écartée, avec lien vers les audits en annexe du
      repo. Formulation suggérée : *« Une extension par super-résolution
      profonde (100 m → 10 m, U-Net guidé) a été prototypée puis écartée du
      périmètre : en l'absence de vérité terrain à 10 m, la validation du
      détail généré n'est pas défendable (voir audit_scientifique.md). Le
      prototype et l'analyse critique sont conservés dans /extension. »*
- [ ] **3.5** Déplacer le code U-Net v1 + notebooks 01-02 + audits dans
      `/extension` (rien n'est jeté : c'est de la matière d'entretien).
- [ ] **3.6** Tests pytest minimaux : split sans fuite (zéro bloc partagé),
      calcul d'anomalie, chargement des rasters. 1-2 h, gros signal
      « hygiène d'ingénieur ».

**Livrable J3** : repo git propre, app démo, README avec résultats.

---

## Ce qui rend ce projet défendable (vs la v1)

| Problème v1 (audits) | Statut dans le projet recentré |
|---|---|
| 🔴 Fuite spatiale train/val | Réglé simplement : GroupKFold par blocs sur tabulaire |
| 🔴 Loss incapable d'apprendre le détail 10 m | N'existe plus : on ne génère pas de détail, on explique le 100 m |
| 🔴 Baseline imbattable | N'existe plus : la régression linéaire EST la baseline, comparée au GBM |
| 🟠 Pas de métriques stratifiées | Intégré J2 (par classe de tissu) |
| 🟠 SUHI/UHI, domaine de validité | Encadré app + README (J3) |
| 🟠 Validation terrain jamais codée | Intégrée J2 (points de fraîcheur) |
| 🟠 Reproductibilité | Réglé J1 (git, seeds, lock, manifeste, années figées) |

## Arguments d'entretien que ce découpage te donne

1. Un résultat **actionnable et chiffré** (effet de la canopée sur la
   température de surface à Nantes) plutôt qu'une loss abstraite.
2. Une **décision de périmètre argumentée** (super-résolution écartée pour
   raisons méthodologiques, audits à l'appui) — signal fort de maturité.
3. De la **rigueur géospatiale** (split spatial, validation terrain,
   stratification) — différenciant vs les projets Kaggle classiques.
4. Un pipeline **reproductible de bout en bout** (manifestes, seeds, tests).

## Répartition Claude Code suggérée

- **Fable (high)** : J1 matin (refonte extraction + manifestes) et J2
  (build_table + split spatial + modèles + SHAP) — les parties où la
  cohérence multi-fichiers et le protocole comptent.
- **Sonnet (medium/high)** : J1 après-midi (compute_icu, assez mécanique),
  J3 entier (adaptation app, README, tests) — pas besoin de Fable ni de
  ses usage credits pour ça.
