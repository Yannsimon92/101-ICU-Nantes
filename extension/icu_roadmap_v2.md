# Feuille de route v2 — Projet ICU Nantes

*Synthèse des audits du 2026-07-07 (`audit_coherence.md`, `audit_scientifique.md`,
`audit_scientifique_resume.md`). Objectif : passer d'un pipeline « qui tourne »
à un projet dont les résultats sont scientifiquement défendables et
présentables en portfolio ML Engineer.*

---

## Vue d'ensemble

**État v1** : pipeline techniquement cohérent de bout en bout (GEE → patchs →
U-Net → prédiction → app Streamlit), architecture corrigée, alignement
géométrique soigné, notebooks pédagogiques solides.

**Verdict des audits** : en l'état, *aucun chiffre produit n'est défendable*.
Trois verrous critiques 🔴 forment un tout indissociable :

1. La **validation est contaminée** (fuite spatiale + patchs dupliqués + pas de
   test set) → tout chiffre est optimiste par construction.
2. La **loss n'apprend pas la super-résolution** sur données réelles (Y n'est
   qu'un Landsat 100 m rééchantillonné ; la L1 pixel récompense la
   reproduction des blocs, pas le détail 10 m).
3. **Aucune baseline** → la valeur ajoutée du deep learning est inaffirmable
   (et la bicubique gagnerait probablement avec la métrique actuelle).

La feuille de route suit le chemin critique identifié par l'audit : on ne
communique aucun résultat avant la fin de la Phase 2.

---

## Phase 0 — Verrous d'hygiène (½ journée) 🔧

*Rapides, indépendants, à faire d'abord car tout le reste en dépend pour être
traçable.*

| # | Tâche | Source | Effort |
|---|-------|--------|:-:|
| 0.1 | `git init` + premier commit (code + manifestes, pas les rasters) + `.gitignore` (données, checkpoints, `*:Zone.Identifier`) | Sci §8 | XS |
| 0.2 | Seeds partout : `pl.seed_everything(seed, workers=True)` dans `train.py` (`--seed 42` par défaut), `deterministic=True` dans le Trainer, `random_split` seedé dans `dataset.py` (provisoire, remplacé en 1.2) | Sci §8 | XS |
| 0.3 | `pip freeze > requirements.lock` + version Python dans le README | Sci §8 | XS |
| 0.4 | Manifeste JSON par export GEE : IDs de scènes (`system:index`), date d'export, bbox snappée, arguments CLI. Figer `--years 2022 2025` pour les expériences (2026 incomplet) | Sci §8, §2 | S |
| 0.5 | Nettoyage doc : corriger la spec §2.2 (« forte chaleur » → seul le filtre nuages existe), corriger le docstring mensonger de `gee_extraction.py` (« retrouve exactement » → approximation) | Sci §2, §4a | XS |

**Critère de sortie** : deux runs identiques donnent le même split et le même
checkpoint ; le dataset est traçable scène par scène.

---

## Phase 1 — Assainir la validation (1–2 jours) 🔴 BLOQUANT

*Sans cette phase, tout chiffre produit ensuite est contaminé. Rien ne se
mesure avant qu'elle soit finie.*

| # | Tâche | Détail | Effort |
|---|-------|--------|:-:|
| 1.1 | **Dédupliquer les patchs** | Retirer `nantes_metropole` de l'entraînement (la réserver comme zone de prédiction/démo), OU dédupliquer dans `prepare_patches.py` par clé `(bounds, date)` lue dans le GeoTIFF. Le chevauchement des zones + snapping crée des patchs pixel-identiques sous des noms différents | S |
| 1.2 | **Split par blocs géographiques disjoints** | Remplacer `random_split` : clé de groupe = position du patch quantifiée en blocs de 2×2 ou 3×3 patchs, toutes les dates d'un bloc dans le même fold, tri déterministe + shuffle seedé. `NantesICUDataset` accepte une liste de fichiers (~3 lignes). L'esquisse de code est dans `audit_scientifique.md` §1 | M |
| 1.3 | **Jeu de test gelé espace × temps** | Hold-out combiné : une zone entière (ex. `reze_sud_loire`) ET un été (ex. 2025), jamais vus par l'EarlyStopping. Marge tampon d'un patch entre blocs train et test | S |
| 1.4 | Garde-fou petits datasets | Vérifier val non vide avant EarlyStopping (bug potentiel noté dans l'audit de cohérence §7) | XS |

**Critère de sortie** : zéro patch partagé (même approximativement) entre
train, val et test ; le test set n'influence ni l'arrêt ni la sélection de
checkpoint.

---

## Phase 2 — Baselines et métriques honnêtes (2–3 jours) 🔴 BLOQUANT

*C'est cette phase qui dit si le deep learning apporte quelque chose — et si
le protocole d'évaluation lui-même est valide.*

| # | Tâche | Détail | Effort |
|---|-------|--------|:-:|
| 2.1 | **`baselines.py`** | (a) Bicubique : `zoom(y_100m, 10, order=3)` ; (b) TsHARP/DisTrad : régression LST~NDVI à 100 m appliquée à 10 m + correction résiduelle (~40 lignes NumPy, LA référence du domaine) ; (c) optionnel : krigeage `pykrige`. Évaluées sur exactement les mêmes patchs de test, mêmes métriques, même masque | M |
| 2.2 | **Export Y à 100 m natif** | Deuxième export GEE (1 ligne) pour que la consistency loss et la métrique `rmse_100m` se comparent au produit natif, pas au rééchantillonné NN | S |
| 2.3 | **`evaluate.py` + métriques stratifiées** | RMSE/MAE par classe de tissu ({dense, pavillonnaire, industriel, végétal, eau} dérivées des canaux canopée/bâti par seuils) et par zone. La moyenne globale cache l'erreur sur les cœurs d'îlots — la population cible | M |
| 2.4 | **Métriques orientées décision** | Binariser l'anomalie (seuil ICU déjà dans `app.py`) → précision/rappel/IoU des zones ICU détectées vs Landsat agrégé | S |
| 2.5 | **Validation terrain points de fraîcheur** | Promise dans la spec §5 depuis le début, jamais codée : delta thermique sur les points de fraîcheur Open Data vs tissu environnant. Seule évaluation qui porte réellement sur le détail infra-100 m | M |
| 2.6 | **Tableau comparatif dans le README** | U-Net vs bicubique vs TsHARP, toutes métriques. Règle de l'audit : « toute communication de résultat sans ce tableau est non étayée » | XS |

**Critère de sortie** : un tableau chiffré défendable. Si la bicubique gagne
sur `rmse_10m` mais perd sur les métriques stratifiées/détection ICU, c'est
un *résultat*, pas un échec — ça démontre que la métrique naïve ne mesure pas
la valeur ajoutée.

---

## Phase 3 — Reformuler l'apprentissage (2–4 jours) 🔴

*Dépend des Phases 1–2 : on ne change pas la loss avant d'avoir un protocole
de mesure fiable pour comparer avant/après.*

| # | Tâche | Détail | Effort |
|---|-------|--------|:-:|
| 3.1 | **Restructurer la loss pour le cas réel** | α_L1→0 (ou suppression) contre le Y rééchantillonné ; consistency loss comme terme principal, calculée contre le Y 100 m natif (2.2) — élimine l'aller-retour NN | M |
| 3.2 | **Prior structurel haute résolution** | SSIM loss (annoncée spec §3.2, jamais implémentée — `torchmetrics.image.SSIM`) et/ou gradient loss guidée par les canaux X. C'est ce qui donne au réseau une raison d'apprendre du détail 10 m juste | M |
| 3.3 | **Stratégie deux temps** (alternative ou complément) | Pré-entraînement supervisé sur paires synthétiques (schéma des notebooks), puis fine-tuning consistency-only sur données réelles — schéma classique du downscaling thermique | L |
| 3.4 | Correctif effet de bord | Padding réflexif à 260 px avant pooling ou kernel adapté (les 6 derniers px de chaque patch sont ignorés) | XS |
| 3.5 | **Équilibrage du dataset** | `WeightedRandomSampler` ou sous-échantillonnage des patchs ruraux, sur la base de la composition par classe de tissu (2.3). Publier la composition dans le README | S |
| 3.6 | (Si 3.1–3.2 insuffisants) Discriminateur Pix2Pix | Évoqué dès la spec §1.1 — à ne tenter qu'après avoir mesuré les limites de l'approche loss-only | L |

**Critère de sortie** : le U-Net bat TsHARP sur au moins une famille de
métriques pertinente (stratifiées ou détection ICU) sur le test gelé — sinon,
conclusion honnête documentée.

---

## Phase 4 — Incertitude et communication responsable (1–2 jours) 🟠

| # | Tâche | Détail | Effort |
|---|-------|--------|:-:|
| 4.1 | **Deep ensemble K=5** | 5 entraînements seeds différentes (modèle petit, coût faible), `predict.py --checkpoints ...` → GeoTIFF 2 bandes (`pred_mean`, `pred_std`). Réutilise `sliding_window_predict` tel quel | M |
| 4.2 | **Couche σ dans l'app** | Écart-type d'ensemble dans le sélecteur de couches, masquage (gris/hachures) des pixels où σ dépasse un seuil | S |
| 4.3 | **Encadré de mise en garde** (app + README) | Quatre faits : température *de surface* (pas de l'air) ; matinée d'été par ciel clair ; détail infra-100 m *modélisé, non mesuré* ; non validé in situ | XS |
| 4.4 | **Domaine de validité écrit** | README + spec §1 : SUHI diurne (~10 h 50) ≠ UHI nocturne visé par les politiques de fraîcheur ; modèle Nantes-only (entrées non normalisées + BatchNorm → non transférable tel quel) ; conditions anticycloniques | XS |
| 4.5 | Filtre nuages local | `QA_PIXEL` moyenné sur l'emprise plutôt que la propriété scène entière → plus de dates utilisables, moins de biais | S |

---

## Phase 5 — Finition portfolio (1 jour) 🟡

| # | Tâche | Détail | Effort |
|---|-------|--------|:-:|
| 5.1 | Tests unitaires pytest | Dataset (shapes, NaN), split (zéro chevauchement train/test — le test qui prouve la Phase 1), loss, forward U-Net | M |
| 5.2 | Nettoyage doc résiduel | NDBI orphelin (retirer de la spec §2.1 ou passer à 8 canaux — décision à prendre), B8/NIR ambigu, « nature des revêtements » absente des canaux | XS |
| 5.3 | Notebook `03_evaluation_et_baselines.ipynb` | Même style pédagogique que 01/02 : pourquoi les baselines, lecture du tableau comparatif, visualisations avant/après | M |
| 5.4 | README vitrine | Problème → approche → schéma → tableau de résultats → limites assumées → stack. Les limites *documentées* sont un argument d'entretien, pas une faiblesse | S |

---

## Ordre d'exécution et dépendances

```
Phase 0 (hygiène)
   │
Phase 1 (split propre)  ──── BLOQUANT : rien ne se mesure avant
   │
Phase 2 (baselines + métriques) ──── BLOQUANT : rien ne se compare avant
   │
Phase 3 (loss v2)  ← itératif, mesuré contre Phase 2
   │
Phase 4 (incertitude + garde-fous)
   │
Phase 5 (portfolio)
```

Estimation totale : **7 à 12 jours** de travail effectif. Phases 0, 4 et 5
sont parallélisables partiellement avec le reste ; 1 → 2 → 3 est strictement
séquentiel.

## Décisions à prendre (arbitrages projet, pas techniques)

1. **NDBI** : retirer de la spec ou passer le modèle à 8 canaux (nécessite le
   SWIR B11 dans l'export GEE) ?
2. **`nantes_metropole`** : zone de prédiction/démo uniquement, ou
   déduplication fine ?
3. **Fenêtre temporelle publiée** : figer 2022–2025 (reproductible) et garder
   2026 pour la démo live ?
4. **Pix2Pix** (3.6) : seulement si les résultats de 3.1–3.2 plafonnent —
   à ne pas décider maintenant.

## Ce qui ne change pas (points forts v1 à préserver)

Alignement géométrique X/Y (grille snappée), conventions de nommage
bout-en-bout, inférence fenêtre glissante + fusion de Hann, notebooks
pédagogiques, honnêteté de la doc sur le périmètre nantais.
