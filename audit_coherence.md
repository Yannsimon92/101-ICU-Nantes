# Audit de cohérence — Projet ICU Nantes

*Généré le 2026-07-07. Corpus analysé : `icu_nantes_deep_learning_spec.md`,
`01_comprendre_les_donnees_ICU.ipynb`, `02_comprendre_le_modele_UNet.ipynb`,
`model.py`, `dataset.py`, `gee_extraction.py`, `prepare_patches.py`,
`train.py`, `predict.py`, `app.py`.*

---

## 1. Verdict global

Le pipeline est cohérent de bout en bout : formats, noms de classes, ordre des
canaux, tailles de patch et unités s'alignent entre la spec, les notebooks et
les scripts. L'incohérence majeure (architecture §4.2) est **corrigée**. Il
reste 3 incohérences documentaires dans la spec (non bloquantes, signalées
ci-dessous sans correction silencieuse) et 2 limites conceptuelles à connaître.

## 2. Incohérence majeure : architecture U-Net §4.2 — CORRIGÉE ✅

- **Avant** : la spec §4.2 avait 2 étages de descente et **pas de pooling avant
  le bottleneck**, mais 2 étages de remontée → `torch.cat([t1, x3])`
  concaténait un tenseur 128×128 avec un 64×128... spatialement désaligné,
  crash immédiat du forward.
- **Après** : §4.2 reprend **à l'identique** l'architecture de référence du
  notebook 02 §2.2 (vérifié couche par couche) : `inc(7→64)`,
  `down1(64→128)`, `down2(128→256)`, `down3(256→512)` (bottleneck avec
  MaxPool), puis `up1/conv_up1(512→256)`, `up2/conv_up2(256→128)`,
  `up3/conv_up3(128→64)`, `outc(64→1)`. Sortie = même (H, W) que l'entrée.
- Le fichier importable `model.py` contient la même architecture (version
  LightningModule). **Testé** : forward 256×256 et 64×64, `training_step` +
  backward OK.
- Une note dans la spec documente le bug et désigne le notebook 02 comme
  référence.

## 3. Vérification GEE → `NantesICUDataset` (mission n°2) ✅

| Attendu par `NantesICUDataset` | Produit par le pipeline | OK |
|---|---|---|
| `data_dir/X/` et `data_dir/Y/`, mêmes noms de fichiers | `prepare_patches.py` écrit `out_dir/X/` et `out_dir/Y/` avec noms identiques (`<zone>_<date>_r<row>_c<col>.tif`) | ✅ |
| X : 7 canaux, 256×256, 10 m | GEE exporte 5 bandes (B4, B3, B2, NDVI, NDWI) ; `prepare_patches.py` fusionne canopée + bâti (Open Data reprojeté à la volée) → **7 canaux**, tuilage 256×256 | ✅ |
| Ordre des canaux | `[R, G, B, NDVI, NDWI, canopée, bâti]` identique dans notebook 01 (cell 8), notebook 02 (`generate_synthetic_batch`), `X_BAND_NAMES` de `prepare_patches.py` et le stack GEE | ✅ |
| Y : 1 canal LST en °C, 100 m rééchantillonné à 10 m | GEE : `ST_B10 × 0.00341802 + 149.0 − 273.15` (facteurs officiels Collection 2), export plus-proche-voisin sur la grille 10 m | ✅ |
| Alignement pixel-perfect X/Y | Même `crsTransform` EPSG:2154 ancré sur l'emprise, snappée en multiples de 2560 m (= 256 px) → nombre entier de patchs, zéro décalage | ✅ |

⚠️ Point d'attention : le GeoTIFF X exporté par GEE **seul** n'a que 5 canaux.
Les 7 canaux n'existent qu'après `prepare_patches.py`. Sans les options
`--canopee/--bati`, les 2 canaux morphologiques sont remplis de zéros (le
script avertit) : le format reste valide mais le modèle perd l'information
« canyon urbain ».

## 4. Vérification app Streamlit (mission n°3) ✅

- `app.py` **ne charge pas le modèle** (choix d'architecture logicielle) : elle
  affiche les GeoTIFF produits par `predict.py`. C'est `predict.py` qui importe
  `GuidedSuperResUNet` depuis `model.py` — donc **la version corrigée à 3
  étages**, jamais celle du markdown bugué.
- Compatibilité des formats : `predict.py` écrit
  `data/predictions/pred_lst_<zone>_<date>.tif` (1 bande, °C, nodata=NaN) ;
  l'app scanne `pred_lst_*` et `Y_lst_*` et apparie par clé `<zone>_<date>` —
  les conventions de nommage se recollent sur toute la chaîne
  (`X_full_` → `pred_lst_` par simple remplacement de préfixe).
- **Testé en exécution réelle** (AppTest, mode démo) : 3 couches, sliders,
  métriques et histogramme fonctionnent sans exception.

## 5. Incohérences documentaires restantes (signalées, non corrigées)

Conformément à la consigne, pas de modification silencieuse de la logique
métier — à arbitrer :

1. **NDBI orphelin** : la spec §2.1 présente le NDBI (nécessite le SWIR B11)
   comme feature, et le notebook 01 le calcule pédagogiquement, mais il n'est
   **ni dans les 7 canaux** du modèle (§3.1, §4.1), ni exporté par GEE (un
   commentaire dans `build_x_stack` indique où l'ajouter). Soit le retirer de
   §2.1, soit passer à 8 canaux — décision projet.
2. **B8/NIR ambigu** : §2.1 liste B8 comme bande d'entrée, mais le NIR ne sert
   qu'au calcul des indices, il n'est pas un canal du tensor. Idem « nature des
   revêtements / imperméabilisation » (§2.1), absente des 7 canaux.
3. **SSIM Loss annoncée, pas implémentée** : §3.2 définit
   `L = α·L1 + β·SSIM + γ·consistency`, mais le code (§4.2, `model.py`,
   notebook 02) n'implémente que `L1 + 0.5·consistency`. Le notebook 02 le dit
   explicitement (« dans la version complète ») ; la spec §4.2, elle, ne le
   signale pas. Ajout possible via `torchmetrics.image.SSIM` — non fait pour ne
   pas toucher à la fonction de perte.

Différence assumée (pas une incohérence) : `GuidedSuperResUNet` est un
`nn.Module` dans le notebook 02 (pédagogique, avec `verbose`) et un
`pl.LightningModule` dans `model.py`/spec (production). Même nom, même forward.

## 6. Limites conceptuelles à connaître (aucune action code)

1. **Pixel loss L1 contre une cible en blocs 100 m** : sur données réelles, Y
   « à 10 m » n'est que le Landsat 100 m rééchantillonné — la L1 pixel tire
   donc la prédiction vers une image en blocs, ce qui bride la super-résolution
   (les notebooks utilisent une vérité 10 m *synthétique*, cas idéal). Pistes
   classiques : baisser α au profit de la consistency loss, ajouter la SSIM
   (cf. §5.3) ou un discriminateur Pix2Pix (déjà évoqué §3).
2. **Alignement des blocs 100 m vs fenêtres d'avg-pooling** : 256 n'est pas un
   multiple de 10, et surtout la grille 100 m native de Landsat (UTM) n'est pas
   alignée sur notre grille d'export Lambert-93. L'`avg_pool2d 10×10` de la
   consistency loss **approxime** donc la mesure Landsat au lieu de la
   retrouver exactement (et ignore les 6 derniers pixels de chaque patch).
   Acceptable en pratique, mais à garder en tête si la loss de cohérence stagne.

## 7. Reste à vérifier manuellement (non testable ici)

- **Authentification GEE** : `earthengine authenticate` + un projet Cloud
  (`--project`) ; vérifier que les collections `LANDSAT/LC09/C02/T1_L2` et
  `COPERNICUS/S2_SR_HARMONIZED` sont accessibles depuis le compte.
- **Emprises des zones** : les coordonnées Lambert-93 de `ZONES` dans
  `gee_extraction.py` sont approximatives — à valider sur une carte avant un
  gros export (surtout `nantes_metropole`).
- **Couches Open Data** : rasteriser canopée et bâti de Nantes Métropole en
  GeoTIFF Lambert-93 (n'importe quelle grille : reprojection à la volée), et
  vérifier visuellement l'alignement sur un patch.
- **Dépendances non installées localement** : `rasterio`,
  `pytorch_lightning`, `earthengine-api` (dans `requirements.txt` mais absents
  de l'env `lewagon`) — `gee_extraction.py`, `prepare_patches.py`, `train.py`
  et `predict.py` ne sont vérifiés qu'en compilation, pas en exécution réelle.
- **Petits jeux de données** : avec < 5 patchs, le split 80/20 de
  `ICUDataModule` peut donner un jeu de validation vide (EarlyStopping sur
  `val_loss` échouerait) — prévoir assez de patchs ou un garde-fou.
- Cosmétique : fichiers `*:Zone.Identifier` (artefacts WSL) à supprimer si le
  dossier passe sous git.
