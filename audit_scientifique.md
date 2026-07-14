# Audit de rigueur scientifique — Projet ICU Nantes

*Généré le 2026-07-07. Indépendant de l'audit de cohérence technique
(`audit_coherence.md`) : ici on juge les choix méthodologiques de fond, pas la
plomberie. Corpus : `dataset.py`, `model.py`, `train.py`, `predict.py`,
`gee_extraction.py`, `prepare_patches.py`, `icu_nantes_deep_learning_spec.md`,
`README.md`, notebooks 01 et 02.*

**Échelle de gravité** : 🔴 critique (invalide les conclusions) ·
🟠 sérieux (biaise les résultats, correction nécessaire avant publication) ·
🟡 modéré (à documenter/corriger, ne remet pas tout en cause).

---

## 1. Biais de split train/val/test — 🔴 CRITIQUE, aggravé par deux facteurs propres à ce projet

### Verdict : problème réel, et pire que la seule autocorrélation spatiale.

Le split est fait dans `dataset.py:65-69` :

```python
dataset = NantesICUDataset(self.data_dir)
train_size = int(0.8 * len(dataset))
self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
```

Trois défauts distincts, tous avérés dans **ce** code :

**a) Fuite par autocorrélation spatiale (le risque anticipé — confirmé).**
Les patchs sont des tuiles 256×256 jointives découpées dans les mêmes scènes
(`prepare_patches.py:115-128`). Deux tuiles adjacentes partagent le même tissu
urbain, la même météo, la même scène Landsat. Le `random_split` en met une en
train et l'autre en val : la `val_loss` mesure alors surtout la capacité
d'interpolation locale, pas la généralisation. Avec l'EarlyStopping et le
ModelCheckpoint pilotés par cette `val_loss` (`train.py:33-35`), le modèle
sélectionné est celui qui exploite le mieux la fuite.

**b) Fuite par duplication pure et simple des patchs — spécifique à ce projet,
plus grave que l'autocorrélation.** Les emprises de `ZONES`
(`gee_extraction.py:43-51`) **se chevauchent** : `nantes_centre` /
`ile_de_nantes` (bande 6 687 500–6 688 000), `nantes_centre` / `nantes_nord`
(bande 6 692 000–6 692 500), `ile_de_nantes` / `reze_sud_loire`, et le snapping
aux multiples de 2 560 m (`snap_bbox`) **élargit** encore ces recouvrements
(ex. `saint_herblain` xmax 352 500 → snappé à 353 280, alors que
`nantes_centre` xmin 353 000 → snappé à 350 720 : ~2,5 km de recouvrement créé
par le snapping). Comme toutes les zones sont snappées sur la **même** grille
globale de 2 560 m et exportées avec le même `crsTransform`, les patchs des
zones qui se recouvrent sont **pixel-identiques** (mêmes valeurs, noms de
fichiers différents). Et `nantes_metropole` englobe intégralement les 5 autres
zones : un export `--zone all` (proposé par le README) suivi de
`prepare_patches.py` produit chaque patch urbain **en double**. Le
`random_split` place alors des copies exactes d'un même patch en train et en
val — ce n'est plus une performance gonflée, c'est une validation sur le jeu
d'entraînement.

**c) Pas de jeu de test du tout.** `ICUDataModule` n'a que train/val ; la val
sert à la fois à l'arrêt précoce, à la sélection du checkpoint et
(implicitement) au chiffre final. Toute métrique rapportée est donc
optimiste par construction, indépendamment des fuites a) et b).

À cela s'ajoute la **fuite temporelle** : la même tuile géographique apparaît à
plusieurs dates (une paire de patchs par scène Landsat, `<zone>_<date>_r_c`).
La morphologie urbaine étant quasi constante entre deux étés, deux dates d'un
même lieu sont des quasi-doublons de plus.

### Correction concrète

1. **Dédupliquer à la source** : soit retirer `nantes_metropole` de `ZONES`
   (ou en faire une zone exclusive de prédiction, jamais d'entraînement), soit
   dédupliquer dans `prepare_patches.py` par clé `(bounds_du_patch, date)` —
   les bounds sont lisibles dans le GeoTIFF, deux patchs de même emprise/date
   sont des doublons à ne garder qu'une fois.
2. **Split par blocs géographiques disjoints, avec les dates groupées.**
   Remplacer le `random_split` par un split **par groupe spatial** : la clé de
   groupe est la position géographique du patch (pas la zone du nom de
   fichier, qui se chevauche), par exemple le coin du patch quantifié en blocs
   de 5×5 patchs (12,8 km... non : 2 560 m × 5 = 12,8 km, prendre 2×2 ou 3×3
   patchs selon la taille du jeu). Toutes les dates d'un même bloc vont dans
   le même fold. Esquisse :

   ```python
   import rasterio, collections

   def spatial_group(path, block_m=7680):          # blocs de 3x3 patchs
       with rasterio.open(path) as src:
           x0, y0 = src.transform.c, src.transform.f
       return (int(x0 // block_m), int(y0 // block_m))

   groups = collections.defaultdict(list)
   for f in file_names:
       groups[spatial_group(os.path.join(x_dir, f))].append(f)
   blocks = sorted(groups)                          # tri déterministe
   rng = random.Random(42); rng.shuffle(blocks)
   n = len(blocks)
   train_f = [f for b in blocks[:int(.7*n)] for f in groups[b]]
   val_f   = [f for b in blocks[int(.7*n):int(.85*n)] for f in groups[b]]
   test_f  = [f for b in blocks[int(.85*n):] for f in groups[b]]
   ```

   `NantesICUDataset` doit accepter une liste de fichiers plutôt que de lister
   le dossier — modification de 3 lignes.
3. **Ajouter un vrai jeu de test gelé**, jamais vu par l'EarlyStopping :
   idéalement un **hold-out combiné espace × temps** (ex. la zone
   `reze_sud_loire` entière **et** l'été 2025) pour mesurer séparément la
   généralisation spatiale et temporelle. Une marge tampon d'un patch entre
   blocs train et test élimine le résidu d'autocorrélation aux frontières.
4. Seed du split (voir §8) pour que le partage soit identique d'un run à
   l'autre.

---

## 2. Biais temporel du filtre "juin–août, nébulosité < 5 %" — 🟠 sérieux, dont une incohérence spec/code

### Verdict : problème réel, en partie assumable si documenté, avec un vrai point d'honnêteté à corriger dans la spec.

Ce que le code fait réellement (`gee_extraction.py:85-92, 206-213`) : scènes
Landsat 8/9, juin–août 2022–2026, `CLOUD_COVER < 5 %`. Conséquences :

- **Le filtre sélectionne des journées anticycloniques claires, pas des
  journées représentatives de l'été.** À Nantes (climat océanique), les
  journées < 5 % de nuages en été sont majoritairement des situations de
  dorsale/canicule. Le jeu de données sur-représente donc les conditions
  chaudes et sèches. Ce n'est pas rédhibitoire — c'est précisément le régime
  où l'ICU intéresse l'aménageur — mais **le domaine de validité doit être
  écrit noir sur blanc** : « LST par ciel clair, matinée d'été (passage
  Landsat ~10 h 50 locale), conditions anticycloniques ». Le modèle n'a
  aucune raison d'être valide pour un après-midi voilé ou une nuit.
- **Point plus fondamental que la nébulosité : l'heure de passage.** Landsat
  passe en fin de matinée. Or l'îlot de chaleur urbain au sens
  climatologique (excès de température **de l'air**) est maximal **la nuit**.
  Le projet cartographie le SUHI (surface, diurne), pas l'UHI nocturne qui
  motive les politiques « zones de fraîcheur ». Aucun document du projet ne
  fait cette distinction — à ajouter absolument (README + spec §1), sinon les
  cartes seront sur-interprétées.
- **Incohérence spec/code à corriger** : la spec §2.2 affirme « ciblant
  uniquement les journées de forte chaleur sans couverture nuageuse ». C'est
  faux : le code ne filtre **que** les nuages, jamais la température. Une
  journée claire et fraîche de juin passe le filtre. Soit corriger la phrase
  de la spec, soit implémenter le filtre annoncé (croiser avec la température
  max journalière d'une station Météo-France/ERA5, ex. garder T_max ≥ 30 °C).
- **Biais secondaire** : `CLOUD_COVER` est une propriété de la **scène
  entière** (185 km de large). On rejette des scènes parfaitement claires sur
  Nantes mais nuageuses ailleurs → moins de dates, et biais vers les grands
  systèmes clairs. Correction concrète : calculer la fraction nuageuse **sur
  l'emprise** à partir du masque `QA_PIXEL` déjà construit
  (`clear.reduceRegion(ee.Reducer.mean(), region, 100)`) et filtrer dessus
  (seuil 5 % local), plutôt que sur la propriété scène.
- Détail : la borne 2026 inclut l'année en cours (incomplète au 2026-07-07) —
  le jeu de données change à chaque relance (voir §8, reproductibilité).

### Généralisabilité

Les résultats ne seront **pas** généralisables aux journées « chaleur
normale », aux ciels voilés, ni à la nuit — et il faut le dire, pas le
découvrir en production. Si l'usage visé reste « cartographier les points
chauds par épisode chaud », le biais est aligné avec l'usage : c'est
acceptable **à condition** que la doc et l'app Streamlit affichent le domaine
de validité.

---

## 3. Biais spatial / représentativité — 🟡 modéré (et pas d'abus de généralisation dans la doc actuelle)

### Verdict : risque réel de déséquilibre d'échantillonnage ; en revanche la doc ne sur-vend pas la généralisation — à verrouiller pour que ça reste vrai.

- **Composition du jeu** : les 5 zones nommées (`gee_extraction.py:43-51`)
  couvrent le centre dense, l'île de Nantes, le nord, Saint-Herblain et Rezé —
  un biais urbain assumé. Mais `nantes_metropole` (40 × 34 km) est aux ¾
  rurale/périurbaine : si elle est incluse à l'entraînement, la loss moyenne
  est dominée par des patchs de champs et de bocage, faciles (LST homogène,
  NDVI élevé), et le modèle peut être médiocre là où ça compte (cœurs d'îlots
  minéralisés) tout en affichant une excellente moyenne. Le seul filtre de
  `prepare_patches.py` est `--min-valid` (fraction de pixels finis) — aucun
  contrôle de composition.
- **Correction concrète** : calculer par patch un descripteur de tissu (les
  canaux 6–7, canopée et bâti, sont déjà dans X : moyenne du canal bâti et du
  canal canopée par patch suffisent), classer chaque patch en
  {dense, pavillonnaire, industriel/commercial, végétal, eau} par simples
  seuils, puis (i) **rapporter la composition du dataset** dans le README,
  (ii) équilibrer par pondération d'échantillonnage
  (`WeightedRandomSampler`) ou sous-échantillonnage des patchs ruraux,
  (iii) stratifier les métriques par classe (cf. §6).
- **Généralisation hors Nantes** : après relecture, ni le README, ni la spec,
  ni l'app ne prétendent que le modèle vaut ailleurs — tout est explicitement
  « Nantes Métropole ». **Pas d'abus à corriger aujourd'hui.** Pour que ça
  reste vrai, ajouter une ligne explicite de domaine de validité (README +
  spec §1) : « Modèle entraîné exclusivement sur Nantes Métropole,
  étés 2022–2026 ; toute application à un autre territoire exige un
  ré-entraînement ou une validation locale. » Raison technique supplémentaire
  de ne pas transférer tel quel : les entrées ne sont pas normalisées
  (réflectances 0–1, NDVI −1–1, canopée/bâti dans une unité Open Data locale)
  et le réseau utilise des BatchNorm — le moindre décalage de distribution
  (autre fournisseur de canopée, autre climat) casse silencieusement les
  prédictions.

---

## 4. Validité de la "Downsampled Consistency Loss" — 🔴 critique, mais pas là où on l'attend

### Verdict : l'approximation avg-pool est réelle mais secondaire ; le problème de fond est que, sur données réelles, la loss totale n'apprend pas de la super-résolution du tout.

**a) L'hypothèse « avg_pool 10×10 = mesure Landsat » est physiquement fausse,
et le docstring qui affirme le contraire doit être corrigé.**
`gee_extraction.py:13-15` promet que « l'average-pooling 10×10 retrouve
exactement la mesure Landsat ». Quatre raisons pour lesquelles c'est une
approximation, pas une identité :

1. **Grilles non alignées** : la grille native Landsat est en UTM 30N, l'export
   est en Lambert-93 ancré sur l'emprise du projet. Le rééchantillonnage
   plus-proche-voisin ne préserve les blocs 100 m que si les deux grilles
   coïncident — elles ne coïncident jamais (rotation + décalage sub-pixel).
   Les « blocs » dans Y sont des blocs déformés, à cheval sur les fenêtres
   d'avg-pool.
2. **Le produit ST de la Collection 2 est distribué sur une grille 30 m** (le
   TIRS natif est ~100 m, rééchantillonné par l'USGS avec partage
   d'information inter-pixel). Les « pavés 100 m » supposés par le pooling
   10×10 n'existent même pas dans le produit source : ce sont des pavés 30 m
   issus d'un traitement en cascade.
3. **La PSF du capteur thermique n'est pas une moyenne en boîte** : un pixel
   TIRS intègre le rayonnement avec une réponse spatiale ~gaussienne qui
   déborde du pixel. `avg_pool2d` (boîte parfaite) est un modèle de capteur
   simplifié.
4. **256 n'est pas multiple de 10** : `avg_pool2d(kernel=10, stride=10)`
   ignore les 6 dernières lignes/colonnes de chaque patch (déjà noté dans
   l'audit de cohérence §6.2).

Chacun de ces effets introduit un lissage/biais systématique aux frontières
des blocs. Ordre de grandeur : quelques dixièmes de °C près des forts
gradients (fronts bâti/eau, Loire) — pas négligeable pour des anomalies ICU
de 2–4 °C.

**b) Le vrai problème scientifique : sur données réelles, Y n'est pas une
vérité 10 m, et la loss actuelle (`model.py:76-91`) le paie doublement.**
Y est le Landsat 100 m rééchantillonné NN à 10 m. Donc :

- `loss_pixel = L1(y_hat, y)` à 10 m pousse le réseau à reproduire **l'image
  en blocs** — l'optimum de ce terme est la sortie bicubique-plate, l'inverse
  de l'objectif de super-résolution ;
- `loss_consistency = MSE(avg_pool(y_hat), avg_pool(y))` est alors
  **quasi redondante** : c'est une version agrégée de l'information déjà
  contenue dans la L1 pixel. Elle n'apporte une contrainte distincte que dans
  le monde des notebooks, où Y est une vérité 10 m *synthétique*.

Autrement dit : le montage de loss est celui d'un problème **supervisé avec
vérité haute résolution**, appliqué à un problème qui est en réalité du
**downscaling faiblement supervisé** (aucune vérité 10 m n'existe). En l'état,
rien dans la fonction de coût ne récompense un détail à 10 m juste, et rien ne
punit un détail inventé — la seule chose mesurée est la fidélité aux blocs.

### Correction concrète

1. **Corriger le docstring** de `gee_extraction.py` (supprimer « retrouve donc
   exactement ») et documenter l'approximation.
2. **Restructurer la loss pour le cas réel** : supprimer (ou α→0) la L1 pixel
   contre le Y rééchantillonné ; garder la consistency loss comme **terme
   principal**, calculée proprement : exporter Y **aussi à 100 m natif** (un
   deuxième export, 1 ligne dans `gee_extraction.py`) et comparer
   `avg_pool(y_hat, 10)` à ce raster 100 m au lieu de `avg_pool(y)` — on
   élimine l'aller-retour NN. Ajouter des régularisations qui portent
   l'information haute résolution : SSIM/gradient loss guidée par les canaux
   X (la spec §3.2 l'annonce déjà), voire le discriminateur Pix2Pix évoqué
   §1.1.
3. **Ou** rester supervisé mais avec une vraie référence : pré-entraîner sur
   paires synthétiques (schéma des notebooks), puis affiner avec la seule
   consistency loss sur données réelles — schéma classique en downscaling
   thermique.
4. Pour l'effet de bord : padding réflexif à 260 px avant pooling, ou
   `count_include_pad=False` avec un kernel adapté — 3 lignes.

---

## 5. Absence de baseline — 🔴 critique, confirmé sans réserve

### Verdict : c'est un manque critique, pas un détail — et dans ce projet précis, la baseline risque de gagner.

Aucune baseline nulle part : ni dans la spec, ni dans les scripts, ni dans les
notebooks. Or ce projet est exactement le cas où la baseline est redoutable :

- **La baseline triviale est déjà presque la cible.** Comme Y (l'étalon
  d'évaluation actuel) est le Landsat rééchantillonné, une simple
  **interpolation bicubique du 100 m** obtient par construction un RMSE
  excellent contre Y — probablement **meilleur** que le U-Net, qui ajoute des
  détails que Y ne peut pas confirmer. Si personne ne fait ce calcul, le
  premier relecteur le fera ; s'il est fait et que le U-Net perd, cela ne
  prouve pas que le U-Net est mauvais, cela prouve que **la métrique actuelle
  ne peut pas mesurer la valeur ajoutée** (cf. §6). Ce point à lui seul
  justifie la baseline : elle révèle si le protocole d'évaluation est valide.
- **La baseline sérieuse du domaine existe et est simple** : TsHARP /
  DisTrad — régression LST ~ NDVI ajustée à 100 m puis appliquée à 10 m,
  résidus réinjectés. C'est ~40 lignes de NumPy avec les rasters déjà
  produits, c'est LA référence citée dans toute la littérature de downscaling
  thermique, et elle garantit la cohérence à 100 m par construction.
- Sans ces deux chiffres, « le deep learning apporte X » est inaffirmable, et
  le coût du pipeline (GEE + GPU + maintenance) est injustifiable face à un
  éventuel `scipy.ndimage.zoom`.

### Correction concrète

Créer `baselines.py` avec trois méthodes évaluées **exactement** comme le
modèle (mêmes patchs de test, mêmes métriques, même masque de validité) :

1. bicubique : `zoom(y_100m, 10, order=3)` ;
2. TsHARP : régression linéaire (ou par classe de tissu) LST~NDVI à 100 m,
   prédiction à 10 m, correction résiduelle 100 m ;
3. (optionnel) krigeage avec dérive externe NDVI/bâti si `pykrige` est
   acceptable en dépendance.

Et intégrer le tableau comparatif au README. Toute communication de résultat
sans ce tableau doit être considérée comme non étayée.

---

## 6. Métriques d'évaluation — 🟠 sérieux : insuffisantes, et en partie mal dirigées

### Verdict : problème réel à deux étages — il manque des métriques stratifiées, et la métrique globale actuelle ne mesure pas le bon objet.

État des lieux : la seule métrique du projet est la `val_loss` = L1 moyenne
(`model.py:93-98`). Pas de RMSE, pas de MAE rapportés séparément, pas de
stratification, et la « validation terrain » sur les points de fraîcheur
annoncée dans la spec §5 n'est implémentée nulle part.

- **RMSE/MAE globaux sont insuffisants, précisément pour la raison
  soulevée** : le jeu étant dominé par des surfaces thermiquement homogènes
  (végétation, eau, pavillonnaire aéré), un modèle qui lisse tout minimise la
  moyenne en échouant sur les cœurs d'îlots denses — exactement les zones
  visées par la politique publique. L'erreur moyenne cache l'erreur sur la
  population cible.
- **En plus**, tant que l'étalon est le Y rééchantillonné (cf. §4-§5), même
  stratifié, un bon RMSE mesure la fidélité aux blocs 100 m, pas la justesse
  du détail 10 m. Les deux corrections sont nécessaires, pas alternatives.

### Correction concrète

1. **Stratifier** : dans `validation_step`/un script `evaluate.py`, calculer
   RMSE et MAE **par classe de tissu** (classes dérivées des canaux
   canopée/bâti, cf. §3) et par zone. Logger `val_rmse_dense`,
   `val_rmse_vegetal`, etc. Publier le tableau stratifié, pas la moyenne
   seule.
2. **Métriques orientées décision** : l'usage final est « où sont les îlots ? »
   — évaluer aussi la détection : binariser l'anomalie (LST > moyenne + 2 °C,
   le seuil existe déjà dans `app.py`) et rapporter précision/rappel/IoU des
   zones ICU prédites contre les zones ICU du Landsat agrégé à 100 m.
3. **Implémenter la validation terrain promise** (spec §5) : delta thermique
   sur les points de fraîcheur Open Data vs tissu environnant — c'est la
   seule évaluation du projet qui porte réellement sur le détail infra-100 m,
   elle est décrite depuis le début et jamais codée.
4. Ajouter la cohérence 100 m comme métrique d'éval séparée
   (`rmse_100m = RMSE(avg_pool(y_hat), y_100m)`) : c'est la seule quantité
   dont on connaisse la vérité exacte.

---

## 7. Incertitude et communication des résultats — 🟠 sérieux pour l'usage annoncé

### Verdict : problème réel — le pipeline produit un point estimate sec, présenté dans une app carto sans aucun avertissement.

`predict.py` écrit une seule bande `LST_pred_10m_C` ; `app.py` l'affiche avec
seuil ICU, statistiques et fraction de surface « en ICU » au dixième de % près
— aucune bande d'incertitude, aucun intervalle, aucune mention du domaine de
validité. Pour un livrable qui se destine à l'aménagement urbain, c'est le
profil type de la sur-confiance : la carte à 10 m *a l'air* précise à 10 m,
alors que l'information thermique source est à 100 m et que le détail fin est
une inférence du modèle, non une mesure.

Facteur aggravant propre à ce projet : les détails 10 m sont, par
construction (cf. §4), la partie **la moins contrainte** de la sortie — c'est
exactement là que l'incertitude est maximale, et c'est exactement ce que
l'utilisateur final regardera (« cette rue est-elle plus chaude que
celle-là ? »).

### Correction concrète

1. **Deep ensemble — le meilleur rapport coût/rigueur ici** : entraîner K=5
   modèles avec des seeds différentes (le modèle est petit, 5 entraînements
   sont bon marché), puis `predict.py --checkpoints ckpt1..ckpt5` écrit un
   GeoTIFF 2 bandes : `pred_mean` et `pred_std`. Le code d'inférence existant
   (`sliding_window_predict`) se réutilise tel quel dans une boucle.
   Alternative à modèle unique : tête hétéroscédastique (prédire μ et log σ²,
   loss NLL gaussienne) — plus élégant mais change la loss, à faire après
   stabilisation du §4. Le MC-dropout n'est **pas** disponible gratuitement :
   l'architecture actuelle n'a aucun dropout.
2. **Afficher l'incertitude dans l'app** : couche « écart-type d'ensemble »
   dans le sélecteur de couches d'`app.py`, et masquage (hachures/gris) des
   pixels où σ dépasse un seuil.
3. **Encadré de mise en garde obligatoire dans l'app et le README** :
   « Température de surface (pas de l'air), matinée d'été par ciel clair,
   détail infra-100 m modélisé et non mesuré, non validé in situ. » Quatre
   faits, quatre lignes, et le risque de sur-interprétation en réunion
   d'aménagement chute drastiquement.

---

## 8. Reproductibilité — 🟠 sérieux : aucun des trois verrous n'est en place

### Verdict : problème réel sur les trois plans (seeds, versions, données), aggravé par l'absence de git.

- **Seeds : non.** Les seeds n'existent que dans les notebooks pédagogiques
  (`np.random.seed(42)`, `torch.manual_seed(0)`). Le pipeline réel n'en a
  aucune : `train.py` n'appelle pas `pl.seed_everything`, et surtout le
  `random_split` de `dataset.py:69` est **non seedé** — le partage train/val
  change à chaque exécution. Combiné à l'EarlyStopping sur `val_loss`, deux
  runs identiques donnent deux modèles, deux checkpoints « best » et deux
  chiffres différents, sans qu'aucun paramètre n'ait changé.
- **Versions : non figées.** `requirements.txt` n'a que des bornes minimales
  (`torch>=2.0`, `rasterio>=1.3`...). Aucun lock file, pas de version Python
  documentée.
- **Données : non reproductibles.** Trois causes : (i) la fenêtre 2022–2026
  inclut l'année en cours — relancer l'extraction en septembre ajoute des
  scènes ; (ii) les collections GEE sont re-traitées par l'USGS/ESA (les
  valeurs d'une même scène peuvent changer entre deux exports) ; (iii) les
  emprises `ZONES` sont commentées « approximatives — à ajuster librement »
  (`gee_extraction.py:41-42`) : le dataset dépend d'un état du code non
  versionné… et **le dossier n'est pas un dépôt git**.
- Documentation : bonne (README pipeline complet, spec, notebooks) — c'est le
  point fort. Mais sans seed + versions + scène-list, un tiers reproduit le
  *pipeline*, pas les *résultats*.

### Correction concrète

1. `train.py` : `pl.seed_everything(args.seed, workers=True)` avec
   `--seed 42` par défaut, et `deterministic=True` dans le `Trainer`.
2. `dataset.py` :
   `random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))`
   — en attendant le split spatial du §1, qui devra lui aussi être seedé.
3. Versions : `pip freeze > requirements.lock` dans l'env d'entraînement
   (ou passage à `uv`/`pip-tools`), + version de Python dans le README.
4. Données : faire écrire par `gee_extraction.py` un manifeste JSON par export
   (IDs de scènes `system:index` Landsat et S2, date d'export, bbox snappée,
   arguments CLI) — c'est un `getInfo()` de plus ; figer `--years 2022 2025`
   pour les expériences publiées.
5. `git init` + commit du code et des manifestes (pas des rasters).

---

## Synthèse des verdicts

| # | Point | Problème réel ici ? | Gravité | Correction prioritaire |
|---|-------|--------------------|---------|------------------------|
| 1 | Split train/val/test | Oui — fuite spatiale **+ patchs dupliqués par zones chevauchantes + aucun test set** | 🔴 | Dédup + split par blocs géographiques + hold-out espace×temps |
| 2 | Biais temporel | Oui — biais ciel clair/matinée assumable si documenté ; spec ment sur le filtre « forte chaleur » ; confusion SUHI diurne / UHI nocturne non signalée | 🟠 | Corriger la spec, documenter le domaine de validité, filtre nuages local |
| 3 | Représentativité spatiale | Partiel — déséquilibre rural/urbain si `--zone all` ; pas d'abus de généralisation dans la doc actuelle | 🟡 | Composition du dataset + pondération + phrase de domaine de validité |
| 4 | Consistency loss | Oui — approximation physique réelle (grilles, PSF, produit 30 m, bord 6 px) **et surtout** loss totale incapable d'apprendre la super-résolution sur données réelles | 🔴 | Reformuler la loss (consistency contre Y 100 m natif, α_L1→0, prior structurel) |
| 5 | Absence de baseline | Oui — critique confirmé ; la bicubique gagnerait probablement avec la métrique actuelle | 🔴 | `baselines.py` (bicubique + TsHARP) évalué à protocole identique |
| 6 | Métriques | Oui — L1 moyenne seule ; ni stratification, ni détection ICU, ni la validation terrain promise | 🟠 | RMSE/MAE par classe de tissu + IoU des zones ICU + points de fraîcheur |
| 7 | Incertitude | Oui — point estimate sec dans une app carto sans avertissement | 🟠 | Deep ensemble (mean+std), couche σ dans l'app, encadré de mise en garde |
| 8 | Reproductibilité | Oui — pas de seed pipeline, versions non figées, données mouvantes, pas de git | 🟠 | seed_everything + split seedé + lock file + manifeste de scènes + git |

**Lecture d'ensemble.** Le pipeline est propre techniquement (l'audit de
cohérence l'a établi), mais en l'état les chiffres qu'il produirait ne
démontreraient rien : la validation est contaminée (§1), l'étalon d'évaluation
ne contient pas l'information à 10 m que le modèle prétend produire (§4, §6),
et aucune référence ne permet de chiffrer la valeur ajoutée (§5). Ces trois
points forment un tout : les corriger ensemble (split propre → baseline →
métriques stratifiées + validation terrain) est le chemin critique avant toute
communication de résultats. Les points 2, 3, 7, 8 sont des corrections
rapides et principalement documentaires ou d'outillage — à faire, mais qui ne
conditionnent pas la validité de l'apprentissage lui-même.
