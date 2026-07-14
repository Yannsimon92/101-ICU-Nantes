# Audit scientifique ICU Nantes — Résumé exécutif

*2026-07-07 — version détaillée : [audit_scientifique.md](audit_scientifique.md)*

## Verdict en une phrase

Le pipeline est techniquement cohérent, mais **en l'état, aucun chiffre qu'il
produirait ne serait scientifiquement défendable** : la validation fuit
(patchs voisins *et* dupliqués entre train et val, pas de jeu de test),
l'étalon d'évaluation ne contient pas l'information 10 m que le modèle prétend
produire, et aucune baseline ne permet de prouver une valeur ajoutée.

## Tableau de synthèse

| # | Point | Gravité | Constat dans CE projet | Correction |
|---|-------|:-:|------------------------|------------|
| 1 | Split 80/20 aléatoire | 🔴 | Fuite spatiale confirmée, **aggravée** : les zones GEE se chevauchent (et `nantes_metropole` englobe les 5 autres) → patchs pixel-identiques en train ET val ; aucun test set ; split non seedé | Dédupliquer les patchs, split par blocs géographiques disjoints (dates groupées), hold-out gelé espace×temps |
| 2 | Filtre juin–août <5 % nuages | 🟠 | Biais « ciel clair anticyclonique » assumable si documenté. Mais la spec prétend filtrer les « journées de forte chaleur » — faux, seul les nuages sont filtrés. Et la carte montre le SUHI diurne (~10 h 50), pas l'UHI nocturne visé par les politiques de fraîcheur | Corriger la spec, écrire le domaine de validité, filtre nuages calculé sur l'emprise (pas la scène) |
| 3 | Représentativité spatiale | 🟡 | Avec `--zone all`, le jeu est dominé par le périurbain/rural (loss moyenne flatteuse, cœurs d'îlots mal servis). Bon point : la doc ne prétend PAS généraliser hors Nantes | Publier la composition du dataset, pondérer l'échantillonnage, ajouter une phrase explicite de domaine de validité |
| 4 | Consistency loss | 🔴 | L'« exactitude » avg-pool 10×10 = Landsat est fausse (grilles UTM/L93 non alignées, produit ST livré à 30 m, PSF ≠ boîte, 6 px ignorés). Pire : sur données réelles, Y = Landsat rééchantillonné → la L1 pixel pousse à reproduire les blocs et la consistency loss devient redondante — **rien dans la loss n'apprend le détail 10 m** | Corriger le docstring mensonger ; consistency contre le Y 100 m natif, α_L1→0 sur données réelles, prior structurel (SSIM/gradient) ou pré-entraînement synthétique |
| 5 | Absence de baseline | 🔴 | Confirmé critique. Avec la métrique actuelle, la **bicubique gagnerait probablement** contre le U-Net — ce qui prouve que le protocole d'évaluation ne mesure pas la valeur ajoutée | `baselines.py` : bicubique + TsHARP (régression LST~NDVI), évaluées à protocole strictement identique, tableau dans le README |
| 6 | Métriques | 🟠 | Une seule métrique : L1 moyenne. Ni stratification par tissu, ni détection ICU, ni la « validation terrain points de fraîcheur » promise par la spec §5 (jamais codée) | RMSE/MAE par classe de tissu (dérivable des canaux canopée/bâti), précision/rappel/IoU des zones ICU, implémenter les points de fraîcheur |
| 7 | Incertitude | 🟠 | Point estimate sec ; l'app affiche des % au dixième sans avertissement. Le détail 10 m est précisément la partie la moins contrainte du modèle | Deep ensemble (5 seeds) → bandes mean+std, couche σ dans l'app, encadré « LST surface, matinée claire, détail modélisé non mesuré » |
| 8 | Reproductibilité | 🟠 | Pas de seed dans le pipeline réel (split différent à chaque run), `requirements.txt` non figé, données GEE mouvantes (2026 en cours, retraitements USGS), emprises « à ajuster librement », pas de git | `seed_everything` + split seedé, lock file, manifeste JSON des IDs de scènes exportées, `git init` |

## Chemin critique (dans l'ordre)

1. **Assainir le split** (dédup + blocs géographiques + test gelé) — sans ça,
   tout chiffre est contaminé.
2. **Baselines** bicubique + TsHARP — sans ça, aucune valeur ajoutée n'est
   démontrable (et le résultat dira si l'évaluation elle-même est valide).
3. **Reformuler loss + évaluation** autour du fait qu'aucune vérité 10 m
   n'existe : consistency 100 m native comme terme principal, métriques
   stratifiées, validation terrain points de fraîcheur.
4. Ensuite seulement : incertitude (ensemble), documentation du domaine de
   validité, verrous de reproductibilité — rapides et indépendants.

## Ce qui est sain

Pipeline bout-en-bout cohérent (cf. `audit_coherence.md`), alignement
géométrique X/Y soigné, doc honnête sur le périmètre nantais (pas de
sur-vente de généralisation), notebooks pédagogiques de qualité, inférence
pleine zone propre (fenêtre glissante + fusion de Hann).
