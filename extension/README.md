# Extension — Super-résolution U-Net (100 m → 10 m)

Cette extension a été **prototypée puis écartée du périmètre principal** du
projet : en l'absence de vérité terrain à 10 m, la validation du détail généré
n'est pas défendable (voir `audit_scientifique.md` et la section « Travaux
d'extension étudiés » du README racine). Le prototype et l'analyse critique
sont conservés ici — matière d'entretien, pas livrable.

## Contenu

| Fichier | Rôle |
|---|---|
| `model.py` | `GuidedSuperResUNet` (LightningModule, 3 étages symétriques) |
| `dataset.py` | `NantesICUDataset` + `ICUDataModule` (patchs 256×256) |
| `train.py` | Entraînement (L1 + consistency 100 m, mixed precision) |
| `predict.py` | Inférence fenêtre glissante + fusion de Hann |
| `prepare_patches.py` | Tuilage 256×256 + fusion canopée/bâti (mode 10 m) |
| `app_unet.py` | App Streamlit d'origine (prédictions 10 m vs Landsat 100 m) |
| `02_comprendre_le_modele_UNet.ipynb` | Notebook pédagogique d'architecture U-Net |
| `icu_nantes_deep_learning_spec.md` | Spec complète (U-Net orientée) |
| `audit_coherence.md` | Audit bout-en-bout du pipeline U-Net |
| `audit_scientifique.md` / `_resume.md` | Audit scientifique (verrous critiques) |
| `icu_roadmap_v2.md` | Feuille de route « réparer la super-résolution » (alternative non suivie) |

## Pour relancer cette extension

Le mode `--resolution 10` de `../gee_extraction.py` exporte toujours les
données attendues (stack S2 5 bandes à 10 m + LST rééchantillonnée à 10 m) ;

```bash
python ../gee_extraction.py --zone nantes_metropole --resolution 10 --project <projet>
python prepare_patches.py --raw-dir ../data/raw --out-dir ../data/patches \
    --canopee ../data/morpho/canopee_10m.tif --bati ../data/morpho/bati_10m.tif
python train.py --data-dir ../data/patches --epochs 50
python predict.py "../data/patches/full/X_full_*.tif" --checkpoint <best.ckpt>
```

Voir `icu_roadmap_v2.md` pour la feuille de route complète (réparer la fuite
spatiale, baselines, loss v2, incertitude) avant de reconsidérer cette piste.
