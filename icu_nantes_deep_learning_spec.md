# PROMPT-SPECIFICATION INFRASTRUCTURE : ARCHITECTURE DE DEEP LEARNING POUR L'IDENTIFICATION DES ÎLOTS DE CHALEUR URBAINS (ICU) À NANTES

---

## 1. CONTEXTE GLOBAL & STRATÉGIE TECHNIQUE

Ce document sert de base de connaissances et de spécification technique exhaustive pour concevoir, coder et déployer une solution de Deep Learning appliquée à la détection et à la super-résolution des Îlots de Chaleur Urbains (ICU) sur le territoire de **Nantes Métropole**.

### 1.1 Le Paradoxe de la Résolution (Le Verrou Technique)

* **La Cible (Y) :** Les capteurs thermiques satellitaires gratuits (Landsat 8/9 TIRS) fournissent la température de surface du sol (*Land Surface Temperature* - LST) à une résolution brute de **100 mètres par pixel**. Cette échelle est trop grossière pour capturer les micro-dynamiques urbaines (rues, alignements d'arbres, cours d'écoles).
* **Les Prédicteurs (X) :** L'Open Data de Nantes Métropole et Sentinel-2 fournissent des données géospatiales, morphologiques et optiques à haute résolution (**0.5m à 10m**).
* **L'Approche :** Utiliser des architectures de **Super-Résolution (SRCNN, EDSR)** ou de **Traduction d'Image à Image (Pix2Pix / Conditional GANs)** pour contraindre des données macro-thermiques (100m) à l'aide de caractéristiques micro-structurelles (10m) afin de générer une carte des ICU à haute résolution (échelle de la rue).

### 1.2 Schéma Conceptuel de la Pipeline de Données

```
[Données Macro (Y)] -> Landsat 8/9 LST (100m) -----------\
                                                          +---> [Modèle Deep Learning] -> [Carte ICU Haute-Résolution (10m)]
[Données Micro (X)] -> Sentinel-2 RGB+NDVI (10m) --------/
                     -> Open Data Nantes (Canopée, Bâti) /
```

---

## 2. INVENTAIRE DES SOURCES DE DONNÉES & FEATURE ENGINEERING

### 2.1 Variables d'Entrée (Features X)

Pour chaque pixel ou patch cible, le modèle doit ingérer une matrice multispectrale et morphologique multi-canaux :

1. **Imagerie Optique (Sentinel-2) :** Bandes B2 (Bleu), B3 (Vert), B4 (Rouge), B8 (Proche Infrarouge - NIR). Résolution : 10m.
2. **Indices Biophysiques Évolués :**
   * **NDVI (Normalized Difference Vegetation Index) :** Pour quantifier la densité de végétation.

     $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

   * **NDWI (Normalized Difference Water Index) :** Pour capter l'humidité des sols et les masses d'eau.

     $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$

   * **NDBI (Normalized Difference Built-Up Index) :** Pour isoler les surfaces imperméabilisées et bâties.

     $$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}$$

3. **Données Morphologiques Urbanistiques (Nantes Métropole Open Data & IGN) :**
   * **Couche Canopée (Rasterisé à 10m) :** Densité et hauteur de la couverture arborée.
   * **Emprise & Hauteur du Bâti (BD TOPO / Open Data) :** Indicateur d'effet "canyon urbain" (accumulation de chaleur par manque de circulation d'air).
   * **Nature des Revêtements :** Taux d'imperméabilisation par zone.

### 2.2 Donnée Cible (Target Y)

* **Source :** Landsat 8/9 Band 10 (Thermal Infrared Sensor - TIRS).
* **Traitement GEE (Google Earth Engine) :** Correction atmosphérique et conversion de la radiance en Température de Surface du Sol (LST) exprimée en Celsius ou Kelvin.
* **Filtre temporel :** Fenêtre stricte (Juin - Août) sur les années 2022 à 2026, ciblant uniquement les journées de forte chaleur sans couverture nuageuse (< 5%).

---

## 3. ARCHITECTURE DU MODÈLE DE DEEP LEARNING

Le modèle principal implémenté est un **U-Net Modifié** agissant comme un réseau de super-résolution guidée, ou un couple Générateur/Discriminateur de type **Pix2Pix**.

### 3.1 Spécifications de l'Architecture U-Net Adaptée

* **Entrée :** Tensor de forme `(BatchSize, C, H, W)` où C = 7 (RGB + NDVI + NDWI + Canopée + Densité Bâti) et (H, W) = (256, 256) à une résolution de 10m.
* **Sortie :** Tensor de forme `(BatchSize, 1, H, W)` représentant la LST prédite à la résolution fine de 10m.
* **Mécanisme de Guidage :** Downsampling de la prédiction haute-résolution pour calculer une perte de cohérence spatiale par rapport à la cible Landsat réelle à 100m.

### 3.2 Fonction de Perte Multi-Objectif (Loss Function)

Pour s'assurer que le modèle ne génère pas de structures aberrantes tout en respectant la physique thermique globale, la fonction de perte combine :

1. **Pixel Loss (L1/MAE) :** Pour la fidélité globale des températures.
2. **Structural Similarity Index (SSIM Loss) :** Pour conserver la cohérence des structures et des contrastes thermiques locaux.
3. **Downsampled Consistency Loss :** Une perte MSE calculée entre la prédiction sous-échantillonnée (par *Average Pooling* 10×10) et la vraie image Landsat à 100m.

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{L1} + \beta \mathcal{L}_{SSIM} + \gamma \mathcal{L}_{consistency}$$

---

## 4. CODE SOURCE DE PRODUCTION (SCRIPTS COMPLETS)

### 4.1 Pipeline de Prétraitement et Datamodule PyTorch (`dataset.py`)

```python
import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class NantesICUDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Répertoire attendu :
        data_dir/
            X/ -> Images multispectrales empilées (RGB, NDVI, NDWI, Canopée, Bâti) - Résolution 10m [256x256]
            Y/ -> Images thermiques Landsat LST - Résolution 100m rééchantillonnées à 10m pour l'alignement [256x256]
        """
        self.data_dir = data_dir
        self.x_dir = os.path.join(data_dir, "X")
        self.y_dir = os.path.join(data_dir, "Y")
        self.file_names = sorted(os.listdir(self.x_dir))
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        # Lecture du fichier features (X)
        with rasterio.open(os.path.join(self.x_dir, file_name)) as src:
            x_data = src.read().astype(np.float32)  # Shape: (C, H, W)

        # Lecture du fichier target (Y)
        with rasterio.open(os.path.join(self.y_dir, file_name)) as src:
            y_data = src.read(1).astype(np.float32)  # Shape: (H, W)
            y_data = np.expand_dims(y_data, axis=0)  # Shape: (1, H, W)

        # Conversion en Tenseurs PyTorch
        x_tensor = torch.tensor(x_data)
        y_tensor = torch.tensor(y_data)

        # Remplacer les éventuels NaNs par des zéros (sécurité géospatiale)
        x_tensor = torch.nan_to_num(x_tensor, nan=0.0)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0)

        if self.transform:
            x_tensor, y_tensor = self.transform(x_tensor, y_tensor)

        return x_tensor, y_tensor


class ICUDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = NantesICUDataset(self.data_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
```

### 4.2 Architecture et Entraînement avec PyTorch Lightning (`model.py`)

> **Note (bug corrigé)** : la version initiale de ce document ne comportait que 2 étages de descente sans pooling avant le bottleneck, mais 2 étages de remontée — les skip-connections `torch.cat` étaient donc spatialement désalignées et le forward plantait. L'architecture ci-dessous, avec **3 étages de descente/remontée symétriques**, est la version corrigée et validée dans le notebook `02_comprendre_le_modele_UNet.ipynb`.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class GuidedSuperResUNet(pl.LightningModule):
    def __init__(self, in_channels=7, out_channels=1, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Encoder (3 étages de descente)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))  # bottleneck

        # Decoder (3 étages de remontée symétriques)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)  # 256 (up) + 256 (skip x3)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)  # 128 (up) + 128 (skip x2)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   # 64 (up) + 64 (skip x1)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

        self.criterion_pixel = nn.L1Loss()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        b = self.down3(x3)

        t1 = self.up1(b)
        t1 = torch.cat([t1, x3], dim=1)  # skip connection avec x3
        t1 = self.conv_up1(t1)

        t2 = self.up2(t1)
        t2 = torch.cat([t2, x2], dim=1)  # skip connection avec x2
        t2 = self.conv_up2(t2)

        t3 = self.up3(t2)
        t3 = torch.cat([t3, x1], dim=1)  # skip connection avec x1
        t3 = self.conv_up3(t3)

        return self.outc(t3)  # Output haute résolution 10m, même (H, W) que l'entrée

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Perte Pixel L1 à haute résolution
        loss_pixel = self.criterion_pixel(y_hat, y)

        # Contrainte de cohérence spatiale (Downsampling 10x10 pour simuler le 100m Landsat)
        y_hat_downscaled = nn.functional.avg_pool2d(y_hat, kernel_size=10, stride=10)
        y_downscaled = nn.functional.avg_pool2d(y, kernel_size=10, stride=10)
        loss_consistency = nn.functional.mse_loss(y_hat_downscaled, y_downscaled)

        total_loss = loss_pixel + 0.5 * loss_consistency

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion_pixel(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
```

---

## 5. RECOMMANDATIONS CLÉS POUR L'INTERACTION AVEC L'IA

Pour maximiser l'efficacité de Claude lors de la génération de code additionnel :

* **Génération d'Interface (Streamlit) :** Demander l'intégration de `streamlit-folium` pour superposer les couches d'ICU générées au format GeoJSON sur une carte de fond de plan nantaise (CartoDB Positron).
* **Gestion de la mémoire GPU :** Configurer un entraînement en précision mixte (`precision=16-mixed` dans Lightning) pour éviter l'explosion de la VRAM avec les rasters géospatiaux.
* **Validation Terrain :** Utiliser le dataset Open Data des "points de fraîcheur de Nantes" pour créer une métrique d'évaluation personnalisée (ex : est-ce que le modèle prédit bien un delta thermique négatif d'au moins 2°C sur ces zones par rapport au tissu urbain environnant ?).

---

## 6. PROMPT D'INGESTION DIRECTE POUR L'IA (MÉTA-PROMPT)

> Tu es un ingénieur expert en Deep Learning et SIG (Systèmes d'Information Géographique). Tu viens de prendre connaissance des spécifications techniques du projet d'identification et super-résolution des ICU à Nantes décrites ci-dessus. En te basant sur ces éléments, génère l'ensemble des scripts manquants, la stratégie complète d'inférence, le script d'extraction Google Earth Engine en Python, et l'application Streamlit finale permettant de visualiser le projet à Nantes.
