"""Dataset et DataModule PyTorch pour les patchs ICU Nantes.

Identique à la section 4.1 de icu_nantes_deep_learning_spec.md.
Les patchs X/Y sont produits par prepare_patches.py à partir des exports GEE.
"""

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
