"""Architecture U-Net de super-résolution guidée pour les ICU de Nantes.

Version corrigée (3 étages de descente/remontée symétriques), identique au
notebook 02_comprendre_le_modele_UNet.ipynb et à la section 4.2 de
icu_nantes_deep_learning_spec.md.
"""

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
