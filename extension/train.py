"""Entraînement du U-Net de super-résolution guidée (PyTorch Lightning).

Exemple :
    python train.py --data-dir data/patches --epochs 50 --batch-size 16
"""

import argparse
import pytorch_lightning as pl

from dataset import ICUDataModule
from model import GuidedSuperResUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/patches")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--in-channels", type=int, default=7)
    args = ap.parse_args()

    datamodule = ICUDataModule(args.data_dir, batch_size=args.batch_size,
                               num_workers=args.num_workers)
    model = GuidedSuperResUNet(in_channels=args.in_channels, out_channels=1, lr=args.lr)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="16-mixed",  # cf. spec §5 : limite la VRAM sur les rasters
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min",
                                         save_top_k=1, filename="icu-unet-{epoch}-{val_loss:.3f}"),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"Meilleur checkpoint : {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
