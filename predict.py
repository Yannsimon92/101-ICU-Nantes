"""Inférence pleine zone par fenêtre glissante avec fusion pondérée.

Prend un stack 7 canaux pleine zone (produit par prepare_patches.py, dossier
full/) et génère la carte LST prédite à 10 m au format GeoTIFF, directement
visualisable dans l'app Streamlit (app.py).

Les fenêtres 256x256 se recouvrent (stride < 256) et sont fusionnées avec une
pondération de Hann 2D : pas d'effet de damier aux jointures.

Exemples :
    python predict.py data/patches/full/X_full_nantes_centre_2023-07-08.tif \
        --checkpoint lightning_logs/version_0/checkpoints/best.ckpt
    # batch sur toutes les zones préparées :
    python predict.py "data/patches/full/X_full_*.tif" --checkpoint best.ckpt
"""

import argparse
import glob
import os

import numpy as np
import rasterio
import torch

from model import GuidedSuperResUNet


def load_model(checkpoint, in_channels, device):
    if checkpoint:
        if checkpoint.endswith(".ckpt"):
            model = GuidedSuperResUNet.load_from_checkpoint(checkpoint, map_location=device)
        else:
            model = GuidedSuperResUNet(in_channels=in_channels)
            model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        print("ATTENTION : aucun --checkpoint fourni, poids aléatoires (démo du pipeline uniquement)")
        model = GuidedSuperResUNet(in_channels=in_channels)
    return model.to(device).eval()


def hann_weight(size):
    w1 = np.hanning(size + 2)[1:-1]  # évite les zéros stricts aux bords
    return (np.outer(w1, w1) + 1e-3).astype(np.float32)


def sliding_window_predict(model, x, patch, stride, device, batch_size=8):
    """x : (C, H, W) float32 sans NaN. Retourne la prédiction (H, W)."""
    c, h, w = x.shape
    pad_h, pad_w = max(0, patch - h), max(0, patch - w)
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        _, h_p, w_p = x.shape
    else:
        h_p, w_p = h, w

    rows = sorted({min(r, h_p - patch) for r in range(0, h_p - patch + stride, stride)})
    cols = sorted({min(cc, w_p - patch) for cc in range(0, w_p - patch + stride, stride)})

    acc = np.zeros((h_p, w_p), dtype=np.float32)
    weights = np.zeros((h_p, w_p), dtype=np.float32)
    hann = hann_weight(patch)

    positions = [(r, cc) for r in rows for cc in cols]
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            chunk = positions[i:i + batch_size]
            batch = torch.from_numpy(
                np.stack([x[:, r:r + patch, cc:cc + patch] for r, cc in chunk])
            ).to(device)
            preds = model(batch).squeeze(1).cpu().numpy()  # (B, patch, patch)
            for (r, cc), pred in zip(chunk, preds):
                acc[r:r + patch, cc:cc + patch] += pred * hann
                weights[r:r + patch, cc:cc + patch] += hann

    return (acc / weights)[:h, :w]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x_full", help="GeoTIFF X 7 canaux pleine zone (glob accepté)")
    ap.add_argument("--checkpoint", default=None, help="checkpoint .ckpt Lightning ou state_dict .pt")
    ap.add_argument("--out-dir", default="data/predictions")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.x_full)) or [args.x_full]
    os.makedirs(args.out_dir, exist_ok=True)

    model = None
    for path in paths:
        with rasterio.open(path) as src:
            x = src.read(masked=True).filled(np.nan).astype(np.float32)
            profile = src.profile.copy()

        if model is None:
            model = load_model(args.checkpoint, in_channels=x.shape[0], device=args.device)

        invalid = ~np.isfinite(x).all(axis=0)
        x = np.nan_to_num(x, nan=0.0)

        pred = sliding_window_predict(model, x, args.patch, args.stride,
                                      args.device, args.batch_size)
        pred[invalid] = np.nan

        name = os.path.basename(path).replace("X_full_", "pred_lst_")
        out_path = os.path.join(args.out_dir, name)
        profile.update(count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(pred[None].astype(np.float32))
            dst.set_band_description(1, "LST_pred_10m_C")
        print(f"{path} -> {out_path} "
              f"(LST {np.nanmin(pred):.1f} à {np.nanmax(pred):.1f} °C)")


if __name__ == "__main__":
    main()
