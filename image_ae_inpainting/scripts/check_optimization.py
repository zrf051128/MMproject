import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage import data
from skimage.transform import resize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mask_generator import random_mask, apply_mask
from src.baselines import opencv_telea, tv_inpainting
from src.metrics import compute_all_metrics
from src.optimization import optimize_image_with_ae_tv


def save_image(image, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_path, image, cmap="gray", vmin=0, vmax=1)


def save_comparison_figure(images, titles, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    plt.figure(figsize=(3 * n, 3))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_loss_curve(log, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(log["total_loss"], label="total")
    plt.plot(log["data_loss"], label="data")
    plt.plot(log["ae_loss"], label="ae")
    plt.plot(log["tv_loss"], label="tv")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_psnr_curve(log, save_path):
    if len(log["step"]) == 0:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(log["step"], log["psnr"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Temporary Tiny AE for debugging
# This part will be replaced by A's train_autoencoder later.
# ============================================================

def extract_visible_patches_np(image, mask, patch_size=8, stride=4, visible_threshold=0.8):
    """
    Extract patches whose visible ratio is larger than visible_threshold.

    image: initialized image, [H, W]
    mask:  1 observed, 0 missing
    """
    patches = []
    H, W = image.shape

    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            patch = image[top:top + patch_size, left:left + patch_size]
            mask_patch = mask[top:top + patch_size, left:left + patch_size]

            visible_ratio = mask_patch.mean()

            if visible_ratio >= visible_threshold:
                patches.append(patch)

    if len(patches) == 0:
        # fallback: use all patches from initialized image
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                patch = image[top:top + patch_size, left:left + patch_size]
                patches.append(patch)

    patches = np.asarray(patches, dtype=np.float32)
    patches = patches[:, None, :, :]  # [N, 1, patch_size, patch_size]

    return patches


def train_debug_tiny_ae(
    y,
    mask,
    patch_size=8,
    stride=4,
    latent_dim=16,
    visible_threshold=0.8,
    epochs=200,
    lr=1e-3,
    batch_size=128,
    verbose=True,
):
    """
    Temporary AE training for debugging.

    Formal experiment should use A's train_autoencoder().
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    class TinyPatchAE(nn.Module):
        def __init__(self, patch_size=8, latent_dim=16):
            super().__init__()
            dim = patch_size * patch_size

            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, dim),
                nn.Sigmoid(),
            )

            self.patch_size = patch_size

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            out = out.view(-1, 1, self.patch_size, self.patch_size)
            return out

    # Use Telea initialized image for temporary AE training
    x_init = opencv_telea(y, mask)

    patches = extract_visible_patches_np(
        x_init,
        mask,
        patch_size=patch_size,
        stride=stride,
        visible_threshold=visible_threshold
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patches_t = torch.tensor(patches, dtype=torch.float32)

    dataset = TensorDataset(patches_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ae = TinyPatchAE(patch_size=patch_size, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ae.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in loader:
            patch_batch = batch[0].to(device)

            optimizer.zero_grad()
            recon = ae(patch_batch)
            loss = criterion(recon, patch_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * patch_batch.size(0)

        epoch_loss /= len(dataset)

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"[Debug AE] epoch {epoch:04d} | loss={epoch_loss:.6f}")

    return ae


def main():
    save_dir = Path("results/debug_optimization")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Cameraman image
    x_gt = data.camera().astype(np.float32) / 255.0
    x_gt = resize(x_gt, (128, 128), anti_aliasing=True).astype(np.float32)

    # Generate random 30% mask
    mask = random_mask(x_gt.shape, missing_ratio=0.3, seed=0)
    y = apply_mask(x_gt, mask)

    save_image(x_gt, save_dir / "original.png")
    save_image(mask, save_dir / "mask.png")
    save_image(y, save_dir / "corrupted.png")

    rows = []

    # --------------------------------------------------------
    # Corrupted
    # --------------------------------------------------------
    corrupted_metrics = compute_all_metrics(y, x_gt)
    rows.append({
        "method": "corrupted",
        "psnr": corrupted_metrics["psnr"],
        "ssim": corrupted_metrics["ssim"],
        "rmse": corrupted_metrics["rmse"],
        "mae": corrupted_metrics["mae"],
        "runtime": 0.0,
    })

    # --------------------------------------------------------
    # Telea initialization
    # --------------------------------------------------------
    start = time.time()
    x_telea = opencv_telea(y, mask)
    runtime = time.time() - start

    telea_metrics = compute_all_metrics(x_telea, x_gt)

    rows.append({
        "method": "opencv_telea",
        "psnr": telea_metrics["psnr"],
        "ssim": telea_metrics["ssim"],
        "rmse": telea_metrics["rmse"],
        "mae": telea_metrics["mae"],
        "runtime": runtime,
    })

    save_image(x_telea, save_dir / "opencv_telea.png")

    # --------------------------------------------------------
    # TV baseline
    # --------------------------------------------------------
    start = time.time()

    x_tv, tv_log = tv_inpainting(
        y,
        mask,
        lam_tv=1e-3,
        lr=1e-2,
        steps=500,
        init="telea",
        verbose=False
    )

    runtime = time.time() - start

    tv_metrics = compute_all_metrics(x_tv, x_gt)

    rows.append({
        "method": "tv",
        "psnr": tv_metrics["psnr"],
        "ssim": tv_metrics["ssim"],
        "rmse": tv_metrics["rmse"],
        "mae": tv_metrics["mae"],
        "runtime": runtime,
    })

    save_image(x_tv, save_dir / "tv.png")
    np.save(save_dir / "tv_log.npy", tv_log)

    # --------------------------------------------------------
    # Temporary AE training
    # Later replace this with A's train_autoencoder()
    # --------------------------------------------------------
    print("Training temporary debug AE...")

    ae = train_debug_tiny_ae(
        y,
        mask,
        patch_size=8,
        stride=4,
        latent_dim=16,
        visible_threshold=0.8,
        epochs=200,
        lr=1e-3,
        batch_size=128,
        verbose=True
    )

    # --------------------------------------------------------
    # Ours: AE + TV optimization
    # --------------------------------------------------------
    print("Running Ours AE+TV optimization...")

    start = time.time()

    x_ours, ours_log = optimize_image_with_ae_tv(
        y=y,
        mask=mask,
        ae=ae,
        patch_size=8,
        stride=4,
        lam_ae=1e-2,
        lam_tv=1e-3,
        lr=1e-2,
        steps=800,
        init="telea",
        ae_input_format="image",
        fix_observed=False,
        x_gt=x_gt,
        log_interval=20,
        verbose=True
    )

    runtime = time.time() - start

    ours_metrics = compute_all_metrics(x_ours, x_gt)

    rows.append({
        "method": "ours_debug_ae_tv",
        "psnr": ours_metrics["psnr"],
        "ssim": ours_metrics["ssim"],
        "rmse": ours_metrics["rmse"],
        "mae": ours_metrics["mae"],
        "runtime": runtime,
    })

    save_image(x_ours, save_dir / "ours_debug_ae_tv.png")
    np.save(save_dir / "ours_log.npy", ours_log)

    save_loss_curve(ours_log, save_dir / "ours_loss_curve.png")
    save_psnr_curve(ours_log, save_dir / "ours_psnr_curve.png")

    # --------------------------------------------------------
    # Save comparison figure and CSV
    # --------------------------------------------------------
    images = [
        x_gt,
        mask,
        y,
        x_telea,
        x_tv,
        x_ours,
    ]

    titles = [
        "Original",
        "Mask",
        "Corrupted",
        "OpenCV Telea",
        "TV",
        "Ours Debug AE+TV",
    ]

    save_comparison_figure(
        images,
        titles,
        save_dir / "optimization_comparison.png"
    )

    df = pd.DataFrame(rows)
    csv_path = save_dir / "optimization_metrics.csv"
    df.to_csv(csv_path, index=False)

    print()
    print("Optimization results:")
    print(df)
    print()
    print(f"Results saved to: {save_dir}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Comparison figure saved to: {save_dir / 'optimization_comparison.png'}")


if __name__ == "__main__":
    main()
