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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.mask_generator import random_mask, apply_mask
from src.metrics import compute_all_metrics
from src.baselines import (
    opencv_telea,
    opencv_ns,
    tv_inpainting,
    wavelet_sparse_inpainting,
    kmeans_patch_prior_inpainting,
)
from src.optimization import optimize_image_with_ae_tv


# ============================================================
# Basic saving utilities
# ============================================================

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
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_loss_curve(log, save_path, title="Loss Curve"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))

    if "total_loss" in log:
        plt.plot(log["total_loss"], label="total")
    if "data_loss" in log:
        plt.plot(log["data_loss"], label="data")
    if "ae_loss" in log:
        plt.plot(log["ae_loss"], label="ae")
    if "tv_loss" in log:
        plt.plot(log["tv_loss"], label="tv")
    if "patch_error" in log:
        plt.plot(log["patch_error"], label="patch_error")

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_psnr_curve(log, save_path):
    if "step" not in log or "psnr" not in log:
        return
    if len(log["step"]) == 0:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(log["step"], log["psnr"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.title("Ours PSNR Curve")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Temporary Debug AE
# Later replace this part with A's formal train_autoencoder().
# ============================================================

def extract_visible_patches_np(
    image,
    mask,
    patch_size=8,
    stride=4,
    visible_threshold=0.8
):
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
    verbose=True
):
    """
    Temporary AE training for debugging.

    正式实验时，把这个函数替换成 A 的 train_autoencoder().
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

    ae = TinyPatchAE(
        patch_size=patch_size,
        latent_dim=latent_dim
    ).to(device)

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


# ============================================================
# Run one method
# ============================================================

def run_one_method(method_name, func, y, mask, x_gt, save_dir):
    print(f"Running {method_name}...")

    start_time = time.time()
    result = func(y, mask)
    runtime = time.time() - start_time

    if isinstance(result, tuple):
        x_hat, log = result
    else:
        x_hat = result
        log = None

    metrics = compute_all_metrics(x_hat, x_gt)

    row = {
        "method": method_name,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "runtime": runtime,
    }

    print(f"Results for {method_name}:")
    print(f"  PSNR: {row['psnr']:.6f}")
    print(f"  SSIM: {row['ssim']:.6f}")
    print(f"  RMSE: {row['rmse']:.6f}")
    print(f"  MAE:  {row['mae']:.6f}")
    print(f"  Time: {row['runtime']:.6f} seconds")
    print()

    save_image(x_hat, save_dir / f"{method_name}.png")

    if log is not None:
        np.save(save_dir / f"{method_name}_log.npy", log)

    return x_hat, row, log


# ============================================================
# Main
# ============================================================

def main():
    save_dir = PROJECT_ROOT / "results" / "check_all_methods_random30"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load image
    # --------------------------------------------------------
    x_gt = data.camera().astype(np.float32) / 255.0
    x_gt = resize(
        x_gt,
        (128, 128),
        anti_aliasing=True
    ).astype(np.float32)

    # --------------------------------------------------------
    # 2. Generate same random 30% mask for all methods
    # --------------------------------------------------------
    mask = random_mask(
        x_gt.shape,
        missing_ratio=0.3,
        seed=0
    )

    y = apply_mask(x_gt, mask)

    save_image(x_gt, save_dir / "original.png")
    save_image(mask, save_dir / "mask.png")
    save_image(y, save_dir / "corrupted.png")

    rows = []
    restored_images = []
    restored_titles = []

    # --------------------------------------------------------
    # 3. Corrupted image metrics
    # --------------------------------------------------------
    corrupted_metrics = compute_all_metrics(y, x_gt)

    corrupted_row = {
        "method": "corrupted",
        "psnr": corrupted_metrics["psnr"],
        "ssim": corrupted_metrics["ssim"],
        "rmse": corrupted_metrics["rmse"],
        "mae": corrupted_metrics["mae"],
        "runtime": 0.0,
    }

    rows.append(corrupted_row)

    print("Corrupted image metrics:")
    print(f"  PSNR: {corrupted_row['psnr']:.6f}")
    print(f"  SSIM: {corrupted_row['ssim']:.6f}")
    print(f"  RMSE: {corrupted_row['rmse']:.6f}")
    print(f"  MAE:  {corrupted_row['mae']:.6f}")
    print()

    # --------------------------------------------------------
    # 4. Baseline 1: OpenCV Telea
    # --------------------------------------------------------
    x_hat, row, log = run_one_method(
        method_name="opencv_telea",
        func=lambda yy, mm: opencv_telea(
            yy,
            mm,
            radius=3
        ),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )

    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("OpenCV Telea")

    # --------------------------------------------------------
    # 5. Baseline 2: OpenCV Navier-Stokes
    # --------------------------------------------------------
    x_hat, row, log = run_one_method(
        method_name="opencv_ns",
        func=lambda yy, mm: opencv_ns(
            yy,
            mm,
            radius=3
        ),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )

    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("OpenCV NS")

    # --------------------------------------------------------
    # 6. Baseline 3: TV inpainting
    # --------------------------------------------------------
    x_hat, row, tv_log = run_one_method(
        method_name="tv",
        func=lambda yy, mm: tv_inpainting(
            yy,
            mm,
            lam_tv=1e-3,
            lr=1e-2,
            steps=500,
            init="telea",
            verbose=False
        ),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )

    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("TV")

    if tv_log is not None:
        save_loss_curve(
            tv_log,
            save_dir / "tv_loss_curve.png",
            title="TV Loss Curve"
        )

    # --------------------------------------------------------
    # 7. Baseline 4: Wavelet sparse inpainting
    # --------------------------------------------------------
    x_hat, row, wavelet_log = run_one_method(
        method_name="wavelet",
        func=lambda yy, mm: wavelet_sparse_inpainting(
            yy,
            mm,
            lam_wavelet=0.01,
            steps=120,
            step_size=1.0,
            wavelet="db2",
            level=2,
            init="telea",
            enforce_observed=True,
            verbose=False
        ),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )

    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("Wavelet")

    # --------------------------------------------------------
    # 8. Baseline 5: K-means patch prior
    # --------------------------------------------------------
    x_hat, row, kmeans_log = run_one_method(
        method_name="kmeans_patch",
        func=lambda yy, mm: kmeans_patch_prior_inpainting(
            yy,
            mm,
            patch_size=8,
            stride=4,
            n_clusters=64,
            outer_iters=15,
            alpha=0.8,
            visible_threshold=0.6,
            init="telea",
            random_state=0,
            verbose=False
        ),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )

    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("K-means Patch")

    if kmeans_log is not None:
        save_loss_curve(
            kmeans_log,
            save_dir / "kmeans_patch_curve.png",
            title="K-means Patch Error"
        )

    # --------------------------------------------------------
    # 9. Train temporary AE
    # --------------------------------------------------------
    print("Training temporary debug AE...")
    print("正式实验时，这里要换成 A 的 train_autoencoder().")
    print()

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
    # 10. Ours: AE + TV optimization
    # --------------------------------------------------------
    x_hat, row, ours_log = run_one_method(
        method_name="ours_ae_tv",
        func=lambda yy, mm: optimize_image_with_ae_tv(
            y=yy,
            mask=mm,
            ae=ae,
            patch_size=8,
            stride=4,
            lam_ae=1e-3,
            lam_tv=1e-3,
            lr=1e-2,
            steps=800,
            init="telea",
            ae_input_format="image",
            fix_observed=True,
            x_gt=x_gt,
            log_interval=20,
            verbose=False
        ),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )

    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("Ours AE+TV")

    if ours_log is not None:
        save_loss_curve(
            ours_log,
            save_dir / "ours_loss_curve.png",
            title="Ours Loss Curve"
        )

        save_psnr_curve(
            ours_log,
            save_dir / "ours_psnr_curve.png"
        )

    # --------------------------------------------------------
    # 11. Save comparison figure
    # --------------------------------------------------------
    images = [
        x_gt,
        mask,
        y,
    ] + restored_images

    titles = [
        "Original",
        "Mask",
        "Corrupted",
    ] + restored_titles

    save_comparison_figure(
        images,
        titles,
        save_dir / "comparison.png"
    )

    # --------------------------------------------------------
    # 12. Save metrics CSV
    # --------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = save_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)

    print("All results:")
    print(df)
    print()
    print(f"Saved images to: {save_dir}")
    print(f"Saved metrics to: {csv_path}")
    print(f"Saved comparison figure to: {save_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
