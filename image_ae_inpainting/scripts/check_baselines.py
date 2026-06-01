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
from src.metrics import compute_all_metrics
from src.baselines import (
    opencv_telea,
    opencv_ns,
    tv_inpainting,
    wavelet_sparse_inpainting,
    kmeans_patch_prior_inpainting,
)


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


def run_one_method(method_name, func, y, mask, x_gt, save_dir):
    print(f"Running {method_name}...")

    start_time = time.time()

    result = func(y, mask)

    # Some functions return (x_hat, log)
    if isinstance(result, tuple):
        x_hat, log = result
    else:
        x_hat = result
        log = None

    runtime = time.time() - start_time

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

    return x_hat, row, log


def main():
    save_dir = Path("results/debug_baselines")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Cameraman image
    x_gt = data.camera().astype(np.float32) / 255.0
    x_gt = resize(
        x_gt,
        (128, 128),
        anti_aliasing=True
    ).astype(np.float32)

    # Generate random 30% missing mask
    mask = random_mask(x_gt.shape, missing_ratio=0.3, seed=0)
    y = apply_mask(x_gt, mask)

    save_image(x_gt, save_dir / "original.png")
    save_image(mask, save_dir / "mask.png")
    save_image(y, save_dir / "corrupted.png")

    print("Corrupted image metrics:")
    corrupted_metrics = compute_all_metrics(y, x_gt)
    for k, v in corrupted_metrics.items():
        print(f"  {k}: {v:.6f}")
    print()

    rows = []

    restored_images = []
    restored_titles = []

    # -----------------------------
    # Baseline 1: OpenCV Telea
    # -----------------------------
    x_hat, row, _ = run_one_method(
        method_name="opencv_telea",
        func=lambda yy, mm: opencv_telea(yy, mm, radius=3),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )
    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("OpenCV Telea")

    # -----------------------------
    # Baseline 2: OpenCV Navier-Stokes
    # -----------------------------
    x_hat, row, _ = run_one_method(
        method_name="opencv_ns",
        func=lambda yy, mm: opencv_ns(yy, mm, radius=3),
        y=y,
        mask=mask,
        x_gt=x_gt,
        save_dir=save_dir
    )
    rows.append(row)
    restored_images.append(x_hat)
    restored_titles.append("OpenCV NS")

    # -----------------------------
    # Baseline 3: TV inpainting
    # -----------------------------
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
        np.save(save_dir / "tv_log.npy", tv_log)

    # -----------------------------
    # Baseline 4: Wavelet sparse inpainting
    # -----------------------------
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
    restored_titles.append("Wavelet Sparse")

    if wavelet_log is not None:
        np.save(save_dir / "wavelet_log.npy", wavelet_log)

    # -----------------------------
    # Baseline 5: K-means patch prior
    # -----------------------------
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
        np.save(save_dir / "kmeans_patch_log.npy", kmeans_log)

    # Save comparison figure
    images = [x_gt, mask, y] + restored_images
    titles = ["Original", "Mask", "Corrupted"] + restored_titles

    save_comparison_figure(
        images,
        titles,
        save_dir / "baseline_comparison.png"
    )

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = save_dir / "baseline_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("All baseline results:")
    print(df)
    print()
    print(f"Images saved to: {save_dir}")
    print(f"Metrics saved to: {csv_path}")
    print(f"Comparison figure saved to: {save_dir / 'baseline_comparison.png'}")


if __name__ == "__main__":
    main()
