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
from src.baselines import opencv_telea, tv_inpainting
from src.optimization import optimize_image_with_ae_tv


# ============================================================
# Saving utilities
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

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_psnr_curve(log, save_path, title="PSNR Curve"):
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
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Temporary Debug AE
# 正式实验时，这一部分换成 A 的 train_autoencoder()
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
    patches = patches[:, None, :, :]

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
    Temporary tiny AE for debugging.

    正式实验时，你应该把这个函数替换为 A 的 train_autoencoder().
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
        image=x_init,
        mask=mask,
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
# Evaluation helper
# ============================================================

def evaluate_and_save(method_name, x_hat, x_gt, runtime, save_dir, rows):
    metrics = compute_all_metrics(x_hat, x_gt)

    row = {
        "method": method_name,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "runtime": runtime,
    }

    rows.append(row)

    print(f"Results for {method_name}:")
    print(f"  PSNR: {row['psnr']:.6f}")
    print(f"  SSIM: {row['ssim']:.6f}")
    print(f"  RMSE: {row['rmse']:.6f}")
    print(f"  MAE:  {row['mae']:.6f}")
    print(f"  Time: {row['runtime']:.6f} seconds")
    print()

    save_image(x_hat, save_dir / f"{method_name}.png")

    return row


# ============================================================
# Main ablation experiment
# ============================================================

def main():
    save_dir = Path("results/check_ablation_random30")
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------
    missing_ratio = 0.3
    seed = 0

    patch_size = 8
    stride = 4

    lam_ae_main = 1e-3
    lam_tv_main = 1e-3

    steps_ours = 800
    steps_tv = 500

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
    # 2. Generate mask and corrupted image
    # --------------------------------------------------------
    mask = random_mask(
        x_gt.shape,
        missing_ratio=missing_ratio,
        seed=seed
    )

    y = apply_mask(x_gt, mask)

    save_image(x_gt, save_dir / "original.png")
    save_image(mask, save_dir / "mask.png")
    save_image(y, save_dir / "corrupted.png")

    rows = []
    ablation_images = []
    ablation_titles = []

    # --------------------------------------------------------
    # 3. Corrupted / Data only
    # --------------------------------------------------------
    print("Evaluating corrupted image...")

    evaluate_and_save(
        method_name="data_only_corrupted",
        x_hat=y,
        x_gt=x_gt,
        runtime=0.0,
        save_dir=save_dir,
        rows=rows
    )

    ablation_images.append(y)
    ablation_titles.append("Data only")

    # --------------------------------------------------------
    # 4. OpenCV Telea initialization
    # --------------------------------------------------------
    print("Running OpenCV Telea...")

    start = time.time()
    x_telea = opencv_telea(y, mask, radius=3)
    runtime = time.time() - start

    evaluate_and_save(
        method_name="opencv_telea",
        x_hat=x_telea,
        x_gt=x_gt,
        runtime=runtime,
        save_dir=save_dir,
        rows=rows
    )

    ablation_images.append(x_telea)
    ablation_titles.append("Telea")

    # --------------------------------------------------------
    # 5. TV only
    # D(x) + lambda_TV TV(x)
    # --------------------------------------------------------
    print("Running TV only...")

    start = time.time()

    x_tv, tv_log = tv_inpainting(
        y,
        mask,
        lam_tv=lam_tv_main,
        lr=1e-2,
        steps=steps_tv,
        init="telea",
        verbose=False
    )

    runtime = time.time() - start

    evaluate_and_save(
        method_name="tv_only",
        x_hat=x_tv,
        x_gt=x_gt,
        runtime=runtime,
        save_dir=save_dir,
        rows=rows
    )

    np.save(save_dir / "tv_only_log.npy", tv_log)
    save_loss_curve(tv_log, save_dir / "tv_only_loss_curve.png", title="TV Only Loss")

    ablation_images.append(x_tv)
    ablation_titles.append("TV only")

    # --------------------------------------------------------
    # 6. Train AE
    # 正式实验时，这里换成 A 的 train_autoencoder()
    # --------------------------------------------------------
    print("Training temporary debug AE...")
    print("正式实验时，这里要替换成 A 的 train_autoencoder().")
    print()

    ae = train_debug_tiny_ae(
        y=y,
        mask=mask,
        patch_size=patch_size,
        stride=stride,
        latent_dim=16,
        visible_threshold=0.8,
        epochs=200,
        lr=1e-3,
        batch_size=128,
        verbose=True
    )

    # --------------------------------------------------------
    # 7. AE only
    # D(x) + lambda_AE R_AE(x)
    # --------------------------------------------------------
    print("Running AE only...")

    start = time.time()

    x_ae_only, ae_only_log = optimize_image_with_ae_tv(
        y=y,
        mask=mask,
        ae=ae,
        patch_size=patch_size,
        stride=stride,
        lam_ae=lam_ae_main,
        lam_tv=0.0,
        lr=1e-2,
        steps=steps_ours,
        init="telea",
        ae_input_format="image",
        fix_observed=True,
        x_gt=x_gt,
        log_interval=20,
        verbose=False
    )

    runtime = time.time() - start

    evaluate_and_save(
        method_name="ae_only",
        x_hat=x_ae_only,
        x_gt=x_gt,
        runtime=runtime,
        save_dir=save_dir,
        rows=rows
    )

    np.save(save_dir / "ae_only_log.npy", ae_only_log)
    save_loss_curve(ae_only_log, save_dir / "ae_only_loss_curve.png", title="AE Only Loss")
    save_psnr_curve(ae_only_log, save_dir / "ae_only_psnr_curve.png", title="AE Only PSNR")

    ablation_images.append(x_ae_only)
    ablation_titles.append("AE only")

    # --------------------------------------------------------
    # 8. AE + TV
    # D(x) + lambda_AE R_AE(x) + lambda_TV TV(x)
    # --------------------------------------------------------
    print("Running AE + TV...")

    start = time.time()

    x_ae_tv, ae_tv_log = optimize_image_with_ae_tv(
        y=y,
        mask=mask,
        ae=ae,
        patch_size=patch_size,
        stride=stride,
        lam_ae=lam_ae_main,
        lam_tv=lam_tv_main,
        lr=1e-2,
        steps=steps_ours,
        init="telea",
        ae_input_format="image",
        fix_observed=True,
        x_gt=x_gt,
        log_interval=20,
        verbose=False
    )

    runtime = time.time() - start

    evaluate_and_save(
        method_name="ae_tv",
        x_hat=x_ae_tv,
        x_gt=x_gt,
        runtime=runtime,
        save_dir=save_dir,
        rows=rows
    )

    np.save(save_dir / "ae_tv_log.npy", ae_tv_log)
    save_loss_curve(ae_tv_log, save_dir / "ae_tv_loss_curve.png", title="AE + TV Loss")
    save_psnr_curve(ae_tv_log, save_dir / "ae_tv_psnr_curve.png", title="AE + TV PSNR")

    ablation_images.append(x_ae_tv)
    ablation_titles.append("AE + TV")

    # --------------------------------------------------------
    # 9. Lambda_AE sweep
    # This is parameter sensitivity, not the main ablation table.
    # --------------------------------------------------------
    print("Running lambda_AE sweep...")

    lam_ae_list = [
        0.0,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
    ]

    sweep_rows = []
    sweep_images = []
    sweep_titles = []

    for lam_ae in lam_ae_list:
        method_name = f"lam_ae_{lam_ae:.0e}".replace("+", "")

        print(f"Running {method_name}...")

        start = time.time()

        x_sweep, sweep_log = optimize_image_with_ae_tv(
            y=y,
            mask=mask,
            ae=ae,
            patch_size=patch_size,
            stride=stride,
            lam_ae=lam_ae,
            lam_tv=lam_tv_main,
            lr=1e-2,
            steps=steps_ours,
            init="telea",
            ae_input_format="image",
            fix_observed=True,
            x_gt=x_gt,
            log_interval=20,
            verbose=False
        )

        runtime = time.time() - start

        metrics = compute_all_metrics(x_sweep, x_gt)

        sweep_row = {
            "lam_ae": lam_ae,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "runtime": runtime,
        }

        sweep_rows.append(sweep_row)

        save_image(x_sweep, save_dir / f"sweep_{method_name}.png")
        np.save(save_dir / f"sweep_{method_name}_log.npy", sweep_log)

        sweep_images.append(x_sweep)
        sweep_titles.append(f"lam_AE={lam_ae:.0e}")

        print(f"  PSNR: {metrics['psnr']:.6f}")
        print(f"  SSIM: {metrics['ssim']:.6f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE:  {metrics['mae']:.6f}")
        print(f"  Time: {runtime:.6f} seconds")
        print()

    # --------------------------------------------------------
    # 10. Save comparison figures
    # --------------------------------------------------------
    ablation_comparison_images = [
        x_gt,
        mask,
        y,
    ] + ablation_images

    ablation_comparison_titles = [
        "Original",
        "Mask",
        "Corrupted",
    ] + ablation_titles

    save_comparison_figure(
        ablation_comparison_images,
        ablation_comparison_titles,
        save_dir / "ablation_comparison.png"
    )

    sweep_comparison_images = [
        x_gt,
        y,
    ] + sweep_images

    sweep_comparison_titles = [
        "Original",
        "Corrupted",
    ] + sweep_titles

    save_comparison_figure(
        sweep_comparison_images,
        sweep_comparison_titles,
        save_dir / "lambda_ae_sweep_comparison.png"
    )

    # --------------------------------------------------------
    # 11. Save CSV files
    # --------------------------------------------------------
    ablation_df = pd.DataFrame(rows)
    ablation_csv = save_dir / "ablation_metrics.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_csv = save_dir / "lambda_ae_sweep_metrics.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    print("Ablation results:")
    print(ablation_df)
    print()

    print("Lambda_AE sweep results:")
    print(sweep_df)
    print()

    print(f"Results saved to: {save_dir}")
    print(f"Ablation metrics saved to: {ablation_csv}")
    print(f"Lambda_AE sweep metrics saved to: {sweep_csv}")
    print(f"Ablation figure saved to: {save_dir / 'ablation_comparison.png'}")
    print(f"Sweep figure saved to: {save_dir / 'lambda_ae_sweep_comparison.png'}")


if __name__ == "__main__":
    main()