import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "image_ae_inpainting"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.baselines import dct_inpainting, tv_inpainting, wavelet_inpainting
from src.data_loader import load_all_images
from src.mask_generator import apply_mask, irregular_mask, random_mask
from src.metrics import compute_all_metrics
from src.optimization import optimize_image_with_ae_tv
from src.train_ae import TrainConfig, load_autoencoder, train_autoencoder


DEFAULT_IMAGES = ["cameraman", "barbara", "house", "peppers"]
DEFAULT_MASKS = ["random10", "random30", "random50", "irregular"]
DEFAULT_METHODS = ["dct", "wavelet", "tv", "ae_only", "tv_ae"]

METHOD_LABELS = {
    "dct": "DCT-only",
    "wavelet": "Wavelet",
    "tv": "TV",
    "ae_only": "AE-only",
    "tv_ae": "TV+AE",
}


def canonical_method_name(method_name):
    raw_name = method_name.lower()
    method_name = raw_name.replace("-", "_").replace("+", "_")
    aliases = {
        "dct_only": "dct",
        "tv_only": "tv",
        "ae": "ae_only",
        "aeonly": "ae_only",
        "ae_tv": "tv_ae",
        "tvae": "tv_ae",
    }
    return aliases.get(method_name, raw_name)


def methods_require_ae(methods):
    return any(canonical_method_name(method) in {"ae_only", "tv_ae"} for method in methods)


def parse_name_list(value, default_values):
    if value is None or str(value).strip() == "":
        return list(default_values)
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if len(items) == 1 and items[0].lower() == "all":
        return list(default_values)
    return items


def ensure_result_dirs():
    dirs = {
        "restored": RESULTS_DIR / "restored_images",
        "logs": RESULTS_DIR / "logs",
        "tables": RESULTS_DIR / "tables",
        "figures": RESULTS_DIR / "figures",
        "weights": RESULTS_DIR / "ae_weights",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_selected_images(image_names, image_size=128):
    all_images = load_all_images(image_size=image_size)
    selected = {}
    missing = []

    for name in image_names:
        if name in all_images:
            selected[name] = all_images[name]
        else:
            missing.append(name)

    if missing:
        available = ", ".join(all_images.keys())
        raise ValueError(f"Images not found: {missing}. Available: {available}")

    return selected


def random_ratio_from_name(mask_type):
    suffix = mask_type.lower().replace("random", "")
    if suffix == "":
        raise ValueError("Random mask name must be like random10, random30, random50.")
    ratio = float(suffix)
    if ratio > 1.0:
        ratio = ratio / 100.0
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Invalid random mask ratio in '{mask_type}'.")
    return ratio


def make_mask(mask_type, shape, seed=0):
    mask_type = mask_type.lower()

    if mask_type.startswith("random"):
        return random_mask(shape, missing_ratio=random_ratio_from_name(mask_type), seed=seed)

    if mask_type == "irregular":
        return irregular_mask(shape, seed=seed)

    raise ValueError(f"Unknown mask type: {mask_type}")


def save_image(image, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_path, image, cmap="gray", vmin=0.0, vmax=1.0)


def save_comparison_figure(images, titles, save_path, max_cols=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    if max_cols is None:
        max_cols = n
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(3.0 * cols, 3.1 * rows))
    for idx, (img, title) in enumerate(zip(images, titles), start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_log(log, save_path):
    if log is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, log)


def append_csv_rows(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def ae_weight_path(image_name, patch_size=8, latent_dim=16, mask_type=None):
    weights_dir = RESULTS_DIR / "ae_weights"
    if mask_type:
        filename = f"{image_name}_{mask_type}_p{patch_size}_d{latent_dim}_current.pt"
    else:
        filename = f"{image_name}_p{patch_size}_d{latent_dim}_combined.pt"
    return weights_dir / filename


def adaptive_visible_threshold(mask, min_threshold=0.3):
    observed_ratio = float(np.mean(mask))
    return max(float(min_threshold), observed_ratio * 0.7)


def get_autoencoder_for_case(
    image_name,
    mask_type,
    clean_image,
    corrupted_image,
    mask,
    ae_source="load",
    patch_size=8,
    latent_dim=16,
    stride=4,
    ae_epochs=50,
    ae_batch_size=128,
    ae_lr=1e-3,
    verbose=False,
):
    """
    ae_source:
      - load: load image-level weight, e.g. cameraman_p8_d16_combined.pt
      - train: train mask-specific AE on the corrupted image and save it
      - auto: use mask-specific weight if present, else image-level, else train
    """
    ae_source = ae_source.lower()
    image_level_path = ae_weight_path(image_name, patch_size, latent_dim, mask_type=None)
    mask_level_path = ae_weight_path(image_name, patch_size, latent_dim, mask_type=mask_type)

    if ae_source == "load":
        if not image_level_path.exists():
            raise FileNotFoundError(
                f"AE weight not found: {image_level_path}. "
                "Use --ae-source train or train the weight first."
            )
        return load_autoencoder(str(image_level_path), patch_size, latent_dim), str(image_level_path)

    if ae_source == "auto":
        if mask_level_path.exists():
            return load_autoencoder(str(mask_level_path), patch_size, latent_dim), str(mask_level_path)
        if image_level_path.exists():
            return load_autoencoder(str(image_level_path), patch_size, latent_dim), str(image_level_path)
        ae_source = "train"

    if ae_source != "train":
        raise ValueError("ae_source must be one of: load, train, auto")

    config = TrainConfig(
        patch_size=patch_size,
        latent_dim=latent_dim,
        stride=stride,
        visible_ratio_thresh=adaptive_visible_threshold(mask),
        epochs=ae_epochs,
        batch_size=ae_batch_size,
        lr=ae_lr,
        mode="current",
        save_dir=str(RESULTS_DIR / "ae_weights"),
        verbose=verbose,
        save_every=max(1, ae_epochs // 5),
    )

    ae, _history = train_autoencoder(
        corrupted_image,
        mask,
        config=config,
        save_name=f"{image_name}_{mask_type}",
    )
    return ae, str(mask_level_path)


def default_method_settings():
    return {
        "lam_dct": 0.01,
        "lam_wavelet": 0.01,
        "lam_tv": 0.01,
        "lam_ae": 0.01,
        "image_lr": 0.03,
        "image_steps": 500,
        "log_interval": 20,
    }


def method_parameter_extra(method_name, settings, patch_size, stride, latent_dim):
    method_name = canonical_method_name(method_name)
    row = {
        "lambda_dct": np.nan,
        "lambda_wavelet": np.nan,
        "lambda_tv": np.nan,
        "lambda_ae": np.nan,
        "patch_size": patch_size,
        "stride": stride,
        "latent_dim": latent_dim,
    }

    if method_name == "dct":
        row["lambda_dct"] = settings["lam_dct"]
    elif method_name == "wavelet":
        row["lambda_wavelet"] = settings["lam_wavelet"]
    elif method_name == "tv":
        row["lambda_tv"] = settings["lam_tv"]
    elif method_name == "ae_only":
        row["lambda_ae"] = settings["lam_ae"]
    elif method_name == "tv_ae":
        row["lambda_tv"] = settings["lam_tv"]
        row["lambda_ae"] = settings["lam_ae"]

    return row


def run_restoration_method(
    method_name,
    y,
    mask,
    x_gt,
    settings,
    ae=None,
    patch_size=8,
    stride=4,
    verbose=False,
):
    method_name = canonical_method_name(method_name)

    start = time.time()
    log = None

    if method_name == "dct":
        x_hat, log = dct_inpainting(
            y,
            mask,
            lam_dct=settings["lam_dct"],
            lr=settings["image_lr"],
            steps=settings["image_steps"],
            x_gt=x_gt,
            log_interval=settings["log_interval"],
            verbose=verbose,
        )

    elif method_name == "wavelet":
        x_hat, log = wavelet_inpainting(
            y,
            mask,
            lam_wavelet=settings["lam_wavelet"],
            lr=settings["image_lr"],
            steps=settings["image_steps"],
            x_gt=x_gt,
            log_interval=settings["log_interval"],
            verbose=verbose,
        )

    elif method_name == "tv":
        x_hat, log = tv_inpainting(
            y,
            mask,
            lam_tv=settings["lam_tv"],
            lr=settings["image_lr"],
            steps=settings["image_steps"],
            x_gt=x_gt,
            log_interval=settings["log_interval"],
            verbose=verbose,
        )

    elif method_name in {"ae_only", "tv_ae"}:
        if ae is None:
            raise ValueError(f"Method '{method_name}' requires a trained autoencoder.")
        x_hat, log = optimize_image_with_ae_tv(
            y=y,
            mask=mask,
            ae=ae,
            patch_size=patch_size,
            stride=stride,
            lam_ae=settings["lam_ae"],
            lam_tv=0.0 if method_name == "ae_only" else settings["lam_tv"],
            lr=settings["image_lr"],
            steps=settings["image_steps"],
            init="mean",
            ae_input_format="flat",
            fix_observed=True,
            x_gt=x_gt,
            log_interval=settings["log_interval"],
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")

    runtime = time.time() - start
    metrics = compute_all_metrics(x_hat, x_gt, mask=mask)
    return x_hat, log, runtime, metrics


def metric_row(image_name, mask_type, method_name, metrics, runtime, extra=None):
    row = {
        "image": image_name,
        "mask_type": mask_type,
        "method": canonical_method_name(method_name),
        "psnr": metrics.get("psnr", np.nan),
        "ssim": metrics.get("ssim", np.nan),
        "rmse": metrics.get("rmse", np.nan),
        "mae": metrics.get("mae", np.nan),
        "psnr_missing": metrics.get("psnr_missing", np.nan),
        "rmse_missing": metrics.get("rmse_missing", np.nan),
        "mae_missing": metrics.get("mae_missing", np.nan),
        "runtime": runtime,
    }
    if extra:
        row.update(extra)
    return row


def plot_loss_curve(log, save_path, title="Loss Curve"):
    if not log:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "total_loss",
        "data_loss",
        "dct_loss",
        "wavelet_loss",
        "tv_loss",
        "ae_loss",
    ]

    plt.figure(figsize=(6, 4))
    for key in keys:
        if key in log:
            plt.plot(log[key], label=key)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_psnr_curve(log, save_path, title="PSNR Curve"):
    if not log or "step" not in log or "psnr" not in log or len(log["step"]) == 0:
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
