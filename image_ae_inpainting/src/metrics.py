"""
Metrics for image inpainting evaluation.
"""


import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_float01(x):
    """
    Convert image to float32 and clip to [0, 1].
    """
    x = np.asarray(x).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return x


def compute_psnr(x, gt):
    """
    Compute PSNR between restored image x and ground truth gt.

    Higher is better.
    """
    x = _to_float01(x)
    gt = _to_float01(gt)

    return peak_signal_noise_ratio(gt, x, data_range=1.0)


def compute_ssim(x, gt):
    """
    Compute SSIM between restored image x and ground truth gt.

    Higher is better.
    """
    x = _to_float01(x)
    gt = _to_float01(gt)

    return structural_similarity(gt, x, data_range=1.0)


def compute_rmse(x, gt):
    """
    Compute RMSE between restored image x and ground truth gt.

    Lower is better.
    """
    x = _to_float01(x)
    gt = _to_float01(gt)

    mse = np.mean((x - gt) ** 2)
    rmse = np.sqrt(mse)

    return float(rmse)


def compute_mae(x, gt):
    """
    Compute MAE between restored image x and ground truth gt.

    Lower is better.
    """
    x = _to_float01(x)
    gt = _to_float01(gt)

    mae = np.mean(np.abs(x - gt))

    return float(mae)


def compute_all_metrics(x, gt):
    """
    Compute all metrics at once.
    """
    return {
        "psnr": compute_psnr(x, gt),
        "ssim": compute_ssim(x, gt),
        "rmse": compute_rmse(x, gt),
        "mae": compute_mae(x, gt),
    }