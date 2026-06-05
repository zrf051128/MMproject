"""
patch_utils.py
--------------
Patch extraction utilities for the patch autoencoder inpainting method.

Key functions:
  extract_patches(image, patch_size, stride)
      → patches (N, p*p), positions (N, 2)

  get_visible_patches(image, mask, patch_size, stride, visible_ratio_thresh)
      → visible patches array (K, p*p)  [P_vis in the paper]

  reconstruct_from_patches(patches, positions, image_shape, patch_size, stride)
      → reconstructed image (H, W)  [for visualization / optimization gradient]

Usage:
    from patch_utils import extract_patches, get_visible_patches
    patches, positions = extract_patches(image, patch_size=8, stride=4)
    vis_patches = get_visible_patches(image, mask, patch_size=8, stride=4)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core patch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_patches(image: np.ndarray,
                    patch_size: int = 8,
                    stride: int = 4) -> tuple:
    """
    Extract overlapping patches from a grayscale image using a sliding window.

    Parameters
    ----------
    image      : np.ndarray, shape (H, W), float32 in [0, 1]
    patch_size : int, side length p  (paper uses 4, 8, or 12)
    stride     : int, sliding step   (typically p//2 or p//4)

    Returns
    -------
    patches   : np.ndarray, shape (N, p*p), float32
    positions : np.ndarray, shape (N, 2),   int32   (top-left row, col)
    """
    H, W = image.shape
    p = patch_size
    patches   = []
    positions = []

    for r in range(0, H - p + 1, stride):
        for c in range(0, W - p + 1, stride):
            patch = image[r:r+p, c:c+p].flatten()   # shape (p*p,)
            patches.append(patch)
            positions.append([r, c])

    patches   = np.array(patches,   dtype=np.float32)   # (N, p*p)
    positions = np.array(positions, dtype=np.int32)     # (N, 2)
    return patches, positions


def compute_visible_ratio(mask: np.ndarray,
                          patch_size: int,
                          stride: int) -> np.ndarray:
    """
    Compute the fraction of visible (mask==1) pixels in each patch.

    Parameters
    ----------
    mask       : np.ndarray, shape (H, W), binary {0, 1}
                 1 = observed pixel, 0 = missing pixel
    patch_size : int
    stride     : int

    Returns
    -------
    ratios : np.ndarray, shape (N,), float32 in [0, 1]
    """
    H, W = mask.shape
    p = patch_size
    ratios = []

    for r in range(0, H - p + 1, stride):
        for c in range(0, W - p + 1, stride):
            patch_mask = mask[r:r+p, c:c+p]
            ratio = patch_mask.mean()
            ratios.append(ratio)

    return np.array(ratios, dtype=np.float32)


def get_visible_patches(image: np.ndarray,
                        mask: np.ndarray,
                        patch_size: int = 8,
                        stride: int = 4,
                        visible_ratio_thresh: float = 0.8) -> np.ndarray:
    """
    Extract patches from the image that are mostly visible (mask==1).
    These form P_vis in the paper: the training set for the tiny autoencoder.

    Parameters
    ----------
    image                : np.ndarray, shape (H, W), float32 in [0, 1]
                           (can be corrupted image y; only visible pixels matter)
    mask                 : np.ndarray, shape (H, W), binary {0, 1}
                           1 = observed, 0 = missing
    patch_size           : int   (paper ablation: 4, 8, 12)
    stride               : int
    visible_ratio_thresh : float (paper uses 0.8, i.e. > 80% visible)

    Returns
    -------
    vis_patches : np.ndarray, shape (K, p*p), float32
                  K = number of patches with visible ratio > threshold
    """
    patches, _ = extract_patches(image, patch_size, stride)
    ratios      = compute_visible_ratio(mask, patch_size, stride)

    keep_idx    = np.where(ratios > visible_ratio_thresh)[0]
    vis_patches = patches[keep_idx]
    return vis_patches


# ─────────────────────────────────────────────────────────────────────────────
# Patch reconstruction (averaging overlapping patches back to image)
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_from_patches(patches: np.ndarray,
                             positions: np.ndarray,
                             image_shape: tuple,
                             patch_size: int = 8) -> np.ndarray:
    """
    Reconstruct an image from (possibly modified) patches by averaging
    overlapping regions.

    Parameters
    ----------
    patches     : np.ndarray, shape (N, p*p), float32
    positions   : np.ndarray, shape (N, 2),   int32   (top-left row, col)
    image_shape : tuple (H, W)
    patch_size  : int

    Returns
    -------
    image_rec : np.ndarray, shape (H, W), float32  (averaged reconstruction)
    """
    H, W = image_shape
    p    = patch_size
    acc  = np.zeros((H, W), dtype=np.float32)
    cnt  = np.zeros((H, W), dtype=np.float32)

    for patch, (r, c) in zip(patches, positions):
        acc[r:r+p, c:c+p] += patch.reshape(p, p)
        cnt[r:r+p, c:c+p] += 1.0

    cnt = np.maximum(cnt, 1.0)   # avoid divide-by-zero at borders
    return acc / cnt


# ─────────────────────────────────────────────────────────────────────────────
# AE regularization term computation (numpy version, for inspection)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ae_prior_numpy(image: np.ndarray,
                           ae_forward_fn,
                           patch_size: int = 8,
                           stride: int = 4) -> float:
    """
    Compute R_AE(x) = Σ_i ‖P_i x − A_θ(P_i x)‖²  (numpy, no gradients).
    Useful for monitoring during optimization.

    Parameters
    ----------
    image        : np.ndarray, shape (H, W), float32
    ae_forward_fn: callable, takes (N, p*p) np.ndarray → (N, p*p) np.ndarray
                   (wrap your trained AE: lambda p: ae.predict_numpy(p))
    patch_size   : int
    stride       : int

    Returns
    -------
    float  (scalar value of R_AE)
    """
    patches, _ = extract_patches(image, patch_size, stride)
    recon       = ae_forward_fn(patches)
    residuals   = patches - recon                 # (N, p*p)
    return float(np.sum(residuals ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Utility: patch statistics
# ─────────────────────────────────────────────────────────────────────────────

def patch_stats(image: np.ndarray,
                mask: np.ndarray,
                patch_size: int = 8,
                stride: int = 4,
                visible_ratio_thresh: float = 0.8) -> dict:
    """
    Print and return a summary of patch extraction statistics.
    Useful for verifying setup before training.
    """
    patches, positions = extract_patches(image, patch_size, stride)
    ratios = compute_visible_ratio(mask, patch_size, stride)
    n_vis  = int((ratios > visible_ratio_thresh).sum())

    stats = {
        "total_patches"  : len(patches),
        "visible_patches": n_vis,
        "patch_size"     : patch_size,
        "stride"         : stride,
        "threshold"      : visible_ratio_thresh,
        "patch_dim"      : patch_size * patch_size,
        "image_shape"    : image.shape,
    }
    print("─" * 40)
    print("Patch extraction stats:")
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")
    print("─" * 40)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # synthetic test: random 128×128 image, 50% random mask
    np.random.seed(42)
    img  = np.random.rand(128, 128).astype(np.float32)
    mask = (np.random.rand(128, 128) > 0.5).astype(np.float32)

    for p in [4, 8, 12]:
        print(f"\n=== patch_size={p} ===")
        stats = patch_stats(img, mask, patch_size=p, stride=p//2)
        vis = get_visible_patches(img, mask, patch_size=p, stride=p//2)
        print(f"  visible patches shape: {vis.shape}")

    # reconstruction test
    patches, positions = extract_patches(img, patch_size=8, stride=4)
    rec = reconstruct_from_patches(patches, positions, img.shape, patch_size=8)
    err = np.abs(img - rec).mean()
    print(f"\nReconstruction MAE (should be ~0): {err:.6f}")
