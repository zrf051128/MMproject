import os
import sys
import numpy as np

from skimage import data
from skimage.transform import resize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mask_generator import random_mask, apply_mask
from src.metrics import compute_all_metrics


def main():
    # Load Cameraman image
    x_gt = data.camera().astype(np.float32) / 255.0
    x_gt = resize(x_gt, (128, 128), anti_aliasing=True).astype(np.float32)

    # Generate corrupted image
    mask = random_mask(x_gt.shape, missing_ratio=0.3, seed=0)
    y = apply_mask(x_gt, mask)

    # Case 1: compare original with itself
    metrics_clean = compute_all_metrics(x_gt, x_gt)

    print("Original vs Original:")
    for k, v in metrics_clean.items():
        print(f"  {k}: {v}")

    print()

    # Case 2: compare corrupted image with original
    metrics_corrupted = compute_all_metrics(y, x_gt)

    print("Corrupted vs Original:")
    for k, v in metrics_corrupted.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
