import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.mask_generator import random_mask, irregular_mask


def save_debug_image(x, mask, y, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(x, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(y, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Corrupted")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    save_dir = PROJECT_ROOT / "results" / "debug_mask"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use Cameraman image from skimage
    x = data.camera().astype(np.float32) / 255.0
    x = resize(x, (128, 128), anti_aliasing=True).astype(np.float32)

    masks = {
        "random_10": random_mask(x.shape, missing_ratio=0.1, seed=0),
        "random_30": random_mask(x.shape, missing_ratio=0.3, seed=0),
        "irregular": irregular_mask(
            x.shape,
            num_strokes=5,
            max_len=25,
            max_width=6,
            seed=0
        ),
    }

    for name, mask in masks.items():
        y = x * mask

        print(f"{name}:")
        print("  mask shape:", mask.shape)
        print("  observed ratio:", mask.mean())
        print("  missing ratio:", 1 - mask.mean())
        print("  y min/max:", y.min(), y.max())

        save_debug_image(
            x,
            mask,
            y,
            save_dir / f"{name}_check.png"
        )


if __name__ == "__main__":
    main()
