import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mask_generator import random_mask, block_mask, irregular_mask


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
    os.makedirs("results/debug_mask", exist_ok=True)

    # Use Cameraman image from skimage
    x = data.camera().astype(np.float32) / 255.0
    x = resize(x, (128, 128), anti_aliasing=True).astype(np.float32)

    masks = {
        "random_30": random_mask(x.shape, missing_ratio=0.3, seed=0),
        "block": block_mask(x.shape, block_size=32, seed=0),
        "irregular": irregular_mask(x.shape, seed=0),
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
            f"results/debug_mask/{name}_check.png"
        )


if __name__ == "__main__":
    main()
