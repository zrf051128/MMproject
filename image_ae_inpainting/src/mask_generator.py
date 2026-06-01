import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def random_mask(shape, missing_ratio, seed=0):
    """
    Generate a random missing mask.

    Parameters
    ----------
    shape : tuple
        Image shape, usually (H, W). If input is (H, W, C), only H and W are used.
    missing_ratio : float
        Ratio of missing pixels, e.g., 0.3, 0.5, 0.7.
    seed : int
        Random seed.

    Returns
    -------
    mask : np.ndarray
        Binary mask with shape (H, W).
        mask = 1 means observed.
        mask = 0 means missing.
    """
    if not 0 <= missing_ratio <= 1:
        raise ValueError("missing_ratio must be between 0 and 1.")

    H, W = shape[:2]
    rng = np.random.default_rng(seed)

    # random value > missing_ratio means observed
    mask = rng.random((H, W)) > missing_ratio

    return mask.astype(np.float32)


def block_mask(shape, block_size=32, seed=0):
    """
    Generate a square block missing mask.

    Parameters
    ----------
    shape : tuple
        Image shape, usually (H, W). If input is (H, W, C), only H and W are used.
    block_size : int
        Size of the missing square block.
    seed : int
        Random seed.

    Returns
    -------
    mask : np.ndarray
        Binary mask with shape (H, W).
        mask = 1 means observed.
        mask = 0 means missing.
    """
    H, W = shape[:2]

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if block_size > min(H, W):
        raise ValueError("block_size cannot be larger than image height or width.")

    rng = np.random.default_rng(seed)
    mask = np.ones((H, W), dtype=np.float32)

    top = rng.integers(0, H - block_size + 1)
    left = rng.integers(0, W - block_size + 1)

    mask[top:top + block_size, left:left + block_size] = 0.0

    return mask


def irregular_mask(shape, num_strokes=10, max_len=40, max_width=12, seed=0):
    """
    Generate a free-form irregular missing mask using random strokes.

    Parameters
    ----------
    shape : tuple
        Image shape, usually (H, W). If input is (H, W, C), only H and W are used.
    num_strokes : int
        Number of random strokes.
    max_len : int
        Maximum length of each stroke.
    max_width : int
        Maximum width of each stroke.
    seed : int
        Random seed.

    Returns
    -------
    mask : np.ndarray
        Binary mask with shape (H, W).
        mask = 1 means observed.
        mask = 0 means missing.
    """
    H, W = shape[:2]

    if num_strokes <= 0:
        raise ValueError("num_strokes must be positive.")
    if max_len <= 0:
        raise ValueError("max_len must be positive.")
    if max_width <= 0:
        raise ValueError("max_width must be positive.")

    rng = np.random.default_rng(seed)
    mask = np.ones((H, W), dtype=np.float32)

    for _ in range(num_strokes):
        x_start = rng.integers(0, W)
        y_start = rng.integers(0, H)

        length = rng.integers(max(2, max_len // 4), max_len + 1)
        width = rng.integers(3, max_width + 1)
        angle = rng.uniform(0, 2 * np.pi)

        x_end = int(x_start + length * np.cos(angle))
        y_end = int(y_start + length * np.sin(angle))

        x_end = np.clip(x_end, 0, W - 1)
        y_end = np.clip(y_end, 0, H - 1)

        num_points = max(abs(x_end - x_start), abs(y_end - y_start)) + 1

        xs = np.linspace(x_start, x_end, num_points).astype(np.int32)
        ys = np.linspace(y_start, y_end, num_points).astype(np.int32)

        for x, y in zip(xs, ys):
            x_min = max(0, x - width // 2)
            x_max = min(W, x + width // 2 + 1)
            y_min = max(0, y - width // 2)
            y_max = min(H, y + width // 2 + 1)

            mask[y_min:y_max, x_min:x_max] = 0.0

    return mask


def apply_mask(image, mask):
    """
    Apply mask to image.

    Parameters
    ----------
    image : np.ndarray
        Original image, shape (H, W), range [0, 1].
    mask : np.ndarray
        Binary mask, shape (H, W).
        mask = 1 means observed.
        mask = 0 means missing.

    Returns
    -------
    corrupted : np.ndarray
        Corrupted image.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError("image and mask must have the same height and width.")

    return image * mask


def save_mask_check_figure(image, mask, corrupted, save_path, title="Mask Check"):
    """
    Save Original | Mask | Corrupted figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Mask\nwhite=observed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(corrupted, cmap="gray", vmin=0, vmax=1)
    plt.title("Corrupted")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def load_demo_image(size=(128, 128)):
    """
    Load a demo grayscale image.

    It tries to use skimage.data.camera().
    If skimage is not installed, it creates a simple synthetic image.
    """
    try:
        from skimage import data
        from skimage.transform import resize

        image = data.camera().astype(np.float32) / 255.0
        image = resize(image, size, anti_aliasing=True).astype(np.float32)

    except Exception:
        H, W = size
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)

        image = 0.5 * xx + 0.5 * yy

        cy, cx = H // 2, W // 2
        radius = min(H, W) // 5
        circle = (xx - cx / W) ** 2 + (yy - cy / H) ** 2 < (radius / W) ** 2
        image[circle] = 1.0

        image = image.astype(np.float32)

    return image


def demo():
    """
    Run a simple mask generation test.

    This function will generate:
    - random_30_check.png
    - block_check.png
    - irregular_check.png

    under results/debug_mask/
    """
    project_root = Path(__file__).resolve().parents[1]
    save_dir = project_root / "results" / "debug_mask"
    save_dir.mkdir(parents=True, exist_ok=True)

    image = load_demo_image(size=(128, 128))

    mask_dict = {
        "random_30": random_mask(image.shape, missing_ratio=0.3, seed=0),
        "block": block_mask(image.shape, block_size=32, seed=0),
        "irregular": irregular_mask(
            image.shape,
            num_strokes=10,
            max_len=40,
            max_width=12,
            seed=0
        ),
    }

    for name, mask in mask_dict.items():
        corrupted = apply_mask(image, mask)

        observed_ratio = mask.mean()
        missing_ratio = 1.0 - observed_ratio

        print(f"{name}")
        print(f"  mask shape: {mask.shape}")
        print(f"  observed ratio: {observed_ratio:.4f}")
        print(f"  missing ratio: {missing_ratio:.4f}")
        print(f"  corrupted min: {corrupted.min():.4f}")
        print(f"  corrupted max: {corrupted.max():.4f}")
        print()

        save_path = save_dir / f"{name}_check.png"

        save_mask_check_figure(
            image=image,
            mask=mask,
            corrupted=corrupted,
            save_path=save_path,
            title=name
        )

    print(f"Mask check figures saved to: {save_dir}")


if __name__ == "__main__":
    demo()
