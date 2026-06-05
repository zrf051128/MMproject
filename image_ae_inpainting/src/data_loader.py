"""
Load grayscale test images for the image inpainting experiments.

Supported sources:
  1. scikit-image built-in images, such as cameraman.
  2. Local classic images in data/images.
  3. Direct file paths passed to load_single_image.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage import data as skdata
    from skimage.color import rgb2gray

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = 128
LOCAL_DIR = PROJECT_ROOT / "data" / "images"

CLASSIC_NAMES = ["barbara", "peppers", "house", "cameraman", "lena"]

DOWNLOAD_HINTS = {
    "barbara": "USC SIPI misc image 4.2.01",
    "peppers": "USC SIPI misc image 4.2.07",
    "house": "USC SIPI misc image 4.2.03",
    "cameraman": "Included in scikit-image as skimage.data.camera().",
    "lena": "Use data/images/lena.png if skimage.data.lena() is unavailable.",
}


def _to_gray_float32(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 grayscale in [0, 1]."""
    if arr.ndim == 3:
        if SKIMAGE_AVAILABLE:
            arr = rgb2gray(arr)
        else:
            arr = arr[..., :3]
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize to a square float32 grayscale image."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.float32)

    pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def _load_pil(path: str | Path) -> np.ndarray:
    """Load an image file as float32 grayscale in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def _load_skimage_builtins() -> dict:
    images = {}
    if not SKIMAGE_AVAILABLE:
        return images

    try:
        arr = skdata.camera()
        images["cameraman"] = _resize(_to_gray_float32(arr), IMAGE_SIZE)
        print(f"  [built-in] cameraman  loaded, shape={images['cameraman'].shape}")
    except Exception as exc:
        print(f"  [built-in] cameraman  FAILED: {exc}")

    lena_loaded = False
    if hasattr(skdata, "lena"):
        try:
            arr = skdata.lena()
            images["lena"] = _resize(_to_gray_float32(arr), IMAGE_SIZE)
            print(f"  [built-in] lena       loaded, shape={images['lena'].shape}")
            lena_loaded = True
        except Exception as exc:
            print(f"  [built-in] lena       FAILED: {exc}")

    return images


def _load_local_classics(existing: dict) -> dict:
    """Load classic image files from data/images when available."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    images = dict(existing)

    for name in CLASSIC_NAMES:
        if name in images:
            continue

        found = False
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            path = LOCAL_DIR / f"{name}{ext}"
            if not path.is_file():
                continue

            try:
                arr = _load_pil(path)
                images[name] = _resize(arr, IMAGE_SIZE)
                print(f"  [local]    {name:<12} loaded from {path}")
                found = True
                break
            except Exception as exc:
                print(f"  [local]    {name:<12} FAILED to load {path}: {exc}")

        if not found and name not in images:
            print(f"  [missing]  {name:<12} not found. Hint: {DOWNLOAD_HINTS.get(name)}")

    return images


def _add_lena_fallback(images: dict) -> dict:
    """Use skimage astronaut as a placeholder only when no real lena is available."""
    if "lena" in images or not SKIMAGE_AVAILABLE:
        return images

    try:
        arr = skdata.astronaut()
        images["lena"] = _resize(_to_gray_float32(arr), IMAGE_SIZE)
        print("  [fallback] lena       loaded from skimage astronaut")
    except Exception as exc:
        print(f"  [fallback] lena       FAILED: {exc}")

    return images


def load_all_images(image_size: int = IMAGE_SIZE) -> dict:
    """
    Load all available classic test images.

    Returns a dict {name: np.ndarray}, each image float32, shape (H, W), in [0, 1].
    """
    global IMAGE_SIZE
    IMAGE_SIZE = image_size

    print("=" * 50)
    print("Loading images...")
    print("=" * 50)

    images = _load_skimage_builtins()
    images = _load_local_classics(images)
    images = _add_lena_fallback(images)

    print("=" * 50)
    print(f"Total images loaded: {len(images)}")
    if images:
        print(f"Names: {list(images.keys())}")
    print("=" * 50)
    return images


def load_single_image(name_or_path: str, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Load a single image by classic name or direct file path.
    """
    if os.path.isfile(name_or_path):
        arr = _load_pil(name_or_path)
        return _resize(arr, image_size)

    all_images = load_all_images(image_size=image_size)
    if name_or_path in all_images:
        return all_images[name_or_path]

    raise FileNotFoundError(
        f"Image '{name_or_path}' not found. Available: {list(all_images.keys())}"
    )


if __name__ == "__main__":
    imgs = load_all_images()
    for name, arr in imgs.items():
        print(
            f"  {name:<12} shape={arr.shape} min={arr.min():.3f} "
            f"max={arr.max():.3f} dtype={arr.dtype}"
        )
