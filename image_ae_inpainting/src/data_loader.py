"""
data_loader.py
--------------
Loads standard grayscale test images for image inpainting experiments.

Supported sources:
  1. scikit-image built-in images (cameraman, etc.)
  2. Local image files (place in ./data/images/)

Usage:
    from data_loader import load_all_images, load_single_image
    images = load_all_images()          # returns dict {name: np.array [H,W] float32 in [0,1]}
    img = load_single_image("cameraman")
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

# ── Try importing skimage (optional but preferred for built-ins) ──────────────
try:
    from skimage import data as skdata
    from skimage.color import rgb2gray
    from skimage.transform import resize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 128          # resize all images to IMAGE_SIZE × IMAGE_SIZE
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DIR  = str(DATA_DIR / "images")      # optional extra local images

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray_float32(arr: np.ndarray) -> np.ndarray:
    """Convert any array to float32 grayscale in [0, 1]."""
    if arr.ndim == 3:
        # RGB / RGBA → grayscale via luminance weights
        if SKIMAGE_AVAILABLE:
            arr = rgb2gray(arr)          # already [0,1] float
        else:
            arr = arr[..., :3]
            arr = (0.299 * arr[..., 0] +
                   0.587 * arr[..., 1] +
                   0.114 * arr[..., 2])
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:              # uint8 range
        arr = arr / 255.0
    return arr


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size) using PIL (no skimage required)."""
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    # PIL works on uint8
    pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def _load_pil(path: str) -> np.ndarray:
    """Load any image file via PIL and return float32 grayscale [0,1]."""
    img = Image.open(path).convert("L")          # force grayscale
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

# ─────────────────────────────────────────────────────────────────────────────
# Built-in images from scikit-image
# ─────────────────────────────────────────────────────────────────────────────

_SKIMAGE_LOADERS = {
    "cameraman": lambda: skdata.camera(),          # native grayscale uint8
    "lena":      lambda: skdata.lena() if hasattr(skdata, "lena") else None,
    "astronaut": lambda: rgb2gray(skdata.astronaut()),   # fallback for lena
    "barbara":   lambda: None,                     # not in skimage; loaded from file
    "peppers":   lambda: None,                     # not in skimage; loaded from file
    "house":     lambda: None,                     # not in skimage; loaded from file
}


def _load_skimage_builtins() -> dict:
    images = {}
    if not SKIMAGE_AVAILABLE:
        return images

    # cameraman
    try:
        arr = skdata.camera()                      # uint8 grayscale
        images["cameraman"] = _resize(_to_gray_float32(arr), IMAGE_SIZE)
        print(f"  [built-in] cameraman  loaded, shape={images['cameraman'].shape}")
    except Exception as e:
        print(f"  [built-in] cameraman  FAILED: {e}")

    # lena / astronaut
    lena_loaded = False
    if hasattr(skdata, "lena"):
        try:
            arr = skdata.lena()
            images["lena"] = _resize(_to_gray_float32(arr), IMAGE_SIZE)
            print(f"  [built-in] lena       loaded, shape={images['lena'].shape}")
            lena_loaded = True
        except Exception:
            pass
    if not lena_loaded:
        try:
            arr = skdata.astronaut()
            images["lena"] = _resize(_to_gray_float32(arr), IMAGE_SIZE)
            print(f"  [built-in] lena (astronaut fallback) loaded")
        except Exception as e:
            print(f"  [built-in] lena       FAILED: {e}")

    return images

# ─────────────────────────────────────────────────────────────────────────────
# Classic images that must be supplied as files
# (Barbara, Peppers, House are not in scikit-image)
# Place them as:  ./data/images/barbara.png  etc.
# Download links are printed if missing.
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIC_NAMES = ["barbara", "peppers", "house", "cameraman", "lena"]

_DOWNLOAD_HINTS = {
    "barbara":   "https://sipi.usc.edu/database/misc/4.2.01.tiff  (USC SIPI Misc, #4.2.01)",
    "peppers":   "https://sipi.usc.edu/database/misc/4.2.07.tiff  (USC SIPI Misc, #4.2.07)",
    "house":     "https://sipi.usc.edu/database/misc/4.2.03.tiff  (USC SIPI Misc, #4.2.03)",
    "cameraman": "Included in scikit-image (skimage.data.camera())",
    "lena":      "Included in scikit-image (skimage.data.lena() or astronaut fallback)",
}


def _load_local_classics(existing: dict) -> dict:
    """Load classic images from ./data/images/ if not already in dict."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    images = dict(existing)

    for name in _CLASSIC_NAMES:
        if name in images:
            continue
        found = False
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            path = os.path.join(LOCAL_DIR, name + ext)
            if os.path.isfile(path):
                try:
                    arr = _load_pil(path)
                    images[name] = _resize(arr, IMAGE_SIZE)
                    print(f"  [local]    {name:<12} loaded from {path}, shape={images[name].shape}")
                    found = True
                    break
                except Exception as e:
                    print(f"  [local]    {name:<12} FAILED to load {path}: {e}")
        if not found and name not in images:
            print(f"  [missing]  {name:<12} not found. Download: {_DOWNLOAD_HINTS.get(name, 'unknown')}")
    return images

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_all_images(image_size: int = IMAGE_SIZE) -> dict:
    """
    Load all available test images.

    Returns
    -------
    dict  {name: np.ndarray}
        Each array is float32, shape (H, W), values in [0, 1].
    """
    global IMAGE_SIZE
    IMAGE_SIZE = image_size

    print("=" * 50)
    print("Loading images...")
    print("=" * 50)

    images = {}

    # 1. scikit-image built-ins
    images.update(_load_skimage_builtins())

    # 2. classic images from ./data/images/
    images = _load_local_classics(images)

    print("=" * 50)
    print(f"Total images loaded: {len(images)}")
    if images:
        print(f"Names: {list(images.keys())}")
    print("=" * 50)
    return images


def load_single_image(name_or_path: str,
                      image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Load a single image by name (classic) or file path.

    Parameters
    ----------
    name_or_path : str
        One of the classic names ("cameraman", "lena", "barbara",
        "peppers", "house") or a direct file path.
    image_size : int
        Target size (square).

    Returns
    -------
    np.ndarray  float32 (H, W) in [0, 1]
    """
    # Direct file path
    if os.path.isfile(name_or_path):
        arr = _load_pil(name_or_path)
        return _resize(arr, image_size)

    # Try built-in or local
    all_imgs = load_all_images(image_size=image_size)
    if name_or_path in all_imgs:
        return all_imgs[name_or_path]

    raise FileNotFoundError(
        f"Image '{name_or_path}' not found. "
        f"Available: {list(all_imgs.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    imgs = load_all_images()
    for name, arr in imgs.items():
        print(f"  {name:<20}  shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  dtype={arr.dtype}")
