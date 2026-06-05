"""
data_loader.py
--------------
Loads standard grayscale test images for image inpainting experiments.

Supported sources:
  1. scikit-image built-in images (cameraman, etc.)
  2. Local image files (place in ./data/images/)
  3. BSD68 subset (place downloaded images in ./data/bsd68/)

Usage:
    from data_loader import load_all_images, load_single_image
    images = load_all_images()          # returns dict {name: np.array [H,W] float32 in [0,1]}
    img = load_single_image("cameraman")
"""

import os
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
BSD68_DIR  = "./data/bsd68"       # put BSD68 .jpg/.png files here
LOCAL_DIR  = "./data/images"      # optional extra local images
BSD68_LIMIT = 10                  # use at most this many BSD68 images

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
# BSD68 subset
# ─────────────────────────────────────────────────────────────────────────────

def _load_bsd68(limit: int = BSD68_LIMIT) -> dict:
    """Load up to `limit` images from BSD68_DIR."""
    images = {}
    if not os.path.isdir(BSD68_DIR):
        print(f"  [BSD68]    directory '{BSD68_DIR}' not found.")
        print(f"             Download BSD68 from: https://github.com/clausmichele/CBSD68-dataset")
        print(f"             or: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/")
        return images

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(BSD68_DIR)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])[:limit]

    for fname in files:
        path = os.path.join(BSD68_DIR, fname)
        name = "bsd_" + os.path.splitext(fname)[0]
        try:
            arr = _load_pil(path)
            images[name] = _resize(arr, IMAGE_SIZE)
            print(f"  [BSD68]    {name:<20} loaded, shape={images[name].shape}")
        except Exception as e:
            print(f"  [BSD68]    {name:<20} FAILED: {e}")

    return images

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_all_images(image_size: int = IMAGE_SIZE,
                    bsd68_limit: int = BSD68_LIMIT) -> dict:
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

    # 3. BSD68
    images.update(_load_bsd68(limit=bsd68_limit))

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
