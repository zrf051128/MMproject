"""
train_ae.py
-----------
Train the patch autoencoder on visible patches of the corrupted image.

Key improvements over v1:
  - visible_ratio_thresh fixed at 0.8 (not adaptive)
  - patch augmentation: horizontal/vertical flip, 90° rotations
  - combined mode by default: current-image + BSD external patches
  - deeper network (in autoencoder.py)
  - 300 epochs, batch=512

Usage:
    python train_ae.py --image cameraman --mask_ratio 0.5
    python train_ae.py --image cameraman --mask_ratio 0.5 --mode current
    python train_ae.py --image cameraman --mask_ratio 0.5 --mode combined
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import PatchAutoencoder, build_autoencoder
from patch_utils  import get_visible_patches, extract_patches
from data_loader  import load_all_images, load_single_image, IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TrainConfig:
    patch_size           : int   = 8
    latent_dim           : int   = 16
    stride               : int   = 4
    visible_ratio_thresh : float = 0.8   # fixed, not adaptive

    epochs               : int   = 300
    batch_size           : int   = 512
    lr                   : float = 1e-3
    weight_decay         : float = 1e-5

    mode                 : str   = "combined"   # current | external | combined
    external_dir         : str   = "./data/bsd68"
    external_limit       : int   = 20

    save_dir             : str   = "./results/ae_weights"
    verbose              : bool  = True
    save_every           : int   = 50

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    def __repr__(self):
        lines = ["TrainConfig:"]
        for k in ["patch_size","latent_dim","stride","visible_ratio_thresh",
                  "epochs","batch_size","lr","mode"]:
            lines.append(f"  {k:<25}: {getattr(self, k)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Patch augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_patches(patches: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Augment patches with flips and 90° rotations.
    Increases dataset size by 8x.

    patches: (N, p*p)  →  returns (N*8, p*p)
    """
    p = patch_size
    imgs = patches.reshape(-1, p, p)   # (N, p, p)

    variants = [imgs]
    variants.append(np.flip(imgs, axis=2))           # horizontal flip
    variants.append(np.flip(imgs, axis=1))           # vertical flip
    variants.append(np.rot90(imgs, k=1, axes=(1,2))) # 90°
    variants.append(np.rot90(imgs, k=2, axes=(1,2))) # 180°
    variants.append(np.rot90(imgs, k=3, axes=(1,2))) # 270°
    variants.append(np.flip(np.rot90(imgs, k=1, axes=(1,2)), axis=2))  # 90°+hflip
    variants.append(np.flip(np.rot90(imgs, k=3, axes=(1,2)), axis=2))  # 270°+hflip

    augmented = np.concatenate([v.reshape(-1, p*p) for v in variants], axis=0)
    return augmented.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _make_random_mask(image_shape, missing_ratio, seed=42):
    rng  = np.random.default_rng(seed)
    mask = (rng.random(image_shape) > missing_ratio).astype(np.float32)
    return mask


def _load_external_patches(external_dir, patch_size, stride, limit=20):
    if not os.path.isdir(external_dir):
        print(f"  [external] '{external_dir}' not found. Skipping.")
        return np.empty((0, patch_size * patch_size), dtype=np.float32)

    from data_loader import _load_pil, _resize
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(external_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])[:limit]

    all_patches = []
    for fname in files:
        path = os.path.join(external_dir, fname)
        try:
            img = _resize(_load_pil(path), IMAGE_SIZE)
            p, _ = extract_patches(img, patch_size, stride)
            all_patches.append(p)
        except Exception as e:
            print(f"  [external] Failed {fname}: {e}")

    if not all_patches:
        return np.empty((0, patch_size * patch_size), dtype=np.float32)
    return np.concatenate(all_patches, axis=0)


def prepare_training_data(image, mask, config):
    p, s = config.patch_size, config.stride

    # Current image visible patches (threshold fixed at 0.8)
    if config.mode in ("current", "combined"):
        vis = get_visible_patches(image, mask, p, s, config.visible_ratio_thresh)
        if config.verbose:
            print(f"  Current-image visible patches (thresh={config.visible_ratio_thresh}): {len(vis)}")
        # If too few patches, warn but don't lower threshold
        if len(vis) < 50:
            print(f"  WARNING: Only {len(vis)} visible patches. "
                  f"Consider lower mask_ratio or combined mode.")
    else:
        vis = np.empty((0, p*p), dtype=np.float32)

    # External patches
    if config.mode in ("external", "combined"):
        ext = _load_external_patches(config.external_dir, p, s, config.external_limit)
        if config.verbose:
            print(f"  External patches: {len(ext)}")
    else:
        ext = np.empty((0, p*p), dtype=np.float32)

    patches = np.concatenate([vis, ext], axis=0)

    if len(patches) == 0:
        raise RuntimeError(
            "No training patches found.\n"
            "  - For 'current' mode: mask_ratio may be too high (try --mask_ratio 0.3)\n"
            "  - For 'external'/'combined': check --external_dir path\n"
            "  - Try: python train_ae.py --image cameraman --mask_ratio 0.5 --mode combined"
        )

    # Augment patches (8x)
    patches_aug = augment_patches(patches, p)
    if config.verbose:
        print(f"  After augmentation: {len(patches_aug)} patches")

    rng = np.random.default_rng(42)
    return patches_aug[rng.permutation(len(patches_aug))]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(image, mask, config=None, save_name="image"):
    if config is None:
        config = TrainConfig()

    if config.verbose:
        print("=" * 50)
        print("Training Patch Autoencoder")
        print(config)
        print("=" * 50)

    patches_np = prepare_training_data(image, mask, config)
    if config.verbose:
        print(f"  Total training patches : {len(patches_np)}")

    patches_t = torch.from_numpy(patches_np).float()
    loader    = DataLoader(TensorDataset(patches_t),
                           batch_size=config.batch_size,
                           shuffle=True, drop_last=False)

    ae        = build_autoencoder(patch_size=config.patch_size,
                                  latent_dim=config.latent_dim)
    ae.train()

    optimizer = optim.Adam(ae.parameters(), lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=config.epochs,
                                                     eta_min=1e-5)

    history   = {"train_loss": [], "train_time_s": 0.0}
    t_start   = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_loss, n = 0.0, 0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = ae.loss_mean(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        scheduler.step()
        avg = epoch_loss / max(n, 1)
        history["train_loss"].append(avg)

        if config.verbose and (epoch % config.save_every == 0 or epoch == 1):
            print(f"  Epoch [{epoch:3d}/{config.epochs}]  "
                  f"loss={avg:.6f}  elapsed={time.time()-t_start:.1f}s")

    history["train_time_s"] = time.time() - t_start
    ae.eval()

    if config.verbose:
        print(f"\nTraining complete in {history['train_time_s']:.1f}s")

    os.makedirs(config.save_dir, exist_ok=True)
    fname = f"{save_name}_p{config.patch_size}_d{config.latent_dim}_{config.mode}.pt"
    save_path = os.path.join(config.save_dir, fname)
    torch.save(ae.state_dict(), save_path)
    if config.verbose:
        print(f"Weights saved → {save_path}")

    return ae, history


def load_autoencoder(path, patch_size=8, latent_dim=16):
    ae = build_autoencoder(patch_size=patch_size, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(path, map_location="cpu"))
    ae.eval()
    print(f"Loaded AE from: {path}")
    return ae


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",        type=str,   default="cameraman")
    parser.add_argument("--mask_ratio",   type=float, default=0.5)
    parser.add_argument("--patch_size",   type=int,   default=8)
    parser.add_argument("--latent_dim",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--mode",         type=str,   default="combined",
                        choices=["current", "external", "combined"])
    parser.add_argument("--external_dir", type=str,   default="./data/bsd68")
    parser.add_argument("--save_dir",     type=str,   default="./results/ae_weights")
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading image: {args.image}")
    image = load_single_image(args.image, image_size=128)
    mask  = _make_random_mask(image.shape, missing_ratio=args.mask_ratio,
                              seed=args.seed)

    print(f"Image shape : {image.shape}")
    print(f"Mask ratio  : {args.mask_ratio} ({int((1-mask.mean())*100)}% missing)")

    stride = max(1, args.patch_size // 2)
    config = TrainConfig(
        patch_size   = args.patch_size,
        latent_dim   = args.latent_dim,
        stride       = stride,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        mode         = args.mode,
        external_dir = args.external_dir,
        save_dir     = args.save_dir,
        verbose      = True,
    )

    ae, history = train_autoencoder(image, mask, config=config,
                                    save_name=args.image)

    print("\nLoss curve:")
    for i, loss in enumerate(history["train_loss"]):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  epoch {i+1:3d}: {loss:.6f}")
