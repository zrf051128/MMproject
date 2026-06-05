"""
train_ae.py
-----------
Train the tiny patch autoencoder A_θ on visible patches extracted from
the corrupted image itself (image-specific, self-supervised prior).

Training mode:
  current-image: patches from the corrupted image's visible region.

After training, saves model weights to:
    ./results/ae_weights/<image_name>_p<patch_size>_d<latent_dim>.pt

Usage (command line):
    python train_ae.py --image cameraman --patch_size 8 --latent_dim 16 --mask_ratio 0.5

Usage (API):
    from train_ae import train_autoencoder, TrainConfig
    ae, history = train_autoencoder(image, mask, config=TrainConfig())
"""

import os
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from .autoencoder import PatchAutoencoder, build_autoencoder
    from .patch_utils import get_visible_patches, extract_patches
    from .data_loader import load_single_image
except ImportError:
    from autoencoder import PatchAutoencoder, build_autoencoder
    from patch_utils import get_visible_patches, extract_patches
    from data_loader import load_single_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

class TrainConfig:
    """All hyperparameters for AE training."""

    # AE architecture
    patch_size           : int   = 8
    latent_dim           : int   = 16
    stride               : int   = 4          # default = patch_size // 2

    # Visible patch threshold
    visible_ratio_thresh : float = 0.8

    # Training
    epochs               : int   = 50
    batch_size           : int   = 128
    lr                   : float = 1e-3
    weight_decay         : float = 1e-5

    # Data source
    mode                 : str   = "current"

    # Output
    save_dir             : str   = str(RESULTS_DIR / "ae_weights")
    verbose              : bool  = True
    save_every           : int   = 10          # print loss every N epochs

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    def __repr__(self):
        lines = ["TrainConfig:"]
        for k in ["patch_size","latent_dim","stride","visible_ratio_thresh",
                  "epochs","batch_size","lr","weight_decay","mode"]:
            lines.append(f"  {k:<25}: {getattr(self, k)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_random_mask(image_shape: tuple, missing_ratio: float,
                      seed: int = 42) -> np.ndarray:
    """Create a random binary mask: 1=observed, 0=missing."""
    rng  = np.random.default_rng(seed)
    mask = (rng.random(image_shape) > missing_ratio).astype(np.float32)
    return mask


def prepare_training_data(image: np.ndarray,
                          mask: np.ndarray,
                          config: TrainConfig) -> np.ndarray:
    """
    Build the patch training set according to config.mode.

    Returns
    -------
    patches : np.ndarray (K, p*p), float32
    """
    p = config.patch_size
    s = config.stride

    if config.mode != "current":
        raise ValueError("Only current-image AE training is supported.")

    patches = get_visible_patches(image, mask, p, s, config.visible_ratio_thresh)
    if config.verbose:
        print(f"  Current-image visible patches : {len(patches)}")

    if len(patches) == 0:
        raise RuntimeError(
            "No training patches found. Check image/mask and visible_ratio_thresh."
        )

    # Shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(patches))
    return patches[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(image: np.ndarray,
                      mask: np.ndarray,
                      config: TrainConfig = None,
                      save_name: str = "image") -> tuple:
    """
    Train the patch autoencoder on visible patches of `image`.

    Parameters
    ----------
    image     : np.ndarray (H, W), float32 in [0, 1]
    mask      : np.ndarray (H, W), binary {0, 1}  (1=observed)
    config    : TrainConfig  (defaults used if None)
    save_name : str  used in saved filename

    Returns
    -------
    ae      : PatchAutoencoder  (trained, on CPU)
    history : dict  {"train_loss": list of float per epoch,
                     "train_time_s": float}
    """
    if config is None:
        config = TrainConfig()

    if config.verbose:
        print("=" * 50)
        print("Training Patch Autoencoder")
        print(config)
        print("=" * 50)

    # ── Build training set ────────────────────────────────────────────────────
    patches_np = prepare_training_data(image, mask, config)
    if config.verbose:
        print(f"  Total training patches : {len(patches_np)}")
        print(f"  Patch dim              : {patches_np.shape[1]}")

    patches_t  = torch.from_numpy(patches_np).float()     # (K, p*p)
    dataset    = TensorDataset(patches_t)
    loader     = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            drop_last=False)

    # ── Build model ───────────────────────────────────────────────────────────
    ae = build_autoencoder(patch_size=config.patch_size,
                           latent_dim=config.latent_dim)
    ae.train()

    optimizer = optim.Adam(ae.parameters(),
                           lr=config.lr,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=config.epochs,
                                                     eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_time_s": 0.0}
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for (batch,) in loader:
            optimizer.zero_grad()
            loss = ae.loss_mean(batch)          # mean MSE per element
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if config.verbose and (epoch % config.save_every == 0 or epoch == 1):
            elapsed = time.time() - t_start
            print(f"  Epoch [{epoch:3d}/{config.epochs}]  "
                  f"loss={avg_loss:.6f}  "
                  f"elapsed={elapsed:.1f}s")

    history["train_time_s"] = time.time() - t_start
    ae.eval()

    if config.verbose:
        print(f"\nTraining complete in {history['train_time_s']:.1f}s")
        final_loss = ae.reconstruction_error_numpy(patches_np)
        print(f"Final R_AE on training patches: {final_loss:.4f}")

    # ── Save weights ─────────────────────────────────────────────────────────
    os.makedirs(config.save_dir, exist_ok=True)
    fname = (f"{save_name}_p{config.patch_size}_d{config.latent_dim}"
             f"_{config.mode}.pt")
    save_path = os.path.join(config.save_dir, fname)
    torch.save(ae.state_dict(), save_path)
    if config.verbose:
        print(f"Weights saved → {save_path}")

    return ae, history


# ─────────────────────────────────────────────────────────────────────────────
# Load a saved AE
# ─────────────────────────────────────────────────────────────────────────────

def load_autoencoder(path: str,
                     patch_size: int = 8,
                     latent_dim: int = 16) -> PatchAutoencoder:
    """
    Load a previously saved autoencoder from a .pt file.

    Parameters
    ----------
    path       : str  path to .pt file
    patch_size : int
    latent_dim : int

    Returns
    -------
    ae : PatchAutoencoder  (eval mode, on CPU)
    """
    ae = build_autoencoder(patch_size=patch_size, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(path, map_location="cpu"))
    ae.eval()
    print(f"Loaded AE weights from: {path}")
    return ae


# ─────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train tiny patch autoencoder for image inpainting prior"
    )
    parser.add_argument("--image",       type=str,   default="cameraman",
                        help="Image name (cameraman/lena/barbara/peppers/house) or file path")
    parser.add_argument("--mask_ratio",  type=float, default=0.5,
                        help="Fraction of pixels to remove (0–1)")
    parser.add_argument("--patch_size",  type=int,   default=8,
                        help="Patch side length (4, 8, or 12)")
    parser.add_argument("--latent_dim",  type=int,   default=16,
                        help="AE bottleneck dimension (8, 16, or 32)")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--mode",        type=str,   default="current",
                        choices=["current"],
                        help="Training data source")
    parser.add_argument("--save_dir",    type=str,   default=str(RESULTS_DIR / "ae_weights"))
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load image
    print(f"Loading image: {args.image}")
    image = load_single_image(args.image, image_size=128)

    # Generate mask (random missing)
    mask = _make_random_mask(image.shape, missing_ratio=args.mask_ratio, seed=args.seed)
    corrupted = image * mask    # zero-fill missing pixels

    print(f"Image shape  : {image.shape}")
    print(f"Mask ratio   : {args.mask_ratio} ({int((1-mask.mean())*100)}% missing)")

    # Build config
    stride = max(1, args.patch_size // 2)
    # Adaptive threshold: at least (1 - missing_ratio) * 0.7
    # e.g. missing=0.5 → thresh=0.35, missing=0.3 → thresh=0.49
    vis_thresh = max(0.3, (1.0 - args.mask_ratio) * 0.7)
    print(f"Visible ratio threshold: {vis_thresh:.2f}")
    config = TrainConfig(
        patch_size           = args.patch_size,
        latent_dim           = args.latent_dim,
        stride               = stride,
        visible_ratio_thresh = vis_thresh,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        mode         = args.mode,
        save_dir     = args.save_dir,
        verbose      = True,
    )

    # Train (pass original image, not corrupted: AE needs clean visible patches)
    ae, history = train_autoencoder(image, mask, config=config,
                                    save_name=args.image)

    # Final summary
    print("\nTraining loss curve (every 10 epochs):")
    for i, loss in enumerate(history["train_loss"]):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  epoch {i+1:3d}: {loss:.6f}")
