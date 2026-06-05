"""
Train an image-specific patch autoencoder on visible patches for inpainting.

The autoencoder is trained only from the current corrupted image's observed
pixels.
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from .autoencoder import PatchAutoencoder, build_autoencoder
    from .data_loader import load_single_image
    from .patch_utils import get_visible_patches
except ImportError:
    from autoencoder import PatchAutoencoder, build_autoencoder
    from data_loader import load_single_image
    from patch_utils import get_visible_patches


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class TrainConfig:
    """Hyperparameters for current-image AE training."""

    patch_size: int = 8
    latent_dim: int = 16
    stride: int = 4
    visible_ratio_thresh: float = 0.8

    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5

    mode: str = "current"
    save_dir: str = str(RESULTS_DIR / "ae_weights")
    verbose: bool = True
    save_every: int = 10

    def __post_init__(self):
        if self.mode != "current":
            raise ValueError("Only mode='current' is supported.")

    def __repr__(self):
        keys = [
            "patch_size",
            "latent_dim",
            "stride",
            "visible_ratio_thresh",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "mode",
        ]
        lines = ["TrainConfig:"]
        for key in keys:
            lines.append(f"  {key:<25}: {getattr(self, key)}")
        return "\n".join(lines)


def _make_random_mask(image_shape: tuple, missing_ratio: float, seed: int = 42) -> np.ndarray:
    """Create a random binary mask: 1=observed, 0=missing."""
    rng = np.random.default_rng(seed)
    return (rng.random(image_shape) > missing_ratio).astype(np.float32)


def prepare_training_data(
    image: np.ndarray,
    mask: np.ndarray,
    config: TrainConfig,
) -> np.ndarray:
    """Extract visible patches from the current image."""
    if config.mode != "current":
        raise ValueError("Only mode='current' is supported.")

    patches = get_visible_patches(
        image,
        mask,
        config.patch_size,
        config.stride,
        config.visible_ratio_thresh,
    )

    if config.verbose:
        print(f"  Current-image visible patches : {len(patches)}")

    if len(patches) == 0:
        raise RuntimeError(
            "No training patches found. Check image/mask and visible_ratio_thresh."
        )

    rng = np.random.default_rng(42)
    return patches[rng.permutation(len(patches))]


def train_autoencoder(
    image: np.ndarray,
    mask: np.ndarray,
    config: TrainConfig = None,
    save_name: str = "image",
) -> tuple:
    """
    Train the patch autoencoder on visible patches of image.

    Parameters
    ----------
    image:
        Grayscale image, float32 in [0, 1]. In the formal pipeline this is the
        corrupted image with missing pixels set to zero.
    mask:
        Binary mask where 1=observed and 0=missing.
    config:
        Training hyperparameters.
    save_name:
        Prefix used for the saved weight filename.
    """
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
        print(f"  Patch dim              : {patches_np.shape[1]}")

    patches_t = torch.from_numpy(patches_np).float()
    dataset = TensorDataset(patches_t)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    ae = build_autoencoder(
        patch_size=config.patch_size,
        latent_dim=config.latent_dim,
    )
    ae.train()

    optimizer = optim.Adam(
        ae.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-5,
    )

    history = {"train_loss": [], "train_time_s": 0.0}
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            optimizer.zero_grad()
            loss = ae.loss_mean(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if config.verbose and (epoch % config.save_every == 0 or epoch == 1):
            elapsed = time.time() - t_start
            print(
                f"  Epoch [{epoch:3d}/{config.epochs}] "
                f"loss={avg_loss:.6f} elapsed={elapsed:.1f}s"
            )

    history["train_time_s"] = time.time() - t_start
    ae.eval()

    if config.verbose:
        print(f"\nTraining complete in {history['train_time_s']:.1f}s")
        final_loss = ae.reconstruction_error_numpy(patches_np)
        print(f"Final R_AE on training patches: {final_loss:.4f}")

    os.makedirs(config.save_dir, exist_ok=True)
    fname = f"{save_name}_p{config.patch_size}_d{config.latent_dim}_{config.mode}.pt"
    save_path = os.path.join(config.save_dir, fname)
    torch.save(ae.state_dict(), save_path)

    if config.verbose:
        print(f"Weights saved to: {save_path}")

    return ae, history


def load_autoencoder(
    path: str,
    patch_size: int = 8,
    latent_dim: int = 16,
) -> PatchAutoencoder:
    """Load a saved autoencoder weight file."""
    ae = build_autoencoder(patch_size=patch_size, latent_dim=latent_dim)
    ae.load_state_dict(torch.load(path, map_location="cpu"))
    ae.eval()
    print(f"Loaded AE weights from: {path}")
    return ae


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train image-specific patch autoencoder for inpainting."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="cameraman",
        help="Image name or file path.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.5,
        help="Fraction of pixels to remove.",
    )
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="current", choices=["current"])
    parser.add_argument("--save_dir", type=str, default=str(RESULTS_DIR / "ae_weights"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading image: {args.image}")
    image = load_single_image(args.image, image_size=128)
    mask = _make_random_mask(image.shape, missing_ratio=args.mask_ratio, seed=args.seed)
    corrupted = image * mask

    print(f"Image shape  : {image.shape}")
    print(f"Mask ratio   : {args.mask_ratio} ({int((1 - mask.mean()) * 100)}% missing)")

    stride = max(1, args.patch_size // 2)
    visible_threshold = max(0.3, (1.0 - args.mask_ratio) * 0.7)
    print(f"Visible ratio threshold: {visible_threshold:.2f}")

    config = TrainConfig(
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        stride=stride,
        visible_ratio_thresh=visible_threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mode=args.mode,
        save_dir=args.save_dir,
        verbose=True,
    )

    ae, history = train_autoencoder(
        corrupted,
        mask,
        config=config,
        save_name=args.image,
    )

    print("\nTraining loss curve (every 10 epochs):")
    for i, loss in enumerate(history["train_loss"]):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  epoch {i + 1:3d}: {loss:.6f}")
