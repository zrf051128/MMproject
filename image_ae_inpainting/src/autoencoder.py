"""
autoencoder.py
--------------
Tiny MLP patch autoencoder  A_θ : R^{p²} → R^{p²}

Architecture (paper default, patch_size=8):
    Encoder:  64 → 32 → latent_dim   (ReLU activations)
    Decoder:  latent_dim → 32 → 64   (ReLU hidden, Sigmoid output)

The paper ablation tests latent_dim ∈ {8, 16, 32} and patch_size ∈ {4, 8, 12}.
This module auto-scales hidden layers to match patch_size.

Usage:
    from autoencoder import PatchAutoencoder

    ae = PatchAutoencoder(patch_size=8, latent_dim=16)
    out = ae(x)          # x: Tensor (N, 64)
    loss = ae.loss(x)    # MSE reconstruction loss
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PatchAutoencoder(nn.Module):
    """
    Tiny fully-connected autoencoder for image patches.

    Parameters
    ----------
    patch_size  : int   side length p; input dim = p*p
    latent_dim  : int   bottleneck dimension d  (paper ablation: 8, 16, 32)

    Architecture scales with patch_size:
      input_dim  = p*p
      hidden_dim = input_dim // 2  (e.g. 32 for p=8)

    Encoder:  input_dim → hidden_dim → latent_dim   (ReLU)
    Decoder:  latent_dim → hidden_dim → input_dim   (ReLU hidden, Sigmoid out)
    """

    def __init__(self, patch_size: int = 8, latent_dim: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        input_dim  = patch_size * patch_size
        hidden_dim = max(input_dim // 2, latent_dim + 4)  # at least latent+4

        # Encoder: input_dim → hidden_dim → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder: latent_dim → hidden_dim → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),               # output in [0, 1] to match normalized patches
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for ReLU layers; zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, p*p) → z: (N, latent_dim)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) → x_rec: (N, p*p)"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Parameters
        ----------
        x : Tensor (N, p*p), float32 in [0, 1]

        Returns
        -------
        x_rec : Tensor (N, p*p), float32 in [0, 1]
        """
        z     = self.encode(x)
        x_rec = self.decode(z)
        return x_rec

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        MSE reconstruction loss:
            L = Σ_i ‖p_i − A_θ(p_i)‖²   (sum, not mean, matching paper notation)

        Parameters
        ----------
        x : Tensor (N, p*p)

        Returns
        -------
        scalar Tensor
        """
        x_rec = self.forward(x)
        return torch.sum((x - x_rec) ** 2)

    def loss_mean(self, x: torch.Tensor) -> torch.Tensor:
        """MSE per-element (mean), useful for monitoring convergence."""
        x_rec = self.forward(x)
        return torch.mean((x - x_rec) ** 2)

    # ── Numpy convenience wrappers ────────────────────────────────────────────

    @torch.no_grad()
    def predict_numpy(self, patches_np: np.ndarray) -> np.ndarray:
        """
        Run forward pass on numpy array (no gradients).

        Parameters
        ----------
        patches_np : np.ndarray (N, p*p), float32

        Returns
        -------
        recon : np.ndarray (N, p*p), float32
        """
        x   = torch.from_numpy(patches_np).float()
        out = self.forward(x)
        return out.cpu().numpy()

    @torch.no_grad()
    def reconstruction_error_numpy(self, patches_np: np.ndarray) -> float:
        """Return scalar R_AE = Σ ‖p - A_θ(p)‖² on numpy patches."""
        x     = torch.from_numpy(patches_np).float()
        x_rec = self.forward(x)
        return float(torch.sum((x - x_rec) ** 2).item())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        print("=" * 45)
        print("PatchAutoencoder")
        print(f"  patch_size  : {self.patch_size}")
        print(f"  input_dim   : {self.patch_size ** 2}")
        print(f"  latent_dim  : {self.latent_dim}")
        print(f"  parameters  : {self.count_parameters()}")
        print("  Encoder:")
        for layer in self.encoder:
            print(f"    {layer}")
        print("  Decoder:")
        for layer in self.decoder:
            print(f"    {layer}")
        print("=" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build all ablation variants
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder(patch_size: int = 8,
                      latent_dim: int = 16) -> PatchAutoencoder:
    """
    Convenience factory.  Creates and returns an untrained PatchAutoencoder.

    Example ablation variants (paper Table 4):
        build_autoencoder(patch_size=8, latent_dim=8)
        build_autoencoder(patch_size=8, latent_dim=16)
        build_autoencoder(patch_size=8, latent_dim=32)
        build_autoencoder(patch_size=4, latent_dim=16)
        build_autoencoder(patch_size=12, latent_dim=16)
    """
    ae = PatchAutoencoder(patch_size=patch_size, latent_dim=latent_dim)
    return ae


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing PatchAutoencoder...\n")

    for p_size in [4, 8, 12]:
        for l_dim in [8, 16, 32]:
            ae = build_autoencoder(patch_size=p_size, latent_dim=l_dim)
            # random batch of patches
            x  = torch.rand(100, p_size * p_size)
            out = ae(x)
            loss_val = ae.loss(x)
            assert out.shape == x.shape, "Shape mismatch!"
            assert 0.0 <= out.min().item() <= out.max().item() <= 1.0, "Output out of [0,1]!"
            print(f"  patch_size={p_size:2d}  latent_dim={l_dim:2d}  "
                  f"params={ae.count_parameters():4d}  "
                  f"loss={loss_val.item():.4f}  ✓")

    print("\nAll tests passed.")

    # Show one full summary
    print()
    ae = build_autoencoder(patch_size=8, latent_dim=16)
    ae.summary()
