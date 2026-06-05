"""
autoencoder.py
--------------
Patch autoencoder A_θ : R^{p²} → R^{p²}

Architecture (improved):
    Encoder:  input_dim → h1 → h2 → latent_dim   (ReLU + BN)
    Decoder:  latent_dim → h2 → h1 → input_dim   (ReLU + BN hidden, Sigmoid output)

Deeper than original to better capture complex textures.
"""

import torch
import torch.nn as nn
import numpy as np


class PatchAutoencoder(nn.Module):
    """
    Improved fully-connected autoencoder for image patches.

    Parameters
    ----------
    patch_size  : int   side length p; input dim = p*p
    latent_dim  : int   bottleneck dimension (ablation: 8, 16, 32)
    """

    def __init__(self, patch_size: int = 8, latent_dim: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim

        input_dim = patch_size * patch_size   # e.g. 64 for p=8
        h1 = input_dim                        # 64
        h2 = max(input_dim // 2, latent_dim * 2)  # 32

        # Encoder: input_dim → h1 → h2 → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder: latent_dim → h2 → h1 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, input_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def loss(self, x):
        return torch.sum((x - self.forward(x)) ** 2)

    def loss_mean(self, x):
        return torch.mean((x - self.forward(x)) ** 2)

    @torch.no_grad()
    def predict_numpy(self, patches_np):
        self.eval()
        x = torch.from_numpy(patches_np).float()
        if x.shape[0] == 1:
            # BN needs >1 sample; handle edge case
            x = x.repeat(2, 1)
            return self.forward(x)[:1].cpu().numpy()
        return self.forward(x).cpu().numpy()

    @torch.no_grad()
    def reconstruction_error_numpy(self, patches_np):
        self.eval()
        x = torch.from_numpy(patches_np).float()
        return float(torch.sum((x - self.forward(x)) ** 2).item())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        print("=" * 45)
        print(f"PatchAutoencoder  patch={self.patch_size}  latent={self.latent_dim}")
        print(f"  parameters: {self.count_parameters()}")
        print("=" * 45)


def build_autoencoder(patch_size=8, latent_dim=16):
    return PatchAutoencoder(patch_size=patch_size, latent_dim=latent_dim)


if __name__ == "__main__":
    print("Testing improved PatchAutoencoder...\n")
    for p in [4, 8, 12]:
        for d in [8, 16, 32]:
            ae = build_autoencoder(patch_size=p, latent_dim=d)
            x = torch.rand(100, p * p)
            out = ae(x)
            assert out.shape == x.shape
            assert 0.0 <= out.min().item() and out.max().item() <= 1.0
            print(f"  p={p:2d}  d={d:2d}  params={ae.count_parameters():5d}  ✓")
    print("\nAll tests passed.")
