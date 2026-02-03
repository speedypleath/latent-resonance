"""GAN inference wrapper for spectrogram generation."""

import os
import pickle
import subprocess
import sys

import numpy as np
import torch

SG3_DIR = "/tmp/stylegan3"


def _ensure_stylegan3() -> None:
    """Clone the StyleGAN3 repo if not already present (needed for pickle resolution)."""
    if not os.path.exists(SG3_DIR):
        subprocess.run(
            ["git", "clone", "--quiet",
             "https://github.com/NVlabs/stylegan3.git", SG3_DIR],
            check=True,
        )
    if SG3_DIR not in sys.path:
        sys.path.insert(0, SG3_DIR)


class SpectrogramGenerator:
    """Wraps a trained StyleGAN2/3 generator for spectrogram inference.

    Parameters
    ----------
    checkpoint_path : str
        Path to a ``.pkl`` checkpoint file containing ``G_ema``.
    device : str | None
        Torch device string.  Defaults to ``"cuda"`` if available.
    truncation_psi : float
        Truncation psi for generation (lower = less variation, higher quality).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        truncation_psi: float = 0.5,
    ) -> None:
        _ensure_stylegan3()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.truncation_psi = truncation_psi

        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)

        self.G = data["G_ema"]
        if self.device == "cuda":
            self.G = self.G.cuda()
        else:
            self.G = self.G.cpu()
        self.G.eval()
        self.z_dim: int = self.G.z_dim

    @torch.no_grad()
    def generate(self, z: np.ndarray) -> np.ndarray:
        """Generate a spectrogram from a latent vector.

        Parameters
        ----------
        z : np.ndarray, shape ``(z_dim,)``
            Latent vector.

        Returns
        -------
        np.ndarray, shape ``(512, 512)``, float32 in ``[-1, 1]``
            Generated spectrogram (single-channel).
        """
        z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(self.device)
        img = self.G(z_tensor, None, truncation_psi=self.truncation_psi)
        # img shape: (1, C, H, W) — take first channel
        spec = img[0, 0].cpu().numpy()
        # GAN output is in image orientation (high freq at row 0) because
        # training PNGs were flipped.  Invert so row 0 = lowest mel bin,
        # matching the natural spectrogram layout.
        return spec[::-1].copy()
