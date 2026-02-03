"""Latent vector mapping from facial features with RBF interpolation and input lerp smoothing."""

from __future__ import annotations

import numpy as np

from .tracker import FacialFeatures


class LatentMapper:
    """Maps :class:`FacialFeatures` to a StyleGAN latent vector ``z``.

    The mapper pre-samples a set of anchor vectors in latent space and uses
    RBF (radial basis function) weights driven by ``jaw_openness`` and
    ``head_yaw`` to blend between them.  Eyebrow tension adds noise (entropy),
    and stillness blends toward a dedicated pure-tone anchor.  Linear
    interpolation (lerp) smooths the input features over time for fluid morphing.

    Parameters
    ----------
    z_dim : int
        Dimensionality of the latent space (typically 512).
    n_anchors : int
        Number of anchor vectors to pre-sample.
    lerp_alpha : float
        Linear interpolation factor for input smoothing (0 = instant, 1 = frozen).
    rbf_sigma : float
        Bandwidth of the RBF kernel.
    noise_scale : float
        Maximum standard deviation of noise added at full eyebrow tension.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        z_dim: int = 512,
        n_anchors: int = 8,
        lerp_alpha: float = 0.8,
        rbf_sigma: float = 0.3,
        noise_scale: float = 0.4,
        seed: int | None = None,
    ) -> None:
        self.z_dim = z_dim
        self.lerp_alpha = lerp_alpha
        self.rbf_sigma = rbf_sigma
        self.noise_scale = noise_scale

        rng = np.random.default_rng(seed)

        # Pre-sample anchor vectors from truncated normal
        self._anchors = rng.standard_normal((n_anchors, z_dim)).astype(np.float32)
        self._anchors = np.clip(self._anchors, -2.0, 2.0)

        # Anchor positions in 2-D feature space (jaw_openness, head_yaw)
        # spread evenly across [0, 1] x [0, 1]
        grid_side = int(np.ceil(np.sqrt(n_anchors)))
        positions = []
        for i in range(n_anchors):
            row = i // grid_side
            col = i % grid_side
            positions.append([
                (col + 0.5) / grid_side,
                (row + 0.5) / grid_side,
            ])
        self._anchor_positions = np.array(positions, dtype=np.float32)

        # Pure-tone anchor: low truncation → near origin → cleaner generation
        self._pure_tone_anchor = (
            rng.standard_normal(z_dim).astype(np.float32) * 0.3
        )

        self._prev_input: np.ndarray | None = None
        self._rng = rng

    def __call__(self, features: FacialFeatures) -> np.ndarray:
        """Map facial features to a latent vector.

        Parameters
        ----------
        features : FacialFeatures
            Normalized facial features (all in ``[0, 1]``).

        Returns
        -------
        np.ndarray, shape ``(z_dim,)``
        """
        raw_input = np.array(
            [features.jaw_openness, features.eyebrow_tension,
             features.head_yaw, features.stillness],
            dtype=np.float32,
        )

        # Lerp: smooth the input features over time
        if self._prev_input is not None:
            raw_input = self._prev_input + (1.0 - self.lerp_alpha) * (raw_input - self._prev_input)
        self._prev_input = raw_input.copy()

        jaw, brow, yaw, still = raw_input
        query = np.array([jaw, yaw], dtype=np.float32)

        # RBF weights over anchors
        dists = np.linalg.norm(self._anchor_positions - query, axis=1)
        weights = np.exp(-0.5 * (dists / self.rbf_sigma) ** 2)
        weight_sum = weights.sum()
        if weight_sum > 1e-8:
            weights /= weight_sum

        # Weighted sum of anchors → base_z
        base_z = weights @ self._anchors

        # Add noise scaled by eyebrow tension
        noise = self._rng.standard_normal(self.z_dim).astype(np.float32)
        z = base_z + noise * self.noise_scale * brow

        # Blend toward pure-tone anchor proportional to stillness
        z = z * (1.0 - still) + self._pure_tone_anchor * still

        return z
