"""Gaussian noise distortion (course: noise / speckle family)."""

from __future__ import annotations

import numpy as np

from vision_robustness.distortions.base import Distortion, DistortionResult


class GaussianNoise(Distortion):
    name = "gaussian_noise"

    def __init__(self, base_std: float = 40.0, seed: int | None = 42):
        self.base_std = float(base_std)
        self.rng = np.random.default_rng(seed)

    def apply(self, image: np.ndarray, intensity: float) -> DistortionResult:
        std = self.base_std * intensity
        noise = self.rng.normal(0.0, std, size=image.shape)
        out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return DistortionResult(
            image=out,
            name=self.name,
            intensity=intensity,
            params={"std": std},
        )
