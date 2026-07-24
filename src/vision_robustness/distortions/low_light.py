"""Low-light / darkening distortion."""

from __future__ import annotations

import numpy as np

from vision_robustness.distortions.base import Distortion, DistortionResult


class LowLight(Distortion):
    name = "low_light"

    def __init__(self, min_gain: float = 0.15, max_gain: float = 0.85, gamma: float = 1.4):
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.gamma = float(gamma)

    def apply(self, image: np.ndarray, intensity: float) -> DistortionResult:
        # Higher intensity → darker image
        gain = self.max_gain - intensity * (self.max_gain - self.min_gain)
        x = image.astype(np.float32) / 255.0
        x = np.clip(x * gain, 0.0, 1.0)
        # Gamma > 1 further darkens midtones
        gamma = 1.0 + intensity * (self.gamma - 1.0)
        x = np.power(x, gamma)
        out = (x * 255.0).astype(np.uint8)
        return DistortionResult(
            image=out,
            name=self.name,
            intensity=intensity,
            params={"gain": gain, "gamma": gamma},
        )
