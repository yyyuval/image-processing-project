"""Bilateral filtering + mild upsample restore for JPEG compression artifacts."""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.enhancements.base import Enhancement, EnhancementResult


class BilateralRestore(Enhancement):
    """Course suggestion: interpolation with bilateral filtering."""

    name = "bilateral_restore"

    def __init__(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0):
        self.d = int(d)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def apply(self, image: np.ndarray, **kwargs) -> EnhancementResult:
        del kwargs
        h, w = image.shape[:2]
        # Mild up/down sampling as "interpolation" restore then bilateral smooth
        up = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        restored = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
        out = cv2.bilateralFilter(
            restored,
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space,
        )
        return EnhancementResult(
            image=out,
            name=self.name,
            params={
                "d": self.d,
                "sigma_color": self.sigma_color,
                "sigma_space": self.sigma_space,
            },
        )
