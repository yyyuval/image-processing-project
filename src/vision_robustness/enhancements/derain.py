"""Classical derain for synthetic rain streaks.

Detect thin bright outliers vs a local median (impulse-style reasoning from the
Noise Smoothing lecture), keep only elongated responses, then inpaint those
pixels. This targets streaks instead of smoothing the whole image.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.enhancements.base import Enhancement, EnhancementResult


class Derain(Enhancement):
    name = "derain"

    def __init__(
        self,
        median_ksize: int = 5,
        bright_delta: int = 28,
        inpaint_radius: int = 2,
    ):
        self.median_ksize = int(median_ksize)
        self.bright_delta = int(bright_delta)
        self.inpaint_radius = int(inpaint_radius)

    def apply(self, image: np.ndarray, **kwargs) -> EnhancementResult:
        del kwargs
        med = cv2.medianBlur(image, self.median_ksize)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.int16)
        gray_med = cv2.cvtColor(med, cv2.COLOR_RGB2GRAY).astype(np.int16)
        residual = gray - gray_med

        mask = (residual > self.bright_delta).astype(np.uint8)

        # Keep elongated structures (streaks); drop isolated speckles
        line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

        if int(mask.sum()) == 0:
            out = image.copy()
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            restored = cv2.inpaint(
                bgr, mask * 255, self.inpaint_radius, cv2.INPAINT_TELEA
            )
            out = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

        return EnhancementResult(
            image=out,
            name=self.name,
            params={
                "median_ksize": self.median_ksize,
                "bright_delta": self.bright_delta,
                "inpaint_radius": self.inpaint_radius,
                "n_masked": int(mask.sum()),
            },
        )
