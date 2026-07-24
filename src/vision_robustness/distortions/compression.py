"""JPEG compression distortion."""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.distortions.base import Distortion, DistortionResult


class JPEGCompression(Distortion):
    name = "jpeg_compression"

    def __init__(self, max_quality: int = 95, min_quality: int = 10):
        self.max_quality = int(max_quality)
        self.min_quality = int(min_quality)

    def apply(self, image: np.ndarray, intensity: float) -> DistortionResult:
        quality = int(
            round(self.max_quality - intensity * (self.max_quality - self.min_quality))
        )
        quality = int(np.clip(quality, 1, 100))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return DistortionResult(
            image=out,
            name=self.name,
            intensity=intensity,
            params={"quality": quality},
        )
