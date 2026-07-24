"""CLAHE enhancement for low-light images (course recommendation)."""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.enhancements.base import Enhancement, EnhancementResult


class CLAHEEnhancement(Enhancement):
    name = "clahe"

    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray, **kwargs) -> EnhancementResult:
        del kwargs
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l2 = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
        return EnhancementResult(
            image=out,
            name=self.name,
            params={"clip_limit": self.clip_limit, "tile_grid_size": self.tile_grid_size},
        )
