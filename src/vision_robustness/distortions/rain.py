"""Synthetic rain-streak distortion."""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.distortions.base import Distortion, DistortionResult


class Rain(Distortion):
    name = "rain"

    def __init__(
        self,
        base_streaks: int = 120,
        streak_length: int = 18,
        streak_thickness: int = 1,
        brightness: float = 0.75,
        seed: int | None = 42,
    ):
        self.base_streaks = int(base_streaks)
        self.streak_length = int(streak_length)
        self.streak_thickness = int(streak_thickness)
        self.brightness = float(brightness)
        self.rng = np.random.default_rng(seed)

    def apply(self, image: np.ndarray, intensity: float) -> DistortionResult:
        h, w = image.shape[:2]
        rain_layer = np.zeros((h, w), dtype=np.float32)
        n_streaks = int(round(self.base_streaks * (0.25 + 0.75 * intensity)))
        length = max(4, int(round(self.streak_length * (0.5 + intensity))))
        angle = -25  # degrees, slanted rain

        for _ in range(n_streaks):
            x = int(self.rng.integers(0, w))
            y = int(self.rng.integers(0, h))
            x2 = int(x + length * np.sin(np.deg2rad(angle)))
            y2 = int(y + length * np.cos(np.deg2rad(angle)))
            cv2.line(
                rain_layer,
                (x, y),
                (x2, y2),
                color=float(self.brightness),
                thickness=self.streak_thickness,
                lineType=cv2.LINE_AA,
            )

        # Slight blur to look more natural
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        rain_rgb = np.stack([rain_layer] * 3, axis=-1)
        alpha = 0.35 + 0.45 * intensity
        # Add streaks only (no global darkening) so derain can actually recover.
        out = np.clip(
            image.astype(np.float32) + rain_rgb * 255.0 * alpha,
            0,
            255,
        ).astype(np.uint8)
        return DistortionResult(
            image=out,
            name=self.name,
            intensity=intensity,
            params={"n_streaks": n_streaks, "length": length, "alpha": alpha},
        )
