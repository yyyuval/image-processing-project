"""Distortion base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DistortionResult:
    image: np.ndarray
    name: str
    intensity: float
    params: dict


class Distortion(ABC):
    """Apply a parametric distortion at a normalized intensity in [0, 1]."""

    name: str = "base"

    @abstractmethod
    def apply(self, image: np.ndarray, intensity: float) -> DistortionResult:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, intensity: float) -> DistortionResult:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        return self.apply(image, intensity)
