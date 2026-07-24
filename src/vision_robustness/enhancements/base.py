"""Enhancement base types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EnhancementResult:
    image: np.ndarray
    name: str
    params: dict


class Enhancement(ABC):
    name: str = "base"

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs) -> EnhancementResult:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, **kwargs) -> EnhancementResult:
        return self.apply(image, **kwargs)
