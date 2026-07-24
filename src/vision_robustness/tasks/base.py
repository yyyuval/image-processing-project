"""Task interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskPrediction:
    task: str
    method: str
    # Task-specific payload (keypoints, edge map, boxes, mask, …)
    data: dict[str, Any] = field(default_factory=dict)
    # Optional RGB visualization
    visualization: np.ndarray | None = None


@dataclass
class TaskMetrics:
    task: str
    method: str
    metrics: dict[str, float]
    per_class: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class VisionTask(ABC):
    name: str
    method: str

    @abstractmethod
    def predict(self, image: np.ndarray) -> TaskPrediction:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        prediction: TaskPrediction,
        *,
        image: np.ndarray,
        reference: TaskPrediction | None = None,
        gt_mask: np.ndarray | None = None,
    ) -> TaskMetrics:
        """Evaluate a prediction.

        Classical tasks typically compare against a *reference* prediction on the
        clean image. DL tasks with GT use `gt_mask` / derived labels.
        """
        raise NotImplementedError
