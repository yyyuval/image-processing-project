"""Canny edge detection task (course low-level geometric feature task)."""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.data.visualize import draw_edges
from vision_robustness.tasks.base import TaskMetrics, TaskPrediction, VisionTask


class CannyEdgesTask(VisionTask):
    name = "edges"
    method = "canny"

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)

    def predict(self, image: np.ndarray) -> TaskPrediction:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        vis = draw_edges(image, edges)
        return TaskPrediction(
            task=self.name,
            method=self.method,
            data={"edges": edges, "edge_density": float(np.mean(edges > 0))},
            visualization=vis,
        )

    def evaluate(
        self,
        prediction: TaskPrediction,
        *,
        image: np.ndarray,
        reference: TaskPrediction | None = None,
        gt_mask: np.ndarray | None = None,
    ) -> TaskMetrics:
        del image, gt_mask
        edges = prediction.data["edges"]
        density = float(prediction.data.get("edge_density", np.mean(edges > 0)))
        if reference is None:
            return TaskMetrics(
                task=self.name,
                method=self.method,
                metrics={"edge_density": density, "edge_f1": 1.0, "edge_iou": 1.0},
            )

        gt = reference.data["edges"] > 0
        pred = edges > 0
        tp = float(np.logical_and(gt, pred).sum())
        fp = float(np.logical_and(~gt, pred).sum())
        fn = float(np.logical_and(gt, ~pred).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        return TaskMetrics(
            task=self.name,
            method=self.method,
            metrics={
                "edge_density": density,
                "edge_precision": float(precision),
                "edge_recall": float(recall),
                "edge_f1": float(f1),
                "edge_iou": float(iou),
            },
        )
