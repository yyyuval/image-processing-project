"""ORB feature detection + matching accuracy (course low-level task)."""

from __future__ import annotations

import cv2
import numpy as np

from vision_robustness.data.visualize import draw_keypoints
from vision_robustness.tasks.base import TaskMetrics, TaskPrediction, VisionTask


class ORBFeaturesTask(VisionTask):
    name = "features"
    method = "orb"

    def __init__(self, n_features: int = 1000, match_ratio: float = 0.75):
        self.n_features = int(n_features)
        self.match_ratio = float(match_ratio)
        self.orb = cv2.ORB_create(nfeatures=self.n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def predict(self, image: np.ndarray) -> TaskPrediction:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        vis = draw_keypoints(image, keypoints)
        return TaskPrediction(
            task=self.name,
            method=self.method,
            data={
                "keypoints": keypoints,
                "descriptors": descriptors,
                "n_keypoints": len(keypoints),
            },
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
        n_kp = float(prediction.data.get("n_keypoints", 0))
        if reference is None:
            return TaskMetrics(
                task=self.name,
                method=self.method,
                metrics={"n_keypoints": n_kp, "match_accuracy": 1.0},
            )

        desc_ref = reference.data.get("descriptors")
        desc_pred = prediction.data.get("descriptors")
        if desc_ref is None or desc_pred is None or len(desc_ref) < 2 or len(desc_pred) < 2:
            return TaskMetrics(
                task=self.name,
                method=self.method,
                metrics={"n_keypoints": n_kp, "match_accuracy": 0.0, "n_good_matches": 0.0},
            )

        knn = self.matcher.knnMatch(desc_ref, desc_pred, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.match_ratio * n.distance:
                good.append(m)

        # Match accuracy: good matches relative to reference keypoints
        n_ref = max(len(reference.data.get("keypoints", [])), 1)
        match_accuracy = len(good) / n_ref
        return TaskMetrics(
            task=self.name,
            method=self.method,
            metrics={
                "n_keypoints": n_kp,
                "n_good_matches": float(len(good)),
                "match_accuracy": float(match_accuracy),
            },
        )
