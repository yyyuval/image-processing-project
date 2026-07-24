"""Unit tests for classical task metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vision_robustness.data import SyntheticDataset
from vision_robustness.tasks.edges import CannyEdgesTask
from vision_robustness.tasks.features import ORBFeaturesTask
from vision_robustness.tasks.segmentation import compute_miou


def test_orb_self_match_is_high():
    ds = SyntheticDataset(n=1, size=(192, 192), seed=0)
    img = ds[0].image
    task = ORBFeaturesTask(n_features=500)
    ref = task.predict(img)
    pred = task.predict(img)
    metrics = task.evaluate(pred, image=img, reference=ref)
    assert metrics.metrics["match_accuracy"] > 0.5


def test_canny_self_f1_is_one():
    ds = SyntheticDataset(n=1, size=(128, 128), seed=1)
    img = ds[0].image
    task = CannyEdgesTask()
    ref = task.predict(img)
    metrics = task.evaluate(ref, image=img, reference=ref)
    assert metrics.metrics["edge_f1"] == pytest.approx(1.0, abs=1e-6)
    assert metrics.metrics["edge_iou"] == pytest.approx(1.0, abs=1e-6)


def test_miou_perfect_and_zero():
    gt = np.zeros((20, 20), dtype=np.int32)
    gt[5:15, 5:15] = 1
    miou, per_class, acc = compute_miou(gt, gt, ignore_index=0)
    assert miou == pytest.approx(1.0, abs=1e-6)
    assert per_class["1"] == pytest.approx(1.0, abs=1e-6)
    assert acc == pytest.approx(1.0, abs=1e-6)

    pred = np.zeros_like(gt)
    pred[5:15, 5:15] = 2
    miou0, _, _ = compute_miou(pred, gt, ignore_index=0)
    assert miou0 == pytest.approx(0.0, abs=1e-6)
