"""Tests for mask→box GT helpers and tidy metrics schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vision_robustness.metrics.gt_boxes import mask_to_boxes
from vision_robustness.metrics.schema import metric_kind, wide_to_tidy
from vision_robustness.tasks.detection import match_boxes


def test_mask_to_boxes_finds_components():
    mask = np.full((40, 40), 255, dtype=np.int32)
    mask[5:15, 5:15] = 3
    mask[20:30, 20:35] = 7
    boxes = mask_to_boxes(mask, min_area=10)
    assert len(boxes) == 2
    labels = {b["label"] for b in boxes}
    assert labels == {"ade_3", "ade_7"}


def test_class_agnostic_matching():
    gt = [{"xyxy": [0, 0, 10, 10], "cls": 3, "label": "ade_3"}]
    pred = [{"xyxy": [1, 1, 9, 9], "cls": 0, "label": "person", "conf": 0.9}]
    metrics, _ = match_boxes(gt, pred, iou_thresh=0.3, class_aware=False)
    assert metrics["detection_f1"] > 0.9


def test_wide_to_tidy_assigns_kinds():
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "stage": "distorted",
                "task": "features",
                "method": "orb",
                "distortion": "gaussian_noise",
                "severity": "L1",
                "severity_index": 1,
                "intensity": 0.2,
                "snr_db": 20.0,
                "psnr": 25.0,
                "n_keypoints": 100,
                "match_accuracy": 0.8,
            }
        ]
    )
    tidy = wide_to_tidy(df)
    kinds = dict(zip(tidy["metric_name"], tidy["metric_kind"]))
    assert kinds["n_keypoints"] == "activity"
    assert kinds["match_accuracy"] == "stability"
    assert metric_kind("miou") == "ground_truth"
