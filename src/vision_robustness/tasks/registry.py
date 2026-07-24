"""Task factory."""

from __future__ import annotations

from typing import Any

import torch

from vision_robustness.tasks.base import VisionTask
from vision_robustness.tasks.detection import YOLODetectionTask
from vision_robustness.tasks.edges import CannyEdgesTask
from vision_robustness.tasks.features import ORBFeaturesTask
from vision_robustness.tasks.segmentation import SegFormerTask


def build_tasks(cfg_tasks: dict[str, Any], device: torch.device) -> dict[str, VisionTask]:
    tasks: dict[str, VisionTask] = {}

    if "features" in cfg_tasks:
        p = cfg_tasks["features"]
        tasks["features"] = ORBFeaturesTask(
            n_features=p.get("n_features", 1000),
            match_ratio=p.get("match_ratio", 0.75),
        )

    if "edges" in cfg_tasks:
        p = cfg_tasks["edges"]
        tasks["edges"] = CannyEdgesTask(
            low_threshold=p.get("low_threshold", 50),
            high_threshold=p.get("high_threshold", 150),
        )

    if "detection" in cfg_tasks:
        p = cfg_tasks["detection"]
        tasks["detection"] = YOLODetectionTask(
            model_name=p.get("model_name", "yolov8n.pt"),
            conf=p.get("conf", 0.25),
            iou=p.get("iou", 0.5),
            device=str(device) if device.type != "cpu" else None,
        )

    if "segmentation" in cfg_tasks:
        p = cfg_tasks["segmentation"]
        tasks["segmentation"] = SegFormerTask(
            model_name=p.get("model_name", "nvidia/segformer-b0-finetuned-ade-512-512"),
            device=device,
        )

    return tasks
