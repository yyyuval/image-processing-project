"""Derive axis-aligned boxes from semantic masks for GT localization checks."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def mask_to_boxes(
    mask: np.ndarray,
    *,
    ignore_index: int = 255,
    min_area: int = 64,
) -> list[dict[str, Any]]:
    """Convert a semantic segmentation mask into instance-like GT boxes.

    For each non-background class id, connected components become separate boxes.
    Class ids remain ADE20K label ids (not COCO), so detection GT matching is
    intentionally *class-agnostic* (localization quality).
    """
    if mask is None:
        return []
    mask = mask.astype(np.int32)
    boxes: list[dict[str, Any]] = []
    for cls in np.unique(mask):
        cls = int(cls)
        if cls == ignore_index:
            continue
        binary = (mask == cls).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            boxes.append(
                {
                    "xyxy": [float(x), float(y), float(x + w), float(y + h)],
                    "cls": cls,
                    "label": f"ade_{cls}",
                    "area": float(area),
                }
            )
    return boxes
