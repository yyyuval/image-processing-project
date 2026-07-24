"""Visualization helpers for samples and annotations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision_robustness.data.dataset import Sample
from vision_robustness.utils.plotting import show_image_grid


def colorize_mask(mask: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Map integer mask to a deterministic RGB color image."""
    if num_classes is None:
        num_classes = int(mask.max()) + 1
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(max(num_classes, 1), 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    clipped = np.clip(mask, 0, len(palette) - 1).astype(np.int32)
    return palette[clipped]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = colorize_mask(mask)
    out = image.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_keypoints(image: np.ndarray, keypoints) -> np.ndarray:
    out = image.copy()
    return cv2.drawKeypoints(
        out,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_edges(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[edges > 0] = (0, 255, 255)
    return out


def visualize_samples(
    samples: list[Sample],
    save_path: str | Path,
    max_n: int = 8,
) -> Path:
    images = []
    titles = []
    for sample in samples[:max_n]:
        if sample.mask is not None:
            images.append(overlay_mask(sample.image, sample.mask))
            titles.append(f"{sample.sample_id} + mask")
        else:
            images.append(sample.image)
            titles.append(sample.sample_id)
    path = show_image_grid(images, titles, ncols=4, save_path=save_path)
    assert path is not None
    return path
