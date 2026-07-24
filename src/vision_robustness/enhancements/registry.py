"""Enhancement registry / factory."""

from __future__ import annotations

from vision_robustness.enhancements.base import Enhancement
from vision_robustness.enhancements.bilateral_restore import BilateralRestore
from vision_robustness.enhancements.clahe import CLAHEEnhancement
from vision_robustness.enhancements.denoise_nlm import NonLocalMeans
from vision_robustness.enhancements.derain import Derain

ENHANCEMENT_REGISTRY: dict[str, type[Enhancement]] = {
    "non_local_means": NonLocalMeans,
    "bilateral_restore": BilateralRestore,
    "clahe": CLAHEEnhancement,
    "derain": Derain,
}


def build_enhancements(mapping: dict[str, str]) -> dict[str, Enhancement]:
    """Map distortion name → enhancement instance from config."""
    out: dict[str, Enhancement] = {}
    for distortion_name, enhancement_name in mapping.items():
        cls = ENHANCEMENT_REGISTRY[enhancement_name]
        out[distortion_name] = cls()
    return out
