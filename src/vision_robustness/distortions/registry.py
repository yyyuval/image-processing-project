"""Distortion registry / factory."""

from __future__ import annotations

from typing import Any

from vision_robustness.distortions.base import Distortion
from vision_robustness.distortions.compression import JPEGCompression
from vision_robustness.distortions.low_light import LowLight
from vision_robustness.distortions.noise import GaussianNoise
from vision_robustness.distortions.rain import Rain

DISTORTION_REGISTRY: dict[str, type[Distortion]] = {
    "gaussian_noise": GaussianNoise,
    "jpeg_compression": JPEGCompression,
    "low_light": LowLight,
    "rain": Rain,
}


def build_distortions(cfg_distortions: dict[str, Any], seed: int = 42) -> dict[str, Distortion]:
    """Instantiate all enabled distortions from config."""
    out: dict[str, Distortion] = {}
    for name, params in cfg_distortions.items():
        if not isinstance(params, dict):
            continue
        if not params.get("enabled", True):
            continue
        cls = DISTORTION_REGISTRY[name]
        kwargs = {k: v for k, v in params.items() if k != "enabled"}
        if name in {"gaussian_noise", "rain"} and "seed" not in kwargs:
            kwargs["seed"] = seed
        out[name] = cls(**kwargs)
    return out
