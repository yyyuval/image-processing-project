"""Unit tests for distortions and SNR."""

from __future__ import annotations

import numpy as np

from vision_robustness.distortions import build_distortions
from vision_robustness.metrics.snr import mse, psnr, snr_db


def _image(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_all_distortions_change_image_and_preserve_shape():
    img = _image()
    cfg = {
        "gaussian_noise": {"enabled": True, "base_std": 30.0},
        "jpeg_compression": {"enabled": True, "max_quality": 95, "min_quality": 10},
        "low_light": {"enabled": True, "min_gain": 0.2, "max_gain": 0.9, "gamma": 1.4},
        "rain": {"enabled": True, "base_streaks": 30, "streak_length": 10},
    }
    dists = build_distortions(cfg, seed=0)
    assert set(dists) == set(cfg)
    for name, dist in dists.items():
        out = dist(img, 0.8)
        assert out.image.shape == img.shape
        assert out.image.dtype == np.uint8
        assert out.name == name
        assert mse(img, out.image) > 0


def test_higher_noise_intensity_lowers_snr():
    img = _image()
    dist = build_distortions(
        {"gaussian_noise": {"enabled": True, "base_std": 40.0}}, seed=1
    )["gaussian_noise"]
    weak = dist(img, 0.2).image
    strong = dist(img, 1.0).image
    assert snr_db(img, weak) > snr_db(img, strong)
    assert psnr(img, weak) > psnr(img, strong)
