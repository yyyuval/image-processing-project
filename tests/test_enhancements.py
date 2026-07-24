"""Unit tests for enhancements."""

from __future__ import annotations

import numpy as np

from vision_robustness.distortions.noise import GaussianNoise
from vision_robustness.enhancements import build_enhancements


def test_enhancements_preserve_shape():
    rng = np.random.default_rng(0)
    clean = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    noisy = GaussianNoise(base_std=40, seed=0)(clean, 0.8).image
    mapping = {
        "gaussian_noise": "non_local_means",
        "jpeg_compression": "bilateral_restore",
        "low_light": "clahe",
        "rain": "derain",
    }
    enhancers = build_enhancements(mapping)
    for name, enhancer in enhancers.items():
        kwargs = {"noise_std": 32.0} if name == "gaussian_noise" else {}
        out = enhancer(noisy, **kwargs)
        assert out.image.shape == clean.shape
        assert out.image.dtype == np.uint8


def test_nlm_improves_psnr_on_gaussian_noise():
    from vision_robustness.enhancements.denoise_nlm import NonLocalMeans
    from vision_robustness.metrics.snr import psnr

    yy, xx = np.mgrid[0:96, 0:96]
    clean = np.stack(
        [
            (xx * 2).astype(np.uint8),
            (yy * 2).astype(np.uint8),
            np.full((96, 96), 120, dtype=np.uint8),
        ],
        axis=-1,
    )
    clean[30:60, 30:60] = (220, 40, 40)
    noisy = GaussianNoise(base_std=40, seed=0)(clean, 0.8).image
    restored = NonLocalMeans()(noisy, noise_std=32.0).image
    assert psnr(clean, restored) > psnr(clean, noisy) + 0.5
