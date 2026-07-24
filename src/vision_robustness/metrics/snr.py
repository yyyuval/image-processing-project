"""SNR / PSNR helpers used as distortion intensity measures."""

from __future__ import annotations

import numpy as np


def mse(clean: np.ndarray, distorted: np.ndarray) -> float:
    a = clean.astype(np.float64)
    b = distorted.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(clean: np.ndarray, distorted: np.ndarray, data_range: float = 255.0) -> float:
    err = mse(clean, distorted)
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(err))


def snr_db(clean: np.ndarray, distorted: np.ndarray) -> float:
    """Signal-to-noise ratio in dB treating (clean - distorted) as noise."""
    signal = clean.astype(np.float64)
    noise = signal - distorted.astype(np.float64)
    signal_power = float(np.mean(signal**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 1e-12:
        return float("inf")
    if signal_power <= 1e-12:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))
