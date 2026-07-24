"""Named severity ladder with explicit distortion parameters.

Severity is expressed as discrete levels (L1…Ln). Each level also has a
normalized intensity in [0, 1] used by distortion implementations, plus the
resolved physical parameters for reporting (σ, JPEG quality, gain, streak count).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeverityLevel:
    """One step on the distortion severity ladder."""

    name: str  # e.g. "L1"
    index: int  # 1-based
    intensity: float  # normalized in [0, 1]


def build_severity_levels(cfg_levels: list[float] | None = None) -> list[SeverityLevel]:
    """Build named levels from a list of intensities."""
    intensities = list(cfg_levels or [0.25, 0.5, 0.75, 1.0])
    levels: list[SeverityLevel] = []
    for i, intensity in enumerate(intensities, start=1):
        levels.append(
            SeverityLevel(name=f"L{i}", index=i, intensity=float(intensity))
        )
    return levels


def resolve_distortion_params(
    distortion_name: str,
    intensity: float,
    distortion_cfg: dict[str, Any],
) -> dict[str, float | int]:
    """Resolve human-readable parameters for README / severity tables."""
    p = distortion_cfg.get(distortion_name, {})
    if distortion_name == "gaussian_noise":
        base_std = float(p.get("base_std", 40.0))
        return {"std": base_std * intensity}
    if distortion_name == "jpeg_compression":
        max_q = int(p.get("max_quality", 95))
        min_q = int(p.get("min_quality", 10))
        quality = int(round(max_q - intensity * (max_q - min_q)))
        return {"jpeg_quality": int(max(1, min(100, quality)))}
    if distortion_name == "low_light":
        min_gain = float(p.get("min_gain", 0.15))
        max_gain = float(p.get("max_gain", 0.85))
        gamma_max = float(p.get("gamma", 1.4))
        gain = max_gain - intensity * (max_gain - min_gain)
        gamma = 1.0 + intensity * (gamma_max - 1.0)
        return {"gain": gain, "gamma": gamma}
    if distortion_name == "rain":
        base = int(p.get("base_streaks", 120))
        length = int(p.get("streak_length", 18))
        n_streaks = int(round(base * (0.25 + 0.75 * intensity)))
        streak_length = max(4, int(round(length * (0.5 + intensity))))
        return {"n_streaks": n_streaks, "streak_length": streak_length}
    return {}
