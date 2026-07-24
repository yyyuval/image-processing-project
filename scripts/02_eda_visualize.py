#!/usr/bin/env python3
"""EDA: visualize samples, masks, and distortion/enhancement before-after grids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_robustness.config import Config
from vision_robustness.data import build_dataset, visualize_samples
from vision_robustness.distortions import build_distortions
from vision_robustness.distortions.severity import resolve_distortion_params
from vision_robustness.enhancements import build_enhancements
from vision_robustness.metrics.snr import psnr, snr_db
from vision_robustness.pipeline.evaluate import make_before_after_grid
from vision_robustness.utils.logging import get_logger

logger = get_logger("eda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    try:
        ds = build_dataset(cfg, use_synthetic=args.synthetic)
    except Exception:
        logger.warning("Falling back to synthetic dataset")
        ds = build_dataset(cfg, use_synthetic=True)

    fig_dir = cfg.path("figures_dir")
    visualize_samples(list(ds)[:8], fig_dir / "eda_samples.png")

    distortions = build_distortions(cfg.get("distortions", default={}), seed=cfg.seed)
    enhancements = build_enhancements(cfg.get("enhancements", default={}))
    sample = ds[0]
    intensity = 0.8

    for dist_name, distortion in distortions.items():
        distorted = distortion(sample.image, intensity)
        enhancer = enhancements[dist_name]
        resolved = resolve_distortion_params(
            dist_name, intensity, cfg.get("distortions", default={})
        )
        enhance_kwargs = {"intensity": intensity}
        if "std" in resolved:
            enhance_kwargs["noise_std"] = float(resolved["std"])
        enhanced = enhancer(distorted.image, **enhance_kwargs)
        out = fig_dir / f"before_after_{dist_name}.png"
        make_before_after_grid(
            sample,
            distorted,
            enhanced,
            out,
            snr=snr_db(sample.image, distorted.image),
        )
        # Log restoration quality vs clean for quick sanity checks
        logger.info(
            "Wrote %s | distorted PSNR=%.2f → enhanced PSNR=%.2f",
            out.name,
            psnr(sample.image, distorted.image),
            psnr(sample.image, enhanced.image),
        )


if __name__ == "__main__":
    main()
