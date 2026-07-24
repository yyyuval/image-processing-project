#!/usr/bin/env python3
"""Part 3: apply mapped enhancements to distorted images and save outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from vision_robustness.config import Config
from vision_robustness.distortions import build_distortions
from vision_robustness.distortions.severity import build_severity_levels, resolve_distortion_params
from vision_robustness.enhancements import build_enhancements
from vision_robustness.pipeline import load_dataset_from_cfg
from vision_robustness.utils.io import ensure_dir, write_image_rgb
from vision_robustness.utils.logging import get_logger

logger = get_logger("apply_enhancements")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    try:
        ds = load_dataset_from_cfg(cfg, synthetic=args.synthetic)
    except Exception:
        logger.warning("Falling back to synthetic dataset")
        ds = load_dataset_from_cfg(cfg, synthetic=True)

    distortions = build_distortions(cfg.get("distortions", default={}), seed=cfg.seed)
    enhancements = build_enhancements(cfg.get("enhancements", default={}))
    levels = build_severity_levels(cfg.get("intensity_levels", default=[0.2, 0.4, 0.6, 0.8, 1.0]))
    root = ensure_dir(cfg.path("enhanced_dir"))
    n = 0

    for sample in tqdm(ds, desc="enhance"):
        for dist_name, dist in distortions.items():
            enhancer = enhancements[dist_name]
            for severity in levels:
                distorted = dist(sample.image, severity.intensity).image
                resolved = resolve_distortion_params(
                    dist_name, severity.intensity, cfg.get("distortions", default={})
                )
                kwargs = {"intensity": severity.intensity}
                if "std" in resolved:
                    kwargs["noise_std"] = float(resolved["std"])
                enhanced = enhancer(distorted, **kwargs).image
                out = root / dist_name / severity.name / f"{sample.sample_id}.png"
                write_image_rgb(out, enhanced)
                n += 1

    logger.info("Wrote %d enhanced images under %s", n, root)


if __name__ == "__main__":
    main()
