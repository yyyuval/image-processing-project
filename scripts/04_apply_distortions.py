#!/usr/bin/env python3
"""Part 2: apply all distortions at all severity levels and save images."""

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
from vision_robustness.metrics.snr import psnr, snr_db
from vision_robustness.pipeline import load_dataset_from_cfg
from vision_robustness.utils.io import ensure_dir, save_json, write_image_rgb
from vision_robustness.utils.logging import get_logger

logger = get_logger("apply_distortions")


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
    levels = build_severity_levels(cfg.get("intensity_levels", default=[0.2, 0.4, 0.6, 0.8, 1.0]))
    root = ensure_dir(cfg.path("distorted_dir"))
    meta = []

    for sample in tqdm(ds, desc="distort"):
        for name, dist in distortions.items():
            for severity in levels:
                result = dist(sample.image, severity.intensity)
                out = root / name / severity.name / f"{sample.sample_id}.png"
                write_image_rgb(out, result.image)
                meta.append(
                    {
                        "sample_id": sample.sample_id,
                        "distortion": name,
                        "severity": severity.name,
                        "severity_index": severity.index,
                        "intensity": severity.intensity,
                        "snr_db": snr_db(sample.image, result.image),
                        "psnr": psnr(sample.image, result.image),
                        "params": result.params,
                        "resolved_params": resolve_distortion_params(
                            name, severity.intensity, cfg.get("distortions", default={})
                        ),
                        "path": str(out.relative_to(cfg.root)),
                    }
                )

    save_json(root / "manifest.json", meta)
    logger.info("Wrote %d distorted images under %s", len(meta), root)


if __name__ == "__main__":
    main()
