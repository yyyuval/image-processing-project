#!/usr/bin/env python3
"""Part 4: prepare distorted YOLO dataset and fine-tune YOLOv8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_robustness.config import Config
from vision_robustness.pipeline import load_dataset_from_cfg
from vision_robustness.training import finetune_yolo, prepare_yolo_finetune_dataset
from vision_robustness.utils.logging import get_logger

logger = get_logger("finetune_yolo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    if args.epochs is not None:
        cfg.raw.setdefault("training", {}).setdefault("yolo", {})["epochs"] = args.epochs

    try:
        ds = load_dataset_from_cfg(cfg, synthetic=args.synthetic)
    except Exception:
        logger.warning("Falling back to synthetic dataset")
        ds = load_dataset_from_cfg(cfg, synthetic=True)

    data_yaml = prepare_yolo_finetune_dataset(cfg, ds)
    if args.prepare_only:
        logger.info("Prepared dataset only: %s", data_yaml)
        return

    weights = finetune_yolo(cfg, data_yaml)
    logger.info("Done. Best weights at %s", weights)


if __name__ == "__main__":
    main()
