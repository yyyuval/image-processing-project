#!/usr/bin/env python3
"""Part 1: run all tasks on clean images and save baseline metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_robustness.config import Config
from vision_robustness.pipeline import EvaluationPipeline, load_dataset_from_cfg
from vision_robustness.utils.logging import get_logger

logger = get_logger("clean_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional subset of tasks: features edges detection segmentation",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    try:
        ds = load_dataset_from_cfg(cfg, synthetic=args.synthetic)
    except Exception:
        logger.warning("Falling back to synthetic dataset")
        ds = load_dataset_from_cfg(cfg, synthetic=True)

    pipe = EvaluationPipeline(cfg)
    if args.tasks:
        pipe.tasks = {k: v for k, v in pipe.tasks.items() if k in args.tasks}

    df = pipe.run_clean(ds)
    logger.info("Clean rows: %d\n%s", len(df), df.head())


if __name__ == "__main__":
    main()
