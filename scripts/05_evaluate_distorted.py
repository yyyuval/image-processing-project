#!/usr/bin/env python3
"""Part 2: evaluate all tasks on distorted images (metrics vs SNR/intensity)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_robustness.config import Config
from vision_robustness.pipeline import EvaluationPipeline, PRIMARY_METRICS, load_dataset_from_cfg
from vision_robustness.utils.logging import get_logger
from vision_robustness.utils.plotting import plot_metric_vs_intensity

logger = get_logger("eval_distorted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--tasks", nargs="*", default=None)
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

    df = pipe.run_distorted(ds, save_images=False)
    fig_dir = cfg.path("figures_dir")
    for task, metric in PRIMARY_METRICS.items():
        sub = df[df["task"] == task]
        if sub.empty or metric not in sub.columns:
            continue
        plot_metric_vs_intensity(
            sub,
            metric=metric,
            save_path=fig_dir / f"distorted_{task}_{metric}.png",
            title=f"{task}: {metric} vs intensity (distorted)",
        )
    logger.info("Distorted evaluation complete (%d rows)", len(df))


if __name__ == "__main__":
    main()
