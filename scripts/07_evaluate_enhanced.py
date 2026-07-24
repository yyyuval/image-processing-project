#!/usr/bin/env python3
"""Part 3: evaluate all tasks on enhanced images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from vision_robustness.config import Config
from vision_robustness.pipeline import EvaluationPipeline, PRIMARY_METRICS, load_dataset_from_cfg
from vision_robustness.utils.io import save_dataframe
from vision_robustness.utils.logging import get_logger
from vision_robustness.utils.plotting import plot_bar_comparison, plot_metric_vs_intensity

logger = get_logger("eval_enhanced")


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

    df = pipe.run_enhanced(ds, save_images=False)
    fig_dir = cfg.path("figures_dir")

    for task, metric in PRIMARY_METRICS.items():
        sub = df[df["task"] == task]
        if sub.empty or metric not in sub.columns:
            continue
        plot_metric_vs_intensity(
            sub,
            metric=metric,
            save_path=fig_dir / f"enhanced_{task}_{metric}.png",
            title=f"{task}: {metric} vs intensity (enhanced)",
        )

    # Optional comparison with distorted metrics if present
    distorted_csv = cfg.path("results_dir") / "distorted" / "metrics.csv"
    if distorted_csv.exists():
        ddf = pd.read_csv(distorted_csv)
        both = pd.concat([ddf.assign(stage="distorted"), df.assign(stage="enhanced")], ignore_index=True)
        save_dataframe(cfg.path("results_dir") / "comparison_distorted_vs_enhanced.csv", both)
        for task, metric in PRIMARY_METRICS.items():
            sub = both[both["task"] == task]
            if sub.empty or metric not in sub.columns:
                continue
            # Average over intensity for bar chart
            agg = sub.groupby(["stage", "distortion"], dropna=False)[metric].mean().reset_index()
            plot_bar_comparison(
                agg,
                metric=metric,
                category="distortion",
                hue="stage",
                save_path=fig_dir / f"compare_{task}_{metric}.png",
                title=f"{task}: distorted vs enhanced",
            )

    logger.info("Enhanced evaluation complete (%d rows)", len(df))


if __name__ == "__main__":
    main()
