#!/usr/bin/env python3
"""Aggregate metrics CSVs and regenerate summary figures for the README."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from vision_robustness.config import Config
from vision_robustness.pipeline import PRIMARY_METRICS
from vision_robustness.utils.io import save_dataframe
from vision_robustness.utils.logging import get_logger
from vision_robustness.utils.plotting import plot_bar_comparison, plot_metric_vs_intensity

logger = get_logger("report_figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    results = cfg.path("results_dir")
    fig_dir = cfg.path("figures_dir")
    frames = []
    for stage in ("clean", "distorted", "enhanced", "finetuned"):
        csv_path = results / stage / "metrics.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
            logger.info("Loaded %s", csv_path)

    if not frames:
        logger.warning("No metrics CSVs found under %s — run evaluation scripts first", results)
        return

    all_df = pd.concat(frames, ignore_index=True)
    save_dataframe(results / "all_metrics.csv", all_df)

    for task, metric in PRIMARY_METRICS.items():
        sub = all_df[all_df["task"] == task]
        if sub.empty or metric not in sub.columns:
            continue
        # Curves for stages that have intensity
        curve = sub[sub["intensity"].notna()] if "intensity" in sub.columns else sub
        if not curve.empty:
            plot_metric_vs_intensity(
                curve,
                metric=metric,
                group_col="stage" if "stage" in curve.columns else "distortion",
                save_path=fig_dir / f"report_{task}_{metric}_curves.png",
                title=f"{task}: {metric}",
            )
        # Stage bars averaged
        if "stage" in sub.columns:
            agg = sub.groupby("stage", dropna=False)[metric].mean().reset_index()
            plot_bar_comparison(
                agg,
                metric=metric,
                category="stage",
                hue=None,
                save_path=fig_dir / f"report_{task}_{metric}_stages.png",
                title=f"{task}: mean {metric} by stage",
            )

    logger.info("Report figures written to %s", fig_dir)


if __name__ == "__main__":
    main()
