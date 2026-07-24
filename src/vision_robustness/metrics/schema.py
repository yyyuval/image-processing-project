"""Tidy (long-form) metrics schema for report-ready CSVs."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

# Canonical long-form columns used across stages.
TIDY_COLUMNS = [
    "sample_id",
    "stage",
    "task",
    "method",
    "distortion",
    "severity",
    "severity_index",
    "intensity",
    "snr_db",
    "psnr",
    "metric_kind",  # activity | stability | ground_truth
    "metric_name",
    "metric_value",
    "class_id",
]


# Which metrics are "activity" (volume of output) vs correctness proxies.
ACTIVITY_METRICS = {
    "n_keypoints",
    "edge_density",
    "n_detections",
    "avg_confidence",
}

GT_METRICS = {
    "miou",
    "pixel_accuracy",
    "gt_mean_iou",
    "gt_precision",
    "gt_recall",
    "gt_f1",
}

STABILITY_METRICS = {
    "match_accuracy",
    "n_good_matches",
    "edge_f1",
    "edge_iou",
    "edge_precision",
    "edge_recall",
    "mean_iou",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "tp",
    "fp",
    "fn",
}


def metric_kind(name: str) -> str:
    if name in ACTIVITY_METRICS:
        return "activity"
    if name in GT_METRICS or name.startswith("class_"):
        # per-class IoU columns from segmentation
        if name.startswith("class_"):
            return "ground_truth"
        return "ground_truth"
    if name in STABILITY_METRICS:
        return "stability"
    return "other"


def wide_to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide pipeline metrics into a long/tidy table for the report."""
    if df.empty:
        return pd.DataFrame(columns=TIDY_COLUMNS)

    id_cols = [
        c
        for c in [
            "sample_id",
            "stage",
            "task",
            "method",
            "distortion",
            "severity",
            "severity_index",
            "intensity",
            "snr_db",
            "psnr",
        ]
        if c in df.columns
    ]
    value_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="metric_name",
        value_name="metric_value",
    )
    long = long.dropna(subset=["metric_value"])
    long["metric_kind"] = long["metric_name"].map(metric_kind)
    long["class_id"] = long["metric_name"].map(
        lambda n: n.replace("class_", "") if isinstance(n, str) and n.startswith("class_") else None
    )
    for col in TIDY_COLUMNS:
        if col not in long.columns:
            long[col] = None
    return long[TIDY_COLUMNS].reset_index(drop=True)


def summarize_snr_by_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean SNR/PSNR per distortion × severity (for README severity table)."""
    cols = [c for c in ["distortion", "severity", "severity_index", "intensity", "snr_db", "psnr"] if c in df.columns]
    if not cols or "snr_db" not in df.columns:
        return pd.DataFrame()
    group = [c for c in ["distortion", "severity", "severity_index", "intensity"] if c in df.columns]
    return (
        df.dropna(subset=["snr_db"])
        .groupby(group, dropna=False)[["snr_db", "psnr"]]
        .mean()
        .reset_index()
        .sort_values(group)
    )


def concat_stage_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
