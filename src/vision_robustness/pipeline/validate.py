"""Validate that required experiment artifacts exist and look well-formed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vision_robustness.config import Config
from vision_robustness.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_METRIC_COLS = {
    "sample_id",
    "task",
    "method",
    "stage",
    "snr_db",
    "psnr",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def print(self) -> None:
        for line in self.info:
            logger.info(line)
        for line in self.warnings:
            logger.warning(line)
        for line in self.errors:
            logger.error(line)
        logger.info("Validation %s", "PASSED" if self.ok else "FAILED")


def validate_project(cfg: Config, stages: list[str] | None = None) -> ValidationReport:
    """Check directories, metric CSVs, and basic schema consistency."""
    report = ValidationReport()
    stages = stages or ["clean", "distorted", "enhanced"]
    results = cfg.path("results_dir")

    for key in ("raw_dir", "distorted_dir", "enhanced_dir", "results_dir", "figures_dir"):
        path = cfg.path(key)
        if path.exists():
            report.info.append(f"Path exists: {path}")
        else:
            report.warnings.append(f"Path missing (ok before that stage): {path}")

    # Dataset cache optional
    cache = cfg.get("dataset", "cache_dir", default="")
    if cache:
        cache_path = Path(cache)
        if not cache_path.is_absolute():
            cache_path = cfg.root / cache_path
        if (cache_path / "index.json").exists():
            report.info.append(f"Dataset cache found: {cache_path}")
        else:
            report.warnings.append(
                f"Dataset cache not found at {cache_path} (use --synthetic or download)"
            )

    for stage in stages:
        csv_path = results / stage / "metrics.csv"
        if not csv_path.exists():
            report.errors.append(f"Missing metrics for stage '{stage}': {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        report.info.append(f"{stage}: {len(df)} rows in {csv_path.name}")
        missing = REQUIRED_METRIC_COLS - set(df.columns)
        # snr/psnr may be empty on clean — columns should still exist after distorted
        if stage == "clean":
            missing -= {"snr_db", "psnr"}
            if "sample_id" not in df.columns or "task" not in df.columns:
                report.errors.append(f"{stage}: missing core columns")
        elif missing:
            report.errors.append(f"{stage}: missing columns {sorted(missing)}")

        if stage != "clean" and "severity" in df.columns and df["severity"].isna().all():
            report.warnings.append(f"{stage}: severity column is entirely empty")

        tidy = results / stage / "metrics_tidy.csv"
        if tidy.exists():
            report.info.append(f"{stage}: tidy metrics present")
        else:
            report.warnings.append(f"{stage}: metrics_tidy.csv not found")

    # Distortion coverage
    dist_cfg = cfg.get("distortions", default={}) or {}
    enabled = [k for k, v in dist_cfg.items() if isinstance(v, dict) and v.get("enabled", True)]
    distorted_csv = results / "distorted" / "metrics.csv"
    if distorted_csv.exists() and enabled:
        ddf = pd.read_csv(distorted_csv)
        if "distortion" in ddf.columns:
            found = set(ddf["distortion"].dropna().unique())
            missing_d = set(enabled) - found
            if missing_d:
                report.errors.append(f"Distorted metrics missing distortions: {sorted(missing_d)}")
            else:
                report.info.append(f"All enabled distortions present: {sorted(found)}")

    return report
