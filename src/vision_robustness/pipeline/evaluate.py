"""Core evaluation pipeline across clean / distorted / enhanced stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from vision_robustness.config import Config
from vision_robustness.data.dataset import Sample, VisionDataset, build_dataset
from vision_robustness.distortions import build_distortions
from vision_robustness.distortions.severity import SeverityLevel, build_severity_levels, resolve_distortion_params
from vision_robustness.enhancements import build_enhancements
from vision_robustness.metrics.schema import summarize_snr_by_severity, wide_to_tidy
from vision_robustness.metrics.snr import psnr, snr_db
from vision_robustness.tasks import VisionTask, build_tasks
from vision_robustness.utils.io import ensure_dir, save_dataframe, save_json, write_image_rgb
from vision_robustness.utils.logging import get_logger
from vision_robustness.utils.plotting import show_image_grid

logger = get_logger(__name__)

PRIMARY_METRICS = {
    "features": "match_accuracy",
    "edges": "edge_f1",
    "detection": "detection_f1",
    "segmentation": "miou",
}


def _row_from_metrics(
    *,
    sample_id: str,
    task: str,
    method: str,
    stage: str,
    distortion: str | None,
    severity: SeverityLevel | None,
    snr: float | None,
    psnr_val: float | None,
    metrics: dict[str, float],
    per_class: dict[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "task": task,
        "method": method,
        "stage": stage,
        "distortion": distortion,
        "severity": None if severity is None else severity.name,
        "severity_index": None if severity is None else severity.index,
        "intensity": None if severity is None else severity.intensity,
        "snr_db": snr,
        "psnr": psnr_val,
        **metrics,
    }
    if per_class:
        for cls, val in per_class.items():
            row[f"class_{cls}"] = val
    return row


def _persist_stage_outputs(results_dir: Path, df: pd.DataFrame) -> None:
    save_dataframe(results_dir / "metrics.csv", df)
    save_dataframe(results_dir / "metrics_tidy.csv", wide_to_tidy(df))
    save_json(results_dir / "summary.json", summarize(df))
    snr_table = summarize_snr_by_severity(df)
    if not snr_table.empty:
        save_dataframe(results_dir / "snr_by_severity.csv", snr_table)


class EvaluationPipeline:
    """Run all tasks on a dataset for one or more stages."""

    def __init__(self, cfg: Config, tasks: dict[str, VisionTask] | None = None):
        self.cfg = cfg
        self.cfg.ensure_dirs()
        self.tasks = tasks or build_tasks(cfg.get("tasks", default={}), cfg.device)
        self.distortions = build_distortions(cfg.get("distortions", default={}), seed=cfg.seed)
        self.enhancements = build_enhancements(cfg.get("enhancements", default={}))
        self.severity_levels = build_severity_levels(
            cfg.get("intensity_levels", default=[0.2, 0.4, 0.6, 0.8, 1.0])
        )

    def run_clean(
        self,
        dataset: VisionDataset,
        results_dir: Path | None = None,
        save_visuals: bool = True,
    ) -> pd.DataFrame:
        results_dir = ensure_dir(results_dir or self.cfg.path("results_dir") / "clean")
        rows: list[dict[str, Any]] = []
        visuals_budget = int(self.cfg.get("evaluation", "visuals_per_stage", default=8))
        saved_visuals = 0

        for sample in tqdm(dataset, desc="clean"):
            for task_name, task in self.tasks.items():
                pred = task.predict(sample.image)
                metrics = task.evaluate(
                    pred,
                    image=sample.image,
                    reference=None,
                    gt_mask=sample.mask,
                )
                rows.append(
                    _row_from_metrics(
                        sample_id=sample.sample_id,
                        task=task_name,
                        method=task.method,
                        stage="clean",
                        distortion=None,
                        severity=None,
                        snr=None,
                        psnr_val=None,
                        metrics=metrics.metrics,
                        per_class=metrics.per_class,
                    )
                )
                if save_visuals and saved_visuals < visuals_budget and pred.visualization is not None:
                    write_image_rgb(
                        results_dir / "visuals" / f"{sample.sample_id}_{task_name}.png",
                        pred.visualization,
                    )
                    saved_visuals += 1

        save_json(results_dir / "sample_ids.json", [s.sample_id for s in dataset])
        df = pd.DataFrame(rows)
        _persist_stage_outputs(results_dir, df)
        logger.info("Clean baseline saved to %s", results_dir)
        return df

    def run_distorted(
        self,
        dataset: VisionDataset,
        results_dir: Path | None = None,
        save_images: bool = True,
        save_visuals: bool = True,
    ) -> pd.DataFrame:
        results_dir = ensure_dir(results_dir or self.cfg.path("results_dir") / "distorted")
        distorted_root = ensure_dir(self.cfg.path("distorted_dir"))
        rows: list[dict[str, Any]] = []
        visuals_budget = int(self.cfg.get("evaluation", "visuals_per_stage", default=8))
        saved_visuals = 0
        clean_refs = self._compute_clean_references(dataset)

        for sample in tqdm(dataset, desc="distorted"):
            for dist_name, distortion in self.distortions.items():
                for severity in self.severity_levels:
                    result = distortion(sample.image, severity.intensity)
                    snr = snr_db(sample.image, result.image)
                    psnr_val = psnr(sample.image, result.image)

                    if save_images:
                        out_path = (
                            distorted_root
                            / dist_name
                            / severity.name
                            / f"{sample.sample_id}.png"
                        )
                        write_image_rgb(out_path, result.image)

                    for task_name, task in self.tasks.items():
                        pred = task.predict(result.image)
                        metrics = task.evaluate(
                            pred,
                            image=result.image,
                            reference=clean_refs[sample.sample_id].get(task_name),
                            gt_mask=sample.mask,
                        )
                        rows.append(
                            _row_from_metrics(
                                sample_id=sample.sample_id,
                                task=task_name,
                                method=task.method,
                                stage="distorted",
                                distortion=dist_name,
                                severity=severity,
                                snr=snr,
                                psnr_val=psnr_val,
                                metrics=metrics.metrics,
                                per_class=metrics.per_class,
                            )
                        )
                        if (
                            save_visuals
                            and saved_visuals < visuals_budget
                            and pred.visualization is not None
                        ):
                            write_image_rgb(
                                results_dir
                                / "visuals"
                                / f"{sample.sample_id}_{dist_name}_{severity.name}_{task_name}.png",
                                pred.visualization,
                            )
                            saved_visuals += 1

        df = pd.DataFrame(rows)
        _persist_stage_outputs(results_dir, df)
        logger.info("Distorted evaluation saved to %s", results_dir)
        return df

    def run_enhanced(
        self,
        dataset: VisionDataset,
        results_dir: Path | None = None,
        save_images: bool = True,
        save_visuals: bool = True,
    ) -> pd.DataFrame:
        results_dir = ensure_dir(results_dir or self.cfg.path("results_dir") / "enhanced")
        enhanced_root = ensure_dir(self.cfg.path("enhanced_dir"))
        rows: list[dict[str, Any]] = []
        visuals_budget = int(self.cfg.get("evaluation", "visuals_per_stage", default=8))
        saved_visuals = 0
        clean_refs = self._compute_clean_references(dataset)

        for sample in tqdm(dataset, desc="enhanced"):
            for dist_name, distortion in self.distortions.items():
                enhancer = self.enhancements.get(dist_name)
                if enhancer is None:
                    logger.warning("No enhancement mapped for %s", dist_name)
                    continue
                for severity in self.severity_levels:
                    distorted = distortion(sample.image, severity.intensity).image
                    enhance_kwargs = self._enhancement_kwargs(dist_name, severity)
                    enhanced = enhancer(distorted, **enhance_kwargs).image
                    snr = snr_db(sample.image, enhanced)
                    psnr_val = psnr(sample.image, enhanced)

                    if save_images:
                        out_path = (
                            enhanced_root
                            / dist_name
                            / severity.name
                            / f"{sample.sample_id}.png"
                        )
                        write_image_rgb(out_path, enhanced)

                    for task_name, task in self.tasks.items():
                        pred = task.predict(enhanced)
                        metrics = task.evaluate(
                            pred,
                            image=enhanced,
                            reference=clean_refs[sample.sample_id].get(task_name),
                            gt_mask=sample.mask,
                        )
                        rows.append(
                            _row_from_metrics(
                                sample_id=sample.sample_id,
                                task=task_name,
                                method=task.method,
                                stage="enhanced",
                                distortion=dist_name,
                                severity=severity,
                                snr=snr,
                                psnr_val=psnr_val,
                                metrics=metrics.metrics,
                                per_class=metrics.per_class,
                            )
                        )
                        if (
                            save_visuals
                            and saved_visuals < visuals_budget
                            and pred.visualization is not None
                        ):
                            write_image_rgb(
                                results_dir
                                / "visuals"
                                / f"{sample.sample_id}_{dist_name}_{severity.name}_{task_name}.png",
                                pred.visualization,
                            )
                            saved_visuals += 1

        df = pd.DataFrame(rows)
        _persist_stage_outputs(results_dir, df)
        logger.info("Enhanced evaluation saved to %s", results_dir)
        return df

    def _enhancement_kwargs(self, dist_name: str, severity: SeverityLevel) -> dict:
        """Pass known degradation cues into enhancers (e.g. noise sigma)."""
        kwargs: dict = {"intensity": severity.intensity}
        resolved = resolve_distortion_params(
            dist_name, severity.intensity, self.cfg.get("distortions", default={})
        )
        if "std" in resolved:
            kwargs["noise_std"] = float(resolved["std"])
        return kwargs

    def _compute_clean_references(
        self, dataset: VisionDataset
    ) -> dict[str, dict[str, Any]]:
        refs: dict[str, dict[str, Any]] = {}
        for sample in dataset:
            refs[sample.sample_id] = {
                name: task.predict(sample.image) for name, task in self.tasks.items()
            }
        return refs


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    summary: dict[str, Any] = {}
    group_cols = [
        c for c in ["stage", "task", "distortion", "severity", "intensity"] if c in df.columns
    ]
    for task, primary in PRIMARY_METRICS.items():
        sub = df[df["task"] == task] if "task" in df.columns else df
        if sub.empty or primary not in sub.columns:
            continue
        grouped = (
            sub.groupby([c for c in group_cols if c in sub.columns], dropna=False)[primary]
            .mean()
            .reset_index()
        )
        summary[task] = grouped.to_dict(orient="records")
    return summary


def make_before_after_grid(
    sample: Sample,
    distorted: Any,
    enhanced: Any,
    save_path: Path,
    snr: float | None = None,
) -> Path:
    snr_txt = f" SNR={snr:.1f}dB" if snr is not None and np_isfinite(snr) else ""
    images = [sample.image, distorted.image, enhanced.image]
    titles = [
        "clean",
        f"{distorted.name} @ {distorted.intensity:.2f}{snr_txt}",
        f"enhanced ({enhanced.name})",
    ]
    path = show_image_grid(images, titles, ncols=3, save_path=save_path)
    assert path is not None
    return path


def np_isfinite(x: float) -> bool:
    import math

    return math.isfinite(float(x))


def load_dataset_from_cfg(cfg: Config, synthetic: bool = False) -> VisionDataset:
    return build_dataset(cfg, use_synthetic=synthetic)
