#!/usr/bin/env python3
"""Part 4: evaluate a fine-tuned YOLO checkpoint on distorted images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tqdm import tqdm

from vision_robustness.config import Config
from vision_robustness.distortions import build_distortions
from vision_robustness.distortions.severity import build_severity_levels
from vision_robustness.metrics.snr import psnr, snr_db
from vision_robustness.pipeline import load_dataset_from_cfg
from vision_robustness.tasks.detection import YOLODetectionTask
from vision_robustness.utils.io import ensure_dir, save_dataframe, save_json
from vision_robustness.utils.logging import get_logger
from vision_robustness.utils.plotting import plot_metric_vs_intensity

logger = get_logger("eval_finetuned")


def _remap_finetuned_pred_to_coco(pred, local_to_coco: dict[int, int], local_names: dict[int, str]):
    """Map fine-tuned local class ids back to original COCO ids for fair matching."""
    boxes = []
    for b in pred.data.get("boxes", []):
        local_id = int(b["cls"])
        coco_id = int(local_to_coco.get(local_id, local_id))
        label = local_names.get(local_id, b.get("label", str(coco_id)))
        boxes.append({**b, "cls": coco_id, "label": label})
    pred.data["boxes"] = boxes
    pred.data["n_detections"] = len(boxes)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to fine-tuned YOLO weights (default: results/finetuned/yolo/train/weights/best.pt)",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()
    try:
        ds = load_dataset_from_cfg(cfg, synthetic=args.synthetic)
    except Exception:
        logger.warning("Falling back to synthetic dataset")
        ds = load_dataset_from_cfg(cfg, synthetic=True)

    weights = args.weights
    if weights is None:
        weights = str(
            cfg.root
            / cfg.get("training", "yolo", "output_dir", default="results/finetuned/yolo")
            / "train"
            / "weights"
            / "best.pt"
        )
    if not Path(weights).exists():
        raise FileNotFoundError(
            f"Fine-tuned weights not found: {weights}. Run scripts/08_finetune_yolo.py first."
        )

    det_cfg = cfg.get("tasks", "detection", default={})
    baseline = YOLODetectionTask(
        model_name=det_cfg.get("model_name", "yolov8n.pt"),
        conf=det_cfg.get("conf", 0.25),
        iou=det_cfg.get("iou", 0.5),
        device=str(cfg.device) if cfg.device.type != "cpu" else None,
    )
    finetuned = YOLODetectionTask(
        model_name=weights,
        conf=det_cfg.get("conf", 0.25),
        iou=det_cfg.get("iou", 0.5),
        device=str(cfg.device) if cfg.device.type != "cpu" else None,
    )

    map_path = (
        cfg.root
        / cfg.get("training", "yolo", "output_dir", default="results/finetuned/yolo")
        / "data"
        / "coco_to_local.yaml"
    )
    local_to_coco: dict[int, int] = {}
    local_names: dict[int, str] = {}
    if map_path.exists():
        import yaml

        with open(map_path, encoding="utf-8") as f:
            mapping = yaml.safe_load(f)
        local_to_coco = {int(v): int(k) for k, v in mapping.get("coco_to_local", {}).items()}
        local_names = {int(k): str(v) for k, v in mapping.get("local_names", {}).items()}
        logger.info("Loaded class remap (%d classes) from %s", len(local_to_coco), map_path)
    else:
        logger.warning("No coco_to_local.yaml found — evaluating with raw fine-tuned class ids")

    distortions = build_distortions(cfg.get("distortions", default={}), seed=cfg.seed)
    levels = build_severity_levels(cfg.get("intensity_levels", default=[0.2, 0.4, 0.6, 0.8, 1.0]))
    rows = []

    for sample in tqdm(ds, desc="finetuned-eval"):
        clean_ref = baseline.predict(sample.image)
        for dist_name, dist in distortions.items():
            for severity in levels:
                distorted = dist(sample.image, severity.intensity).image
                pred = finetuned.predict(distorted)
                if local_to_coco:
                    pred = _remap_finetuned_pred_to_coco(pred, local_to_coco, local_names)
                metrics = finetuned.evaluate(
                    pred, image=distorted, reference=clean_ref, gt_mask=sample.mask
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": "detection",
                        "method": "yolov8_finetuned",
                        "stage": "finetuned",
                        "distortion": dist_name,
                        "severity": severity.name,
                        "severity_index": severity.index,
                        "intensity": severity.intensity,
                        "snr_db": snr_db(sample.image, distorted),
                        "psnr": psnr(sample.image, distorted),
                        **metrics.metrics,
                    }
                )

    df = pd.DataFrame(rows)
    out_dir = ensure_dir(cfg.path("results_dir") / "finetuned")
    save_dataframe(out_dir / "metrics.csv", df)
    from vision_robustness.metrics.schema import wide_to_tidy

    save_dataframe(out_dir / "metrics_tidy.csv", wide_to_tidy(df))
    save_json(
        out_dir / "summary.json",
        df.groupby(["distortion", "severity"])["detection_f1"]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
    )
    plot_metric_vs_intensity(
        df,
        metric="detection_f1",
        save_path=cfg.path("figures_dir") / "finetuned_detection_f1.png",
        title="Fine-tuned YOLO detection_f1 vs intensity",
    )
    logger.info("Fine-tuned evaluation complete (%d rows)", len(df))


if __name__ == "__main__":
    main()
