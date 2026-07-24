"""YOLOv8 object detection task (course high-level DL task)."""

from __future__ import annotations

from typing import Any

import numpy as np

from vision_robustness.tasks.base import TaskMetrics, TaskPrediction, VisionTask


class YOLODetectionTask(VisionTask):
    name = "detection"
    method = "yolov8"

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
    ):
        from ultralytics import YOLO

        self.model_name = model_name
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.model = YOLO(model_name)

    def predict(self, image: np.ndarray) -> TaskPrediction:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        result = results[0]
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            names = result.names
            for i in range(len(xyxy)):
                boxes.append(
                    {
                        "xyxy": xyxy[i].tolist(),
                        "cls": int(cls[i]),
                        "label": names.get(int(cls[i]), str(int(cls[i]))),
                        "conf": float(conf[i]),
                    }
                )
        # Ultralytics plot returns BGR
        plotted = result.plot()
        vis = plotted[:, :, ::-1].copy()
        return TaskPrediction(
            task=self.name,
            method=self.method,
            data={"boxes": boxes, "n_detections": len(boxes)},
            visualization=vis,
        )

    def evaluate(
        self,
        prediction: TaskPrediction,
        *,
        image: np.ndarray,
        reference: TaskPrediction | None = None,
        gt_mask: np.ndarray | None = None,
    ) -> TaskMetrics:
        """Evaluate detections with both stability and ground-truth views.

        - Activity: number of detections and mean confidence.
        - Stability: class-aware match against clean-image YOLO boxes.
        - Ground truth: class-agnostic localization vs boxes derived from the
          ADE20K semantic mask (ADE and COCO label spaces differ).
        """
        del image
        pred_boxes = prediction.data.get("boxes", [])
        avg_conf = (
            float(np.mean([b["conf"] for b in pred_boxes])) if pred_boxes else 0.0
        )
        metrics: dict[str, float] = {
            "n_detections": float(len(pred_boxes)),
            "avg_confidence": avg_conf,
        }
        per_class: dict[str, float] = {
            k: float(v) for k, v in _count_by_class(pred_boxes).items()
        }

        if reference is None:
            metrics.update(
                {
                    "mean_iou": 1.0,
                    "detection_precision": 1.0,
                    "detection_recall": 1.0,
                    "detection_f1": 1.0,
                }
            )
        else:
            stab, stab_per_class = match_boxes(
                reference.data.get("boxes", []),
                pred_boxes,
                iou_thresh=self.iou,
                class_aware=True,
            )
            metrics.update(stab)
            per_class = stab_per_class

        if gt_mask is not None:
            from vision_robustness.metrics.gt_boxes import mask_to_boxes

            gt_boxes = mask_to_boxes(gt_mask)
            gt_metrics, _ = match_boxes(
                gt_boxes, pred_boxes, iou_thresh=self.iou, class_aware=False
            )
            metrics["gt_mean_iou"] = gt_metrics["mean_iou"]
            metrics["gt_precision"] = gt_metrics["detection_precision"]
            metrics["gt_recall"] = gt_metrics["detection_recall"]
            metrics["gt_f1"] = gt_metrics["detection_f1"]

        return TaskMetrics(
            task=self.name,
            method=self.method,
            metrics=metrics,
            per_class=per_class,
        )


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)


def match_boxes(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    iou_thresh: float = 0.5,
    class_aware: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    matched_gt = set()
    matched_pred = set()
    ious: list[float] = []
    per_class_tp: dict[str, int] = {}
    per_class_gt: dict[str, int] = {}

    for g in gt_boxes:
        label = str(g.get("label", g.get("cls")))
        per_class_gt[label] = per_class_gt.get(label, 0) + 1

    candidates = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            if class_aware and int(g.get("cls", -1)) != int(p.get("cls", -2)):
                continue
            iou = _box_iou(g["xyxy"], p["xyxy"])
            if iou >= iou_thresh:
                candidates.append((iou, gi, pi, str(g.get("label", g.get("cls")))))
    candidates.sort(reverse=True)

    for iou, gi, pi, label in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        ious.append(iou)
        per_class_tp[label] = per_class_tp.get(label, 0) + 1

    tp = float(len(matched_gt))
    fp = float(len(pred_boxes) - len(matched_pred))
    fn = float(len(gt_boxes) - len(matched_gt))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    per_class_recall = {
        label: per_class_tp.get(label, 0) / max(per_class_gt.get(label, 1), 1)
        for label in per_class_gt
    }
    return (
        {
            "mean_iou": mean_iou,
            "detection_precision": float(precision),
            "detection_recall": float(recall),
            "detection_f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        {k: float(v) for k, v in per_class_recall.items()},
    )


def _count_by_class(boxes: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in boxes:
        label = str(b.get("label", b.get("cls")))
        out[label] = out.get(label, 0) + 1
    return out
