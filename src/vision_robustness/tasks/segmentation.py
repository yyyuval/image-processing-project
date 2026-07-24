"""SegFormer semantic segmentation (course high-level DL task with GT)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vision_robustness.data.visualize import overlay_mask
from vision_robustness.tasks.base import TaskMetrics, TaskPrediction, VisionTask


class SegFormerTask(VisionTask):
    name = "segmentation"
    method = "segformer"

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str | torch.device | None = None,
    ):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> TaskPrediction:
        pil = Image.fromarray(image)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, C, h, w)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)
        vis = overlay_mask(image, pred_mask)
        return TaskPrediction(
            task=self.name,
            method=self.method,
            data={"mask": pred_mask, "num_labels": int(pred_mask.max()) + 1},
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
        del image, reference
        pred = prediction.data["mask"].astype(np.int32)
        if gt_mask is None:
            return TaskMetrics(
                task=self.name,
                method=self.method,
                metrics={"miou": float("nan"), "pixel_accuracy": float("nan")},
            )
        gt = gt_mask.astype(np.int32)
        miou, per_class, pixel_acc = compute_miou(pred, gt)
        return TaskMetrics(
            task=self.name,
            method=self.method,
            metrics={"miou": miou, "pixel_accuracy": pixel_acc},
            per_class=per_class,
        )


def compute_miou(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_index: int = 255,
    num_classes: int | None = None,
) -> tuple[float, dict[str, float], float]:
    """Mean IoU over classes present in GT (ignoring ``ignore_index``)."""
    if num_classes is None:
        valid_vals = np.concatenate(
            [pred.ravel(), gt.ravel()[gt.ravel() != ignore_index]]
        )
        num_classes = int(valid_vals.max()) + 1 if valid_vals.size else 1
    per_class: dict[str, float] = {}
    ious: list[float] = []
    valid = gt != ignore_index
    pixel_acc = float((pred[valid] == gt[valid]).mean()) if valid.any() else 0.0

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        gt_c = gt == cls
        if not gt_c.any():
            continue
        pred_c = pred == cls
        inter = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        iou = float(inter / (union + 1e-8))
        per_class[str(cls)] = iou
        ious.append(iou)
    miou = float(np.mean(ious)) if ious else 0.0
    return miou, per_class, pixel_acc
