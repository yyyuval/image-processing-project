"""Fine-tune YOLOv8 on distorted images with clean-derived labels."""

from __future__ import annotations

from pathlib import Path

from vision_robustness.config import Config
from vision_robustness.utils.io import ensure_dir
from vision_robustness.utils.logging import get_logger

logger = get_logger(__name__)


def finetune_yolo(
    cfg: Config,
    data_yaml: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    from ultralytics import YOLO

    train_cfg = cfg.get("training", "yolo", default={})
    output_dir = ensure_dir(
        output_dir or cfg.root / train_cfg.get("output_dir", "results/finetuned/yolo")
    )
    model_name = cfg.get("tasks", "detection", "model_name", default="yolov8n.pt")
    model = YOLO(model_name)

    epochs = int(train_cfg.get("epochs", 30))
    imgsz = int(train_cfg.get("imgsz", 640))
    batch = int(train_cfg.get("batch", 8))

    logger.info(
        "Fine-tuning %s for %d epochs on %s", model_name, epochs, data_yaml
    )
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    weights = Path(output_dir) / "train" / "weights" / "best.pt"
    logger.info("Fine-tuned weights: %s", weights)
    return weights
