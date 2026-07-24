"""Prepare YOLO fine-tuning data: distort images, transfer labels from clean predictions."""

from __future__ import annotations

from pathlib import Path

import yaml
from tqdm import tqdm

from vision_robustness.config import Config
from vision_robustness.data.dataset import VisionDataset
from vision_robustness.distortions import build_distortions
from vision_robustness.tasks.detection import YOLODetectionTask
from vision_robustness.utils.io import ensure_dir, write_image_rgb
from vision_robustness.utils.logging import get_logger

logger = get_logger(__name__)


def _xyxy_to_yolo(xyxy: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def prepare_yolo_finetune_dataset(
    cfg: Config,
    dataset: VisionDataset,
    detector: YOLODetectionTask | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Create a YOLO dataset from clean detections applied to distorted images.

    Course Part 4: create labels from clean, fine-tune on distorted images.
    COCO class ids are remapped to contiguous ``0..N-1`` (required by Ultralytics).
    """
    train_cfg = cfg.get("training", "yolo", default={})
    dist_name = train_cfg.get("distortion", "gaussian_noise")
    intensity = float(train_cfg.get("intensity", 0.6))
    output_dir = ensure_dir(
        output_dir or cfg.root / train_cfg.get("output_dir", "results/finetuned/yolo") / "data"
    )

    images_dir = ensure_dir(output_dir / "images" / "train")
    labels_dir = ensure_dir(output_dir / "labels" / "train")

    # Clear previous labels/images so remapped ids stay consistent
    for folder in (images_dir, labels_dir):
        for p in folder.glob("*"):
            if p.is_file():
                p.unlink()

    distortions = build_distortions(cfg.get("distortions", default={}), seed=cfg.seed)
    if dist_name not in distortions:
        raise KeyError(f"Distortion {dist_name} not configured")
    distortion = distortions[dist_name]

    if detector is None:
        det_cfg = cfg.get("tasks", "detection", default={})
        detector = YOLODetectionTask(
            model_name=det_cfg.get("model_name", "yolov8n.pt"),
            conf=det_cfg.get("conf", 0.25),
            iou=det_cfg.get("iou", 0.5),
            device=str(cfg.device) if cfg.device.type != "cpu" else None,
        )

    # First pass: collect detections so we can build a contiguous class map
    samples_payload: list[tuple[str, object, list[dict]]] = []
    coco_to_name: dict[int, str] = {}
    for sample in tqdm(dataset, desc="prepare-yolo-scan"):
        clean_pred = detector.predict(sample.image)
        boxes = clean_pred.data.get("boxes", [])
        if not boxes:
            continue
        for b in boxes:
            coco_to_name[int(b["cls"])] = str(b.get("label", b["cls"]))
        samples_payload.append((sample.sample_id, sample.image, boxes))

    coco_ids = sorted(coco_to_name)
    coco_to_local = {coco_id: i for i, coco_id in enumerate(coco_ids)}
    local_names = {i: coco_to_name[coco_id] for i, coco_id in enumerate(coco_ids)}

    n_written = 0
    for sample_id, image, boxes in tqdm(samples_payload, desc="prepare-yolo-write"):
        distorted = distortion(image, intensity).image
        h, w = distorted.shape[:2]
        write_image_rgb(images_dir / f"{sample_id}.jpg", distorted)

        label_lines = []
        for b in boxes:
            local_id = coco_to_local[int(b["cls"])]
            cx, cy, bw, bh = _xyxy_to_yolo(b["xyxy"], w, h)
            label_lines.append(f"{local_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(labels_dir / f"{sample_id}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
        n_written += 1

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": local_names,
        "nc": len(local_names),
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    # Keep reverse map for debugging / report
    with open(output_dir / "coco_to_local.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "coco_to_local": {str(k): v for k, v in coco_to_local.items()},
                "local_names": local_names,
            },
            f,
            sort_keys=False,
        )

    logger.info(
        "Prepared YOLO fine-tune set: %d images, %d classes (remapped 0..%d) → %s",
        n_written,
        len(local_names),
        max(len(local_names) - 1, 0),
        yaml_path,
    )
    return yaml_path
