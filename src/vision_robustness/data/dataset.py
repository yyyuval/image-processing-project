"""Dataset loading: ADE20K via Hugging Face + synthetic fallback for tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image

from vision_robustness.config import Config
from vision_robustness.utils.io import ensure_dir, save_json, write_image_rgb, write_mask
from vision_robustness.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Sample:
    """One evaluation sample with optional GT mask and metadata."""

    sample_id: str
    image: np.ndarray  # RGB uint8
    mask: np.ndarray | None = None  # HxW int labels (ADE20K)
    meta: dict[str, Any] | None = None


class VisionDataset:
    """Iterable dataset interface used by all pipelines."""

    def __len__(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Sample:  # pragma: no cover - interface
        raise NotImplementedError

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]


class SyntheticDataset(VisionDataset):
    """Small synthetic RGB + mask dataset for offline smoke tests."""

    def __init__(self, n: int = 8, size: tuple[int, int] = (256, 256), seed: int = 42):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self._samples = [self._make_sample(i) for i in range(n)]

    def _make_sample(self, i: int) -> Sample:
        h, w = self.size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.int32)

        # Background gradient
        yy, xx = np.mgrid[0:h, 0:w]
        image[..., 0] = (xx / max(w - 1, 1) * 180).astype(np.uint8)
        image[..., 1] = (yy / max(h - 1, 1) * 180).astype(np.uint8)
        image[..., 2] = 80

        # Rectangle (class 1)
        x0, y0 = 40 + i * 3, 50
        x1, y1 = x0 + 70, y0 + 90
        image[y0:y1, x0:x1] = (220, 40, 40)
        mask[y0:y1, x0:x1] = 1

        # Circle (class 2)
        cy, cx, r = h // 2 + 20, w // 2 + 30, 35
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        image[circle] = (40, 180, 220)
        mask[circle] = 2

        # Textured patch for features/edges
        patch = self.rng.integers(0, 255, size=(60, 60, 3), dtype=np.uint8)
        image[20:80, w - 90 : w - 30] = patch
        mask[20:80, w - 90 : w - 30] = 3

        return Sample(
            sample_id=f"synth_{i:03d}",
            image=image,
            mask=mask,
            meta={"source": "synthetic", "classes": [0, 1, 2, 3]},
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


class ADE20KDataset(VisionDataset):
    """ADE20K / SceneParse150 via Hugging Face `datasets`.

    Falls back to a previously exported local cache under `cache_dir`.
    """

    def __init__(
        self,
        split: str = "validation",
        max_samples: int | None = 50,
        image_size: tuple[int, int] | None = (512, 512),
        cache_dir: str | Path | None = None,
        hf_id: str = "scene_parse_150",
        seed: int = 42,
    ):
        self.split = split
        self.max_samples = max_samples
        self.image_size = image_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.hf_id = hf_id
        self.seed = seed
        self._samples: list[Sample] = []
        self._load()

    def _load(self) -> None:
        if self.cache_dir and (self.cache_dir / "index.json").exists():
            self._load_from_cache()
            if self.max_samples is None or len(self._samples) >= self.max_samples:
                return
            logger.info(
                "Cache has %d samples but max_samples=%s — re-fetching from Hugging Face",
                len(self._samples),
                self.max_samples,
            )
            self._samples = []
        try:
            self._load_from_hf()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load ADE20K from Hugging Face (%s). "
                "Use SyntheticDataset or run scripts/01_download_data.py.",
                exc,
            )
            raise

    def _load_from_hf(self) -> None:
        from datasets import load_dataset

        logger.info("Loading Hugging Face dataset %s [%s]…", self.hf_id, self.split)
        ds = load_dataset(self.hf_id, split=self.split)
        n = len(ds) if self.max_samples is None else min(self.max_samples, len(ds))
        for i in range(n):
            row = ds[i]
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            image = image.convert("RGB")
            arr = np.asarray(image, dtype=np.uint8)

            mask = row.get("annotation") or row.get("label") or row.get("segmentation")
            mask_arr = None
            if mask is not None:
                if isinstance(mask, Image.Image):
                    mask_arr = np.asarray(mask)
                else:
                    mask_arr = np.asarray(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
                mask_arr = _ade_mask_to_segformer_labels(mask_arr)

            if self.image_size is not None:
                h, w = self.image_size
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
                if mask_arr is not None:
                    mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)

            sample_id = f"ade20k_{self.split}_{i:05d}"
            self._samples.append(
                Sample(
                    sample_id=sample_id,
                    image=arr,
                    mask=mask_arr,
                    meta={"source": self.hf_id, "split": self.split, "index": i},
                )
            )
        logger.info("Loaded %d ADE20K samples", len(self._samples))
        if self.cache_dir:
            self.export_cache(self.cache_dir)

    def _load_from_cache(self) -> None:
        assert self.cache_dir is not None
        import json

        with open(self.cache_dir / "index.json", encoding="utf-8") as f:
            index = json.load(f)
        if self.max_samples is not None:
            index = index[: self.max_samples]
        for item in index:
            image = cv2.cvtColor(
                cv2.imread(str(self.cache_dir / item["image"]), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
            mask = None
            if item.get("mask"):
                mask = cv2.imread(str(self.cache_dir / item["mask"]), cv2.IMREAD_UNCHANGED)
                if mask is not None and mask.ndim == 3:
                    mask = mask[..., 0]
                if mask is not None:
                    mask = mask.astype(np.int32)
            self._samples.append(
                Sample(
                    sample_id=item["sample_id"],
                    image=image,
                    mask=mask,
                    meta=item.get("meta", {}),
                )
            )
        logger.info("Loaded %d samples from cache %s", len(self._samples), self.cache_dir)

    def export_cache(self, cache_dir: str | Path) -> None:
        cache_dir = ensure_dir(cache_dir)
        img_dir = ensure_dir(cache_dir / "images")
        mask_dir = ensure_dir(cache_dir / "masks")
        index: list[dict[str, Any]] = []
        for sample in self._samples:
            img_rel = f"images/{sample.sample_id}.png"
            write_image_rgb(cache_dir / img_rel, sample.image)
            mask_rel = None
            if sample.mask is not None:
                mask_rel = f"masks/{sample.sample_id}.png"
                write_mask(cache_dir / mask_rel, sample.mask)
            index.append(
                {
                    "sample_id": sample.sample_id,
                    "image": img_rel,
                    "mask": mask_rel,
                    "meta": sample.meta or {},
                }
            )
        save_json(cache_dir / "index.json", index)
        logger.info("Exported %d samples to %s", len(index), cache_dir)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]


def _ade_mask_to_segformer_labels(mask: np.ndarray) -> np.ndarray:
    """Map ADE20K annotation ids to SegFormer-ADE label space.

    ADE masks from SceneParse150 use ``0 = unlabeled`` and ``1..150 = classes``.
    ``nvidia/segformer-*-ade-512-512`` was trained with reduce-zero-label, so it
    predicts ``0..149``. We shift valid labels by -1 and mark original 0 as 255
    (ignore).
    """
    out = mask.astype(np.int32)
    unlabeled = out == 0
    out = out - 1
    out[unlabeled] = 255
    return out


def build_dataset(cfg: Config, use_synthetic: bool = False) -> VisionDataset:
    """Factory used by scripts. Prefer ADE20K; fall back to synthetic if requested."""
    if use_synthetic or cfg.get("dataset", "name") == "synthetic":
        n = cfg.get("dataset", "max_samples", default=8) or 8
        size = tuple(cfg.get("dataset", "image_size", default=[256, 256]))
        return SyntheticDataset(n=n, size=(size[0], size[1]), seed=cfg.seed)

    cache = cfg.get("dataset", "cache_dir", default="data/raw/ade20k")
    cache_path = Path(cache)
    if not cache_path.is_absolute():
        cache_path = cfg.root / cache_path
    size = cfg.get("dataset", "image_size", default=[512, 512])
    return ADE20KDataset(
        split=cfg.get("dataset", "split", default="validation"),
        max_samples=cfg.get("dataset", "max_samples", default=50),
        image_size=(size[0], size[1]) if size else None,
        cache_dir=cache_path,
        hf_id=cfg.get("dataset", "hf_id", default="scene_parse_150"),
        seed=cfg.seed,
    )
