#!/usr/bin/env python3
"""Download / cache the ADE20K (SceneParse150) dataset subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_robustness.config import Config
from vision_robustness.data import ADE20KDataset, SyntheticDataset, visualize_samples
from vision_robustness.utils.logging import get_logger

logger = get_logger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create a synthetic dataset instead of downloading ADE20K",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.ensure_dirs()

    if args.synthetic:
        n = args.max_samples or cfg.get("dataset", "max_samples", default=16) or 16
        size = tuple(cfg.get("dataset", "image_size", default=[256, 256]))
        ds = SyntheticDataset(n=n, size=(size[0], size[1]), seed=cfg.seed)
        cache = cfg.path("raw_dir") / "synthetic"
        # Export via ADE20K-compatible layout
        from vision_robustness.data.dataset import Sample
        from vision_robustness.utils.io import ensure_dir, save_json, write_image_rgb, write_mask

        cache = ensure_dir(cache)
        index = []
        for sample in ds:
            assert isinstance(sample, Sample)
            img_rel = f"images/{sample.sample_id}.png"
            write_image_rgb(cache / img_rel, sample.image)
            mask_rel = f"masks/{sample.sample_id}.png"
            write_mask(cache / mask_rel, sample.mask)
            index.append(
                {
                    "sample_id": sample.sample_id,
                    "image": img_rel,
                    "mask": mask_rel,
                    "meta": sample.meta,
                }
            )
        save_json(cache / "index.json", index)
        logger.info("Wrote synthetic dataset to %s (%d samples)", cache, len(index))
        visualize_samples(list(ds), cfg.path("figures_dir") / "eda_synthetic.png")
        return

    max_samples = args.max_samples or cfg.get("dataset", "max_samples", default=50)
    cache = cfg.get("dataset", "cache_dir", default="data/raw/ade20k")
    cache_path = Path(cache)
    if not cache_path.is_absolute():
        cache_path = cfg.root / cache_path
    size = cfg.get("dataset", "image_size", default=[512, 512])
    ds = ADE20KDataset(
        split=cfg.get("dataset", "split", default="validation"),
        max_samples=max_samples,
        image_size=(size[0], size[1]),
        cache_dir=cache_path,
        hf_id=cfg.get("dataset", "hf_id", default="scene_parse_150"),
        seed=cfg.seed,
    )
    visualize_samples(list(ds)[:8], cfg.path("figures_dir") / "eda_ade20k.png")
    logger.info("Dataset ready: %d samples cached at %s", len(ds), cache_path)


if __name__ == "__main__":
    main()
