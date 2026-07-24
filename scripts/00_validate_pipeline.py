#!/usr/bin/env python3
"""Validate experiment artifacts after one or more pipeline stages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_robustness.config import Config
from vision_robustness.pipeline.validate import validate_project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--stages",
        nargs="*",
        default=["clean", "distorted", "enhanced"],
        help="Stages whose metrics.csv must exist",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    report = validate_project(cfg, stages=args.stages)
    report.print()
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
