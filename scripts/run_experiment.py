#!/usr/bin/env python3
"""Orchestrate the full ADE20K robustness experiment in stage order.

This is the single entrypoint for a complete run. Individual scripts under
scripts/01–10 remain available for incremental work and partner parallelization.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = [
    ("download", ["scripts/01_download_data.py"]),
    ("eda", ["scripts/02_eda_visualize.py"]),
    ("clean", ["scripts/03_run_clean_baseline.py"]),
    # Evaluation regenerates distortions/enhancements in-memory (avoids writing 10k+ PNGs).
    ("distort", ["scripts/05_evaluate_distorted.py"]),
    ("enhance", ["scripts/07_evaluate_enhanced.py"]),
    ("finetune", ["scripts/08_finetune_yolo.py"]),
    ("finetune_eval", ["scripts/09_evaluate_finetuned.py"]),
    ("report", ["scripts/10_generate_report_figures.py"]),
    ("validate", ["scripts/00_validate_pipeline.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--until",
        choices=[s[0] for s in STAGES],
        default="validate",
        help="Stop after this stage (inclusive)",
    )
    parser.add_argument(
        "--from-stage",
        dest="from_stage",
        choices=[s[0] for s in STAGES],
        default="download",
        help="Start from this stage",
    )
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    names = [s[0] for s in STAGES]
    start = names.index(args.from_stage)
    stop = names.index(args.until)

    for name, scripts in STAGES[start : stop + 1]:
        if args.skip_finetune and name in {"finetune", "finetune_eval"}:
            print(f"[skip] {name}")
            continue
        for script in scripts:
            cmd = [sys.executable, str(ROOT / script), "--config", args.config]
            if args.synthetic and name not in {"validate", "report"}:
                # validate/report don't always take --synthetic; download/eda/eval do
                if name != "validate":
                    cmd.append("--synthetic")
            if name == "validate":
                cmd = [
                    sys.executable,
                    str(ROOT / script),
                    "--config",
                    args.config,
                    "--stages",
                    "clean",
                    "distorted",
                    "enhanced",
                ]
            print("\n==>", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nExperiment orchestration finished.")


if __name__ == "__main__":
    main()
