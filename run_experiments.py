#!/usr/bin/env python3
"""
Top-level entrypoint for AI4MIA Assignment 4 Experiments.

Delegates to modular scripts in scripts/:
- scripts/train.py: Multi-experiment training engine
- scripts/evaluate.py: Quantitative evaluation engine
- scripts/generate_figures.py: Visualization & figure generation engine
- scripts/compile_report.py: LaTeX PDF compilation engine
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="AI4MIA Assignment 4 Experiment Runner")
    parser.add_argument(
        "--action",
        type=str,
        default="evaluate",
        choices=["train", "evaluate", "figures", "compile", "all"],
        help="Action to execute",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["resnet_6blocks", "unet_128", "monai_resunet", "monai_resunet_ssim", "resnet_6blocks_minmax", "all"],
        help="Target model preset",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Initial epochs")
    parser.add_argument("--epochs-decay", type=int, default=20, help="Decay epochs")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size")

    args = parser.parse_args()

    python_exe = sys.executable

    if args.action in ["train", "all"]:
        cmd = [python_exe, "scripts/train.py", "--model", args.model, "--epochs", str(args.epochs), "--epochs-decay", str(args.epochs_decay), "--batch-size", str(args.batch_size)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if args.action in ["evaluate", "all"]:
        cmd = [python_exe, "scripts/evaluate.py", "--model", args.model, "--batch-size", str(args.batch_size)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if args.action in ["figures", "all"]:
        cmd = [python_exe, "scripts/generate_figures.py"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if args.action in ["compile", "all"]:
        cmd = [python_exe, "scripts/compile_report.py"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
