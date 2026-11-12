#!/usr/bin/env python3
"""
Figure and Code Capture Generation CLI for AI4MIA Assignment 4 Report.

Generates:
1. Syntax-highlighted concrete code captures:
   - code_dataset.png: Data handling, domain permutation, and MONAI augmentation
   - code_resnet.png: ResnetGenerator (resnet_6blocks)
   - code_unet.png: UnetGenerator (unet_128)
   - code_training.png: CycleGAN training scheme, forward cycling, and multi-loss optimization
   - code_metrics.png: Explicit PSNR and 2D Gaussian-filtered SSIM metrics
2. Publication-quality evaluation and analysis figures:
   - fig_preprocessing.png: Raw intensity histograms vs. naive min-max vs. MONAI percentile normalization
   - fig_training_curves.png: Epoch convergence dynamics (Cycle Loss, LSGAN, Validation PSNR/SSIM)
   - fig_visual_comparison_3examples.png: Side-by-side 3-example comparison (ResNet-6 vs U-Net-128)
   - fig_error_heatmaps.png: Absolute error heatmaps with BraTS tumor segmentations (Edema & Core)
   - fig_quantitative_analysis.png: PSNR/SSIM bar charts, region-wise MAE, and slice-depth profile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualize import (
    plot_error_heatmaps_comparison,
    plot_preprocessing_analysis,
    plot_quantitative_analysis,
    plot_three_visual_examples,
    plot_three_visual_examples_comparison,
    plot_training_curves,
    render_code_snippet,
)


def generate_all_figures(
    fig_dir: Path = Path("report/figures"),
    data_root: str = "AI4MIA_Assignment4_Dataset",
    results_dir: str = "results",
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Generating Figures and Code Captures in: {fig_dir}")
    print(f"{'='*70}\n")

    # 1. Render syntax-highlighted concrete code captures
    print("1/5 Rendering syntax-highlighted concrete code captures...")
    render_code_snippet(
        "src/dataset.py",
        175,
        217,
        "src/dataset.py: UnpairedBraTSDataset (Domain Permutation & MONAI Augmentation)",
        fig_dir / "code_dataset.png",
    )
    render_code_snippet(
        "src/networks.py",
        94,
        136,
        "src/networks.py: ResnetGenerator (resnet_6blocks: 2 Downsampling Convs + 6 Residual Blocks)",
        fig_dir / "code_resnet.png",
    )
    render_code_snippet(
        "src/networks.py",
        234,
        276,
        "src/networks.py: UnetGenerator (unet_128: 7-Level Recursive Skip Connections)",
        fig_dir / "code_unet.png",
    )
    render_code_snippet(
        "src/cyclegan.py",
        114,
        155,
        "src/cyclegan.py: CycleGAN Training Scheme (LSGAN, Cycle L1, Identity & SSIM Loss)",
        fig_dir / "code_training.png",
    )
    render_code_snippet(
        "src/metrics.py",
        29,
        72,
        "src/metrics.py: Closed-Form PSNR & Gaussian-Windowed 2D SSIM Metrics",
        fig_dir / "code_metrics.png",
    )

    # 2. Render preprocessing analysis
    print("2/5 Plotting preprocessing and intensity normalization analysis...")
    plot_preprocessing_analysis(data_root, fig_dir / "fig_preprocessing.png")

    # 3. Render training curves and results
    summary_path = Path(results_dir) / "all_experiments_summary.json"
    if summary_path.exists():
        summaries = json.loads(summary_path.read_text())

        print("3/5 Plotting training and validation convergence curves...")
        plot_training_curves(summaries, fig_dir / "fig_training_curves.png")

        # 4. Render visual comparison of 3 required examples
        print("4/5 Plotting 3 visual examples and side-by-side generator comparison...")
        npz = Path(results_dir) / "resnet_6blocks" / "visual_samples.npz"
        if npz.exists():
            import numpy as np

            available = sorted(list(np.load(npz, allow_pickle=True).keys()))
            picked = [
                "BraTS2021_00025_z080",
                "BraTS2021_00026_z075",
                "BraTS2021_00030_z080",
            ]
            picked = [k for k in picked if k in available][:3]

            plot_three_visual_examples(
                "resnet_6blocks",
                "Generator 1: resnet_6blocks (Default Settings)",
                picked,
                fig_dir / "fig_visual_3examples_resnet6.png",
            )
            plot_three_visual_examples(
                "unet_128",
                "Generator 2: unet_128 (Default Settings)",
                picked,
                fig_dir / "fig_visual_3examples_unet128.png",
            )
            plot_three_visual_examples_comparison(
                picked,
                fig_dir / "fig_visual_comparison_3examples.png",
            )

            # Error heatmaps with tumor contours
            plot_error_heatmaps_comparison(picked[0], fig_dir / "fig_error_heatmaps.png")

        # 5. Render quantitative bar charts and depth profiles
        print("5/5 Plotting quantitative metrics and anatomical error decomposition...")
        plot_quantitative_analysis(summaries, fig_dir / "fig_quantitative_analysis.png")

    print("\nAll publication figures and code captures successfully generated!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures for report")
    parser.add_argument("--fig-dir", type=str, default="report/figures", help="Output directory for figures")
    parser.add_argument("--data-root", type=str, default="AI4MIA_Assignment4_Dataset", help="BraTS dataset root")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory with experiment summaries")
    args = parser.parse_args()

    generate_all_figures(
        fig_dir=Path(args.fig_dir),
        data_root=args.data_root,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
