"""
Generate all figures and syntax-highlighted code captures for the Assignment 4 PDF Report.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.visualize import (
    plot_error_heatmaps_comparison,
    plot_preprocessing_analysis,
    plot_quantitative_analysis,
    plot_three_visual_examples,
    plot_training_curves,
    render_code_snippet,
)


def main() -> None:
    fig_dir = Path("report/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render concrete code screenshots (Slide 23 requirement)
    print("Rendering syntax-highlighted concrete code captures...")
    render_code_snippet(
        "src/dataset.py",
        175,
        217,
        "src/dataset.py: UnpairedBraTSDataset (Unpaired Domain Shuffling & MONAI Augmentations)",
        fig_dir / "code_dataset.png",
    )
    render_code_snippet(
        "src/networks.py",
        94,
        136,
        "src/networks.py: ResnetGenerator (resnet_6blocks: 2 Stride-2 Convs + 6 Residual Blocks)",
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

    # 2. Render preprocessing & normalization figure
    print("Plotting preprocessing and intensity normalization analysis...")
    plot_preprocessing_analysis("AI4MIA_Assignment4_Dataset", fig_dir / "fig_preprocessing.png")

    # 3. Render experimental results figures if training summary exists
    summary_path = Path("results/all_experiments_summary.json")
    if summary_path.exists():
        summaries = json.loads(summary_path.read_text())
        print("Plotting training and validation convergence curves...")
        plot_training_curves(summaries, fig_dir / "fig_training_curves.png")

        sample_keys = [
            "BraTS2021_00025_z075",
            "BraTS2021_00026_z075",
            "BraTS2021_00030_z080",
        ]
        # Verify exact sample keys from npz
        npz = Path("results/resnet_6blocks/visual_samples.npz")
        if npz.exists():
            import numpy as np
            available = sorted(list(np.load(npz, allow_pickle=True).keys()))
            print("Available visual sample keys:", available)
            picked = [
                "BraTS2021_00025_z080",
                "BraTS2021_00026_z075",
                "BraTS2021_00030_z080",
            ]
            picked = [k for k in picked if k in available][:3]

            print("Plotting 3 visual examples for ResNet-6, U-Net-128, and MONAI ResUNet+SSIM...")
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
            if "monai_resunet_ssim" in summaries:
                plot_three_visual_examples(
                    "monai_resunet_ssim",
                    "Technique Attempt: MONAI ResUNet + SSIM Cycle Loss",
                    picked,
                    fig_dir / "fig_visual_3examples_monai.png",
                )

            print("Plotting multi-model error heatmaps and tumor contour analysis...")
            plot_error_heatmaps_comparison(picked[0], fig_dir / "fig_error_heatmaps.png")

        print("Plotting quantitative bar charts and anatomical error source analysis...")
        plot_quantitative_analysis(summaries, fig_dir / "fig_quantitative_analysis.png")

    print("All figures generated in report/figures/!")


if __name__ == "__main__":
    main()
