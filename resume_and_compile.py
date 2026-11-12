"""
Master Experiment Runner with Resumption and Automated PDF Report Compilation.

Features:
1. Automatically resumes unet_128 from epoch 95 and trains to epoch 200.
2. Trains MONAI native ResUNet (monai_resunet) and MONAI ResUNet with SSIM cycle regularization (monai_resunet_ssim).
3. Trains Min-Max preprocessing ablation (resnet_6blocks_minmax).
4. Skips already completed experiments (resnet_6blocks).
5. After each experiment finishes, immediately regenerates all figures and compiles StudentID_Name_Report.pdf.
6. Commits all progress and outputs to local Git repository.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from run_experiments import run_single_experiment
from src.cyclegan import CycleGANConfig
from train_overnight import git_commit, update_latex_with_results

RESUME_EXPERIMENTS = [
    (
        "resnet_6blocks",
        CycleGANConfig(
            netG="resnet_6blocks",
            n_epochs=75,
            n_epochs_decay=75,
            lambda_ssim=0.0,
        ),
        "percentile",
    ),
    (
        "unet_128",
        CycleGANConfig(
            netG="unet_128",
            n_epochs=100,
            n_epochs_decay=100,
            lambda_ssim=0.0,
        ),
        "percentile",
    ),
    (
        "monai_resunet",
        CycleGANConfig(
            netG="monai_resunet",
            n_epochs=50,
            n_epochs_decay=50,
            lambda_ssim=0.0,
        ),
        "percentile",
    ),
    (
        "monai_resunet_ssim",
        CycleGANConfig(
            netG="monai_resunet",
            n_epochs=50,
            n_epochs_decay=50,
            lambda_ssim=2.0,
        ),
        "percentile",
    ),
    (
        "resnet_6blocks_minmax",
        CycleGANConfig(
            netG="resnet_6blocks",
            n_epochs=15,
            n_epochs_decay=15,
            lambda_ssim=0.0,
        ),
        "minmax",
    ),
]


def compile_pdf_report() -> None:
    """Generate figures, update LaTeX with latest metrics, and compile PDF."""
    print("\n--- Running automated figure generator ---")
    subprocess.run([sys.executable, "generate_report_figs.py"], check=False)

    tex_path = Path("report/StudentID_Name_Report.tex")
    summary_path = Path("results/all_experiments_summary.json")
    update_latex_with_results(summary_path, tex_path)

    print("--- Compiling StudentID_Name_Report.tex with pdflatex ---")
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory=report",
        str(tex_path),
    ]
    subprocess.run(cmd, check=False)
    subprocess.run(cmd, check=False)

    pdf_src = Path("report/StudentID_Name_Report.pdf")
    if pdf_src.exists():
        dest_pdf = Path("StudentID_Name_Report.pdf")
        shutil.copy(pdf_src, dest_pdf)
        print(f"SUCCESS: StudentID_Name_Report.pdf updated ({dest_pdf.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    overall_start = time.time()
    print("=" * 75)
    print("RESUMING EXPERIMENT SUITE & COMPILATION PIPELINE")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    all_summaries: Dict[str, Any] = {}
    master_summary_path = Path("results/all_experiments_summary.json")
    if master_summary_path.exists():
        try:
            with open(master_summary_path, "r") as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = {}

    for exp_name, cfg, norm_mode in RESUME_EXPERIMENTS:
        exp_summary_file = Path("results") / exp_name / "summary.json"
        # If already finished full training and test evaluation, skip
        if exp_summary_file.exists():
            try:
                loaded_s = json.loads(exp_summary_file.read_text())
                if "test_overall" in loaded_s and loaded_s.get("test_overall", {}).get("psnr_T2_mean") is not None:
                    print(f"\n[Experiment {exp_name} already completed] Skipping training.")
                    all_summaries[exp_name] = loaded_s
                    continue
            except Exception:
                pass

        print(f"\n>>> Running / Resuming Experiment: {exp_name} at {time.strftime('%H:%M:%S')} <<<")
        exp_start = time.time()
        summary = run_single_experiment(exp_name, cfg, norm_mode=norm_mode, batch_size=12, seed=42)
        all_summaries[exp_name] = summary
        exp_elapsed = time.time() - exp_start
        print(f">>> Finished Experiment: {exp_name} in {exp_elapsed/60:.1f} min <<<")

        with open(master_summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

        # Incrementally compile report after each experiment
        compile_pdf_report()
        git_commit(f"feat: completed {exp_name} and updated report PDF")

    # Final compilation and commit
    compile_pdf_report()
    git_commit("feat: complete all experiments and compile final StudentID_Name_Report.pdf")
    print("\n" + "=" * 75)
    print(f"ALL EXPERIMENTS AND FINAL REPORT COMPILED in {(time.time() - overall_start)/3600:.2f} hours!")
    print("=" * 75)


if __name__ == "__main__":
    main()
