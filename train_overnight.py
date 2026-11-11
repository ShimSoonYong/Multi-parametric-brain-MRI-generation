"""
Overnight 8-Hour Experiment Runner & Autonomous Report Pipeline.

Total Training Time Calibrated for ~7.5 - 8.0 Hours on NVIDIA RTX 5070 GPU:
1. resnet_6blocks (Default CycleGAN Baseline, Percentile Norm):
   - 150 epochs (75 constant + 75 decay) @ ~72.5s/epoch ~= 3.02 hours
2. unet_128 (Default CycleGAN Baseline, Percentile Norm):
   - 200 epochs (100 constant + 100 decay) @ ~20.5s/epoch ~= 1.14 hours
3. monai_resunet (MONAI Residual UNet Generator, Percentile Norm):
   - 200 epochs (100 constant + 100 decay) @ ~19.8s/epoch ~= 1.10 hours
4. monai_resunet_ssim (MONAI ResUNet + MONAI SSIMLoss Regularization):
   - 200 epochs (100 constant + 100 decay) @ ~21.5s/epoch ~= 1.19 hours
5. resnet_6blocks_minmax (Ablation on Preprocessing Outlier Sensitivity):
   - 60 epochs (30 constant + 30 decay) @ ~72.5s/epoch ~= 1.21 hours

Total Estimated Execution: ~7 hours 45 mins.
Immediately upon completion, automatically executes:
- High-resolution figure & syntax-highlighted code capture generation
- Exact quantitative metrics formatting into LaTeX source
- LaTeX PDF compilation into StudentID_Name_Report.pdf in the assignment4 root directory
- Local git version control commit
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from run_experiments import run_single_experiment
from src.cyclegan import CycleGANConfig


EXPERIMENTS = [
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
            n_epochs=100,
            n_epochs_decay=100,
            lambda_ssim=0.0,
        ),
        "percentile",
    ),
    (
        "monai_resunet_ssim",
        CycleGANConfig(
            netG="monai_resunet",
            n_epochs=100,
            n_epochs_decay=100,
            lambda_ssim=2.0,
        ),
        "percentile",
    ),
    (
        "resnet_6blocks_minmax",
        CycleGANConfig(
            netG="resnet_6blocks",
            n_epochs=30,
            n_epochs_decay=30,
            lambda_ssim=0.0,
        ),
        "minmax",
    ),
]


def git_commit(message: str) -> None:
    try:
        subprocess.run(["git", "add", "src/", "report/", "*.py", "*.json"], check=False)
        subprocess.run(["git", "commit", "-m", message], check=False)
    except Exception as e:
        print(f"Git commit warning: {e}")


def update_latex_with_results(summary_path: Path, tex_path: Path) -> None:
    """Inject actual measured test set metrics into LaTeX table."""
    if not summary_path.exists() or not tex_path.exists():
        return

    summaries = json.loads(summary_path.read_text())
    tex_content = tex_path.read_text()

    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model / Setting} & \textbf{PSNR (T2w)} & \textbf{SSIM (T2w)} & \textbf{PSNR (T1w)} & \textbf{SSIM (T1w)} \\")
    lines.append(r"\midrule")

    model_display = [
        ("resnet_6blocks", r"\texttt{resnet\_6blocks}"),
        ("unet_128", r"\texttt{unet\_128}"),
        ("monai_resunet", r"\texttt{monai\_resunet}"),
        ("monai_resunet_ssim", r"\texttt{monai\_resunet\_ssim}"),
        ("resnet_6blocks_minmax", r"\textit{Ablation (MinMax)}"),
    ]

    for key, display_name in model_display:
        if key in summaries and "test_overall" in summaries[key]:
            ov = summaries[key]["test_overall"]
            p_t2_m, p_t2_s = ov.get("psnr_T2_mean", 0.0), ov.get("psnr_T2_std", 0.0)
            s_t2_m, s_t2_s = ov.get("ssim_T2_mean", 0.0), ov.get("ssim_T2_std", 0.0)
            p_t1_m, p_t1_s = ov.get("psnr_T1_mean", 0.0), ov.get("psnr_T1_std", 0.0)
            s_t1_m, s_t1_s = ov.get("ssim_T1_mean", 0.0), ov.get("ssim_T1_std", 0.0)
            lines.append(
                f"{display_name} & ${p_t2_m:.2f} \\pm {p_t2_s:.2f}$ & ${s_t2_m:.3f} \\pm {s_t2_s:.3f}$ & "
                f"${p_t1_m:.2f} \\pm {p_t1_s:.2f}$ & ${s_t1_m:.3f} \\pm {s_t1_s:.3f}$ \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    new_table_body = "\n".join(lines)

    # Replace table body in LaTeX
    pattern = re.compile(r"\\begin\{tabular\}\{lcccc\}.*?\\end\{tabular\}", re.DOTALL)
    if pattern.search(tex_content):
        updated_tex = pattern.sub(new_table_body, tex_content)
        tex_path.write_text(updated_tex)
        print("Updated LaTeX table with real quantitative test metrics.")


def main() -> None:
    overall_start = time.time()
    print("=" * 75)
    print("STARTING OVERNIGHT 8-HOUR EXPERIMENT SUITE FOR AI4MIA ASSIGNMENT 4")
    print("Target Duration: ~7.5 - 8.0 hours on NVIDIA GeForce RTX 5070 GPU")
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

    for exp_name, cfg, norm_mode in EXPERIMENTS:
        exp_summary_file = Path("results") / exp_name / "summary.json"
        if exp_summary_file.exists():
            print(f"\n[Found completed experiment {exp_name}] Loading cached results...")
            with open(exp_summary_file, "r") as f:
                all_summaries[exp_name] = json.load(f)
            continue

        print(f"\n>>> Starting Experiment: {exp_name} at {time.strftime('%H:%M:%S')} <<<")
        exp_start = time.time()
        summary = run_single_experiment(exp_name, cfg, norm_mode=norm_mode, batch_size=12, seed=42)
        all_summaries[exp_name] = summary
        exp_elapsed = time.time() - exp_start
        print(f">>> Finished Experiment: {exp_name} in {exp_elapsed/3600:.2f} hours <<<")

        with open(master_summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

        git_commit(f"feat: completed experiment {exp_name} ({exp_elapsed/60:.1f} min)")

    total_training_time = time.time() - overall_start
    print("\n" + "=" * 75)
    print(f"ALL TRAINING EXPERIMENTS COMPLETED in {total_training_time/3600:.2f} hours!")
    print("=" * 75)

    # 1. Generate Figures and Code Captures
    print("\nGenerating report figures and code captures...")
    subprocess.run([sys.executable, "generate_report_figs.py"], check=True)

    # 2. Update LaTeX with actual results
    tex_path = Path("report/StudentID_Name_Report.tex")
    update_latex_with_results(master_summary_path, tex_path)

    # 3. Compile LaTeX Report into PDF
    print("\nCompiling LaTeX Report to PDF...")
    if tex_path.exists():
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory=report",
            str(tex_path),
        ]
        # Run twice for cross-references & TOC
        subprocess.run(cmd, check=False)
        subprocess.run(cmd, check=False)

        pdf_src = Path("report/StudentID_Name_Report.pdf")
        if pdf_src.exists():
            dest_pdf = Path("StudentID_Name_Report.pdf")
            shutil.copy(pdf_src, dest_pdf)
            print(f"\nSUCCESS: Report generated at {dest_pdf.resolve()} ({dest_pdf.stat().st_size / 1024:.1f} KB)")

    # 4. Final Git Commit
    git_commit("feat: complete overnight experiments, generate figures, and compile report PDF")
    print(f"\nPipeline finished at {time.strftime('%Y-%m-%d %H:%M:%S')}. Total elapsed: {(time.time() - overall_start)/3600:.2f} hours.")


if __name__ == "__main__":
    main()
