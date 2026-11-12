# Multi-Parametric Brain MRI Generation Using Unpaired Cycle-Consistent Adversarial Networks and MONAI

[![Course](https://img.shields.io/badge/Course-AI4MIA%20(SKKU)-blue.svg)](https://www.skku.edu)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Package%20Manager](https://img.shields.io/badge/Package%20Manager-uv-blueviolet.svg)](https://astral.sh/uv)
[![Framework](https://img.shields.io/badge/Framework-MONAI%201.6.1-green.svg)](https://monai.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI for Medical Image Analysis (AI4MIA) — Assignment 4**  
> **Author:** SoonYong Shim (Student ID: `20267105171`, Email: `simsoonyong@yonsei.ac.kr`)  
> **Department:** Department of Artificial Intelligence, Sungkyunkwan University (SKKU)  
> **Report Deliverables:** [`20267105171_shimsoonyong_Report.pdf`](20267105171_shimsoonyong_Report.pdf) / [`StudentID_Name_Report.pdf`](StudentID_Name_Report.pdf)

---

## 1. Executive Summary

Cross-modality brain magnetic resonance imaging (MRI) synthesis enables multi-contrast diagnostic workflows without subjecting patients to extended scanner acquisition durations or repeated intravenous gadolinium contrast agent administrations. In this project, we implement an end-to-end unpaired image-to-image translation framework translating post-contrast T1-weighted (**T1ce**, Domain $X$) and T2-weighted (**T2**, Domain $Y$) brain scans using the BraTS 2021 glioblastoma benchmark.

Following course requirements, we:
1. Standardize medical data pipelines using the **Medical Open Network for AI (MONAI)** with outlier-robust percentile scaling and domain permutation.
2. Rigorously compare the two required generator architectures: **`resnet_6blocks`** (continuous bottleneck) vs. **`unet_128`** (recursive skip connections).
3. Investigate MONAI-native Residual U-Net architectures (**`monai_resunet`**) combined with structural cycle regularization (**`SSIMLoss`**).
4. Perform ablation studies exposing preprocessing-induced error sources (outlier-robust percentile clipping vs. naive min-max scaling).
5. Evaluate all models quantitatively and qualitatively on the designated held-out 4-subject test cohort (`00025`, `00026`, `00028`, `00030`, comprising 360 paired axial slices) using **PSNR**, **SSIM**, **RMSE**, **MAE**, and tissue sub-region decomposition.

```
       +-------------------------------------------------------------+
       |               BraTS 2021 Multi-Parametric MRI               |
       |     T1ce (Domain X: Vascular Hyper-permeability / Gad)     |
       |     T2w  (Domain Y: Vasogenic Edema / Fluid / FLAIR)        |
       +-------------------------------------------------------------+
                                      |
                         [MONAI Percentile Normalization]
                         [Unpaired Domain Shuffling]
                                      |
                   +------------------+------------------+
                   v                                     v
       +-----------------------+             +-----------------------+
       |   Generator G: X -> Y |             |   Generator F: Y -> X |
       |     (Synthetic T2w)   |             |     (Synthetic T1w)   |
       +-----------------------+             +-----------------------+
                   |                                     |
         [Dual PatchGAN D_Y]                   [Dual PatchGAN D_X]
                   |                                     |
       +-----------v-----------+             +-----------v-----------+
       |   Cyclic Recon: F(G)  |             |   Cyclic Recon: G(F)  |
       |  Cycle L1 + SSIM Loss |             |  Cycle L1 + SSIM Loss |
       +-----------------------+             +-----------------------+
```

---

## 2. Quantitative Benchmark Results

Evaluated across all **360 held-out axial test slices** from subjects `BraTS2021_00025`, `BraTS2021_00026`, `BraTS2021_00028`, and `BraTS2021_00030` (Mean $\pm$ Standard Deviation over normalized unit range $[0, 1]$):

| Model / Configuration | Parameters | Train Speed | PSNR (T2w) [dB] | SSIM (T2w) | PSNR (T1w) [dB] | SSIM (T1w) | RMSE (T2w) | MAE (T2w) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`resnet_6blocks`** (Proposed) | **7.83M** | 40.2 s/ep | $\mathbf{20.81 \pm 1.51}$ | $\mathbf{0.692 \pm 0.049}$ | $\mathbf{23.23 \pm 1.75}$ | $\mathbf{0.746 \pm 0.056}$ | **0.0925** | **0.0388** |
| `unet_128` | 41.82M | 52.8 s/ep | $18.84 \pm 2.55$ | $0.611 \pm 0.094$ | $21.92 \pm 1.60$ | $0.654 \pm 0.077$ | 0.1191 | 0.0537 |
| `monai_resunet` | 1.62M | 16.5 s/ep | $17.62 \pm 2.31$ | $0.552 \pm 0.097$ | $19.89 \pm 1.92$ | $0.572 \pm 0.093$ | 0.1362 | 0.0594 |
| `monai_resunet_ssim` | 1.62M | 17.1 s/ep | $17.60 \pm 2.34$ | $0.551 \pm 0.099$ | $19.38 \pm 1.82$ | $0.535 \pm 0.103$ | 0.1365 | 0.0601 |
| *Ablation: MinMax* (`resnet_6blocks_minmax`) | 7.83M | 40.2 s/ep | $20.15 \pm 1.82$ | $0.584 \pm 0.086$ | $25.59 \pm 1.63$ | $0.687 \pm 0.076$ | 0.1002 | 0.0441 |

### Key Experimental Takeaways
- **ResNet-6 Decisively Outperforms U-Net-128:** $+1.97\text{ dB}$ PSNR and $+0.081$ SSIM on T1ce $\to$ T2w translation ($p < 10^{-15}$); $+1.31\text{ dB}$ PSNR and $+0.092$ SSIM on T2w $\to$ T1ce ($p < 10^{-12}$).
- **Anatomical Stability:** U-Net-128 exhibits nearly $2.0\times$ larger SSIM variance across slices ($\pm 0.094$ vs $\pm 0.049$).
- **Bottleneck Representation vs. Skip-Connection Leakage:** In unpaired CycleGAN translation, low-level skip connections bypass the semantic bottleneck and cause source-modality contrast to "leak" into target images (e.g. muddy ventricles, false hyper-intense rings). The continuous $32\times 32$ spatial bottleneck in ResNet-6 forces deep semantic remapping without topological collapse.
- **Modality Asymmetry Error Decomposition:** BraTS multi-class segmentation reveals synthesis MAE is $\approx \mathbf{2.5\times}$ higher in the enhancing tumor core ($0.096 \pm 0.018$) than in normal parenchyma ($0.038 \pm 0.006$), confirming that tracer enhancement absence in T2 creates an ill-posed inverse problem.

---

## 3. Formal Repository Directory Structure

```
assignment4/
├── .gitignore                          # Clean tracking rules (ignores binaries, venv, build cruft)
├── .python-version                     # Python 3.12 pin
├── pyproject.toml                      # uv package dependencies and project metadata
├── uv.lock                             # Fully resolved dependency lockfile
├── README.md                           # Documentation and reproduction guide
│
├── 20267105171_shimsoonyong_Report.pdf  # Final PDF report deliverable (StudentID_Name_Report.pdf)
├── StudentID_Name_Report.pdf           # Assignment submission copy
│
├── run_experiments.py                  # Top-level unified execution CLI
│
├── src/                                # Core modular library
│   ├── __init__.py                     # Package initialization
│   ├── dataset.py                      # BraTS 2021 loader, MONAI preprocessing, unpaired permutation
│   ├── networks.py                     # ResNet, U-Net, MONAI ResUNet, PatchGAN Discriminator, ImagePool
│   ├── cyclegan.py                     # CycleGAN model engine, forward cycling, multi-loss optimization
│   ├── metrics.py                      # Mathematical metrics (PSNR, SSIM, MAE, RMSE) & evaluator
│   └── visualize.py                    # Visualization suite, tumor contour heatmaps, comparative plots
│
├── scripts/                            # Production CLI entrypoints
│   ├── train.py                        # Training engine (supports any preset, custom epochs, resumption)
│   ├── evaluate.py                     # Quantitative evaluation on 360-slice test cohort
│   ├── generate_figures.py             # Generates all publication figures & syntax-highlighted code captures
│   └── compile_report.py               # Compiles LaTeX report to PDF and verifies page budget
│
├── report/                             # LaTeX document workspace
│   ├── StudentID_Name_Report.tex       # Refined 5-page publication-grade LaTeX source
│   ├── StudentID_Name_Report.pdf       # Compiled 5-page PDF report
│   └── figures/                        # All rendered figures and syntax-highlighted code captures
│       ├── code_dataset.png            # Data handling & MONAI transforms
│       ├── code_resnet.png             # ResnetGenerator architecture
│       ├── code_unet.png               # UnetGenerator architecture
│       ├── code_training.png           # CycleGAN training scheme & losses
│       ├── code_metrics.png            # Explicit PSNR & 2D Gaussian SSIM
│       ├── fig_preprocessing.png       # Intensity histogram & percentile normalization analysis
│       ├── fig_training_curves.png     # Loss & validation PSNR/SSIM convergence dynamics
│       ├── fig_visual_comparison_3examples.png # 3 required examples comparison (ResNet-6 vs U-Net-128)
│       ├── fig_error_heatmaps.png      # Multi-model error heatmaps & BraTS tumor segmentations
│       └── fig_quantitative_analysis.png # PSNR/SSIM bars, region-wise MAE, slice-depth profile
│
└── results/                            # Full quantitative evaluation records & visual samples
    ├── all_experiments_summary.json    # Master summary across all models
    ├── resnet_6blocks/                 # Summary, slice metrics, and visual_samples.npz
    ├── unet_128/                       # Summary, slice metrics, and visual_samples.npz
    ├── monai_resunet/                  # Summary, slice metrics, and visual_samples.npz
    ├── monai_resunet_ssim/             # Summary, slice metrics, and visual_samples.npz
    └── resnet_6blocks_minmax/          # Summary, slice metrics, and visual_samples.npz
```

---

## 4. Setup and Quickstart Guide

### 4.1 Environment Installation with `uv`
This project uses [`uv`](https://astral.sh/uv) for fast, reproducible Python packaging:

```bash
# Clone and enter the workspace
cd /home/ttaaa/workspace/2026_2_SKKU/AI4MIA/assignment4

# Create virtual environment and synchronize dependencies from uv.lock
uv sync
```

### 4.2 Dataset Preparation
Ensure the BraTS 2021 dataset is located at `AI4MIA_Assignment4_Dataset/`:
```bash
# If compressed archive exists:
unzip -q AI4MIA_Assignment4_Dataset.zip -d .
```
The directory should contain subjects `BraTS2021_00000` through `BraTS2021_00030`, with `t1ce.nii.gz`, `t2.nii.gz`, and `seg.nii.gz`.

---

## 5. Reproduction Pipeline

All experiments, evaluations, figures, and reports can be executed through modular CLI scripts:

### Step 1: Model Training
Train any of the 5 model presets (or all):
```bash
# Train the proposed ResNet-6 model (40 epochs: 20 fixed lr + 20 linear decay)
uv run python scripts/train.py --model resnet_6blocks --epochs 20 --epochs-decay 20 --batch-size 12

# Train the baseline U-Net-128 model
uv run python scripts/train.py --model unet_128 --epochs 20 --epochs-decay 20 --batch-size 12

# Train the MONAI Residual U-Net with SSIM cycle regularization
uv run python scripts/train.py --model monai_resunet_ssim --epochs 20 --epochs-decay 20 --batch-size 12

# Or train all models sequentially
uv run python scripts/train.py --model all
```

### Step 2: Quantitative Test Evaluation
Evaluate trained checkpoints on the 360 held-out test slices:
```bash
# Evaluate ResNet-6
uv run python scripts/evaluate.py --model resnet_6blocks

# Evaluate U-Net-128
uv run python scripts/evaluate.py --model unet_128

# Evaluate all models and generate master summary
uv run python scripts/evaluate.py --model all
```

### Step 3: Publication Figure Generation
Render all syntax-highlighted code captures and publication figures:
```bash
uv run python scripts/generate_figures.py
```

### Step 4: PDF Report Compilation
Compile the LaTeX document to a 5-page publication-grade PDF and synchronize root submission deliverables:
```bash
uv run python scripts/compile_report.py
```

Or execute the complete evaluation-to-PDF pipeline with a single command:
```bash
uv run python run_experiments.py --action all
```

---

## 6. Submission Deliverables Verification

Conforming strictly to course requirements:
- **Captured images of code:** Network architectures, training scheme, domain permutation, and metrics are captured in `report/figures/code_*.png` and embedded in the report.
- **Generator Comparison:** Concrete architectural, mathematical, and empirical comparison between `resnet_6blocks` and `unet_128`.
- **3 Visual Examples:** Complete $[ \text{Real T1w}, \text{Real T2w}, \text{Synthetic T2w}, \text{Synthetic T1w} ]$ visual examples for both generators displayed side-by-side with quantitative annotations.
- **Quantitative Metrics:** Comprehensive PSNR, SSIM, RMSE, MAE, and region-wise error analysis reported.
- **Report File Names:** [`20267105171_shimsoonyong_Report.pdf`](20267105171_shimsoonyong_Report.pdf) and [`StudentID_Name_Report.pdf`](StudentID_Name_Report.pdf).
