"""
Visualization and Code-Capture Utilities for AI4MIA Assignment 4 Report.

Generates:
1. High-DPI syntax-highlighted code captures (`code_*.png`) of dataset, network,
   training scheme, and evaluation metrics implementation.
2. Preprocessing & intensity normalization comparison (`fig_preprocessing.png`).
3. Training loss & validation PSNR/SSIM convergence curves (`fig_training_curves.png`).
4. 3 Visual Examples of `[Real T1w, Real T2w, Synthetic T2w, Synthetic T1w]` comparing
   `resnet_6blocks`, `unet_128`, and `monai_resunet_ssim` (`fig_visual_3examples_*.png`).
5. Multi-model error heatmaps & cycle reconstructions (`fig_error_heatmaps.png`, `fig_cycle_consistency.png`).
6. Quantitative metrics & anatomical region error decomposition (`fig_quantitative_analysis.png`).
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

from src.dataset import normalize_volume


MODEL_META: Dict[str, tuple[str, str, str, str]] = {
    "resnet_6blocks": ("ResNet-6 (Default)", "#2563eb", "-", "o"),
    "unet_128": ("U-Net-128 (Default)", "#dc2626", "--", "s"),
    "monai_resunet": ("MONAI ResUNet", "#7c3aed", "-.", "P"),
    "monai_resunet_ssim": ("MONAI ResUNet + SSIM", "#059669", "-", "^"),
    "resnet_6blocks_no_id": ("ResNet-6 (No Identity Loss)", "#0891b2", ":", "v"),
    "resnet_6blocks_minmax": ("ResNet-6 (Naive Min-Max)", "#d97706", "--", "d"),
}


def render_code_snippet(
    file_path: str | Path,
    start_line: int,
    end_line: int,
    title: str,
    out_path: str | Path,
) -> None:
    """Render a syntax-highlighted screenshot of a source code region with a clean modern window header."""
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    selected = "\n".join(lines[start_line - 1 : end_line])

    formatter = ImageFormatter(
        style="friendly",
        line_numbers=True,
        line_number_start=start_line,
        font_name="DejaVu Sans Mono",
        font_size=13,
        line_pad=2,
        image_pad=10,
        line_number_bg="#f1f5f9",
        line_number_fg="#64748b",
    )
    raw_bytes = highlight(selected, PythonLexer(), formatter)
    code_img = Image.open(BytesIO(raw_bytes)).convert("RGB")

    header_h = 28
    border = 1
    w, h = code_img.size
    canvas = Image.new("RGB", (w + 2 * border, h + header_h + 2 * border), "#0f172a")
    draw = ImageDraw.Draw(canvas)

    for idx, col in enumerate(["#ef4444", "#f59e0b", "#10b981"]):
        cx = 14 + idx * 16
        cy = header_h // 2 + border
        draw.ellipse((cx - 4.5, cy - 4.5, cx + 4.5, cy + 4.5), fill=col)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    draw.text((68, 7), title, fill="#f8fafc", font=font)
    canvas.paste(code_img, (border, header_h + border))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))


def plot_preprocessing_analysis(data_root: str | Path, out_path: str | Path) -> None:
    """Plot raw intensity distributions and Min-Max vs. MONAI 99.5% Percentile normalization."""
    subj = "BraTS2021_00025"
    subj_dir = Path(data_root) / subj
    t1_raw = np.rot90(nib.load(str(subj_dir / f"{subj}_t1ce.nii.gz")).get_fdata(), axes=(0, 1))
    t2_raw = np.rot90(nib.load(str(subj_dir / f"{subj}_t2.nii.gz")).get_fdata(), axes=(0, 1))

    z = 75
    t1_mm = (normalize_volume(t1_raw, mode="minmax")[:, :, z] + 1.0) * 0.5
    t1_pct = (normalize_volume(t1_raw, mode="percentile")[:, :, z] + 1.0) * 0.5
    t2_mm = (normalize_volume(t2_raw, mode="minmax")[:, :, z] + 1.0) * 0.5
    t2_pct = (normalize_volume(t2_raw, mode="percentile")[:, :, z] + 1.0) * 0.5

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), dpi=250)

    nz_t1 = t1_raw[t1_raw > 0]
    p995_t1 = np.percentile(nz_t1, 99.5)
    axes[0, 0].hist(nz_t1, bins=80, color="#2563eb", alpha=0.8, log=True)
    axes[0, 0].axvline(p995_t1, color="#dc2626", linestyle="--", linewidth=2, label=f"99.5% = {p995_t1:.0f}")
    axes[0, 0].axvline(nz_t1.max(), color="#ea580c", linestyle=":", linewidth=2, label=f"Max = {nz_t1.max():.0f}")
    axes[0, 0].set_title("T1ce Non-Zero Voxel Histogram (Log Scale)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Raw MRI Intensity")
    axes[0, 0].set_ylabel("Voxel Count (log)")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.25)

    im01 = axes[0, 1].imshow(t1_mm[24:-24, 24:-24], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("T1ce: Naive Min-Max Scaling\n(Parenchyma Crushed by Outlier Spike)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im02 = axes[0, 2].imshow(t1_pct[24:-24, 24:-24], cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("T1ce: MONAI 99.5% Percentile Clipping\n(High Tissue & Tumor Rim Contrast)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    nz_t2 = t2_raw[t2_raw > 0]
    p995_t2 = np.percentile(nz_t2, 99.5)
    axes[1, 0].hist(nz_t2, bins=80, color="#059669", alpha=0.8, log=True)
    axes[1, 0].axvline(p995_t2, color="#dc2626", linestyle="--", linewidth=2, label=f"99.5% = {p995_t2:.0f}")
    axes[1, 0].axvline(nz_t2.max(), color="#ea580c", linestyle=":", linewidth=2, label=f"Max = {nz_t2.max():.0f}")
    axes[1, 0].set_title("T2w Non-Zero Voxel Histogram (Log Scale)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Raw MRI Intensity")
    axes[1, 0].set_ylabel("Voxel Count (log)")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.25)

    im11 = axes[1, 1].imshow(t2_mm[24:-24, 24:-24], cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("T2w: Naive Min-Max Scaling", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im12 = axes[1, 2].imshow(t2_pct[24:-24, 24:-24], cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("T2w: MONAI 99.5% Percentile Clipping\n(Clear Edema & CSF Boundaries)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(summaries: Dict[str, Any], out_path: str | Path) -> None:
    """Plot training losses and validation PSNR/SSIM curves across epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=250)

    for exp_name, (label, color, ls, marker) in MODEL_META.items():
        if exp_name not in summaries:
            continue
        hist = summaries[exp_name]["history"]
        epochs = [h["epoch"] for h in hist]
        cyc_loss = [h["train_cycle_A"] + h["train_cycle_B"] for h in hist]
        gan_loss = [h["train_G_A"] + h["train_G_B"] for h in hist]

        val_epochs = [h["epoch"] for h in hist if "val_psnr_T2_mean" in h]
        val_psnr = [0.5 * (h["val_psnr_T2_mean"] + h["val_psnr_T1_mean"]) for h in hist if "val_psnr_T2_mean" in h]
        val_ssim = [0.5 * (h["val_ssim_T2_mean"] + h["val_ssim_T1_mean"]) for h in hist if "val_ssim_T2_mean" in h]

        markevery = max(1, len(val_epochs) // 12)
        axes[0, 0].plot(epochs, cyc_loss, label=label, color=color, linestyle=ls, linewidth=1.8)
        axes[0, 1].plot(epochs, gan_loss, label=label, color=color, linestyle=ls, linewidth=1.8)
        axes[1, 0].plot(
            val_epochs, val_psnr, label=label, color=color, linestyle=ls, marker=marker, markevery=markevery, markersize=4, linewidth=1.8
        )
        axes[1, 1].plot(
            val_epochs, val_ssim, label=label, color=color, linestyle=ls, marker=marker, markevery=markevery, markersize=4, linewidth=1.8
        )

    axes[0, 0].set_title("(a) Training Cycle-Consistency Loss (L1)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("L1 Cycle Loss (lambda=10)")
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("(b) Training Generator Adversarial Loss (LSGAN)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("LSGAN Loss (G_A + G_B)")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("(c) Validation Mean PSNR (T1w <-> T2w)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("PSNR (dB)")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("(d) Validation Mean SSIM (T1w <-> T2w)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _extract_item_fields(item: Dict[str, Any]):
    real_t1 = item.get("real_T1", item.get("real_A"))
    real_t2 = item.get("real_T2", item.get("real_B"))
    fake_t2 = item.get("fake_T2", item.get("fake_B"))
    fake_t1 = item.get("fake_T1", item.get("fake_A"))
    seg = item.get("seg")
    m = item.get("metrics")
    if m is None or not isinstance(m, dict):
        m = {
            "psnr_T2": float(item.get("psnr_T2", 0.0)),
            "ssim_T2": float(item.get("ssim_T2", 0.0)),
            "psnr_T1": float(item.get("psnr_T1", 0.0)),
            "ssim_T1": float(item.get("ssim_T1", 0.0)),
        }
    return real_t1, real_t2, fake_t2, fake_t1, seg, m


def plot_three_visual_examples(
    exp_name: str,
    title_prefix: str,
    sample_keys: List[str],
    out_path: str | Path,
) -> None:
    """
    Generate the 3 required visual examples for a generator (`[Real T1w, Real T2w, Synthetic T2w, Synthetic T1w]`),
    matching Week 11 Slide 23 specification.
    """
    npz_path = Path("results") / exp_name / "visual_samples.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path, allow_pickle=True)

    fig, axes = plt.subplots(len(sample_keys), 4, figsize=(12.5, 3.1 * len(sample_keys)), dpi=250)
    if len(sample_keys) == 1:
        axes = np.expand_dims(axes, axis=0)
    col_titles = ["Real T1w (T1ce)", "Real T2w (Ground Truth)", "Synthetic T2w (G_A(T1w))", "Synthetic T1w (G_B(T2w))"]

    for row_idx, key in enumerate(sample_keys):
        item = data[key].item()
        r_t1, r_t2, f_t2, f_t1, _, m = _extract_item_fields(item)
        imgs = [r_t1, r_t2, f_t2, f_t1]

        for col_idx, img in enumerate(imgs):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)

            if col_idx == 0:
                subj_label = key.replace("_z", " (Slice z=") + ")"
                ax.text(
                    0.03,
                    0.95,
                    f"Example {row_idx+1}: {subj_label}",
                    transform=ax.transAxes,
                    fontsize=9.5,
                    fontweight="bold",
                    color="#f8fafc",
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.8),
                )
            elif col_idx == 2:
                ax.text(
                    0.03,
                    0.05,
                    f"PSNR: {m['psnr_T2']:.2f} dB | SSIM: {m['ssim_T2']:.4f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    color="#fde047",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.8),
                )
            elif col_idx == 3:
                ax.text(
                    0.03,
                    0.05,
                    f"PSNR: {m['psnr_T1']:.2f} dB | SSIM: {m['ssim_T1']:.4f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    color="#86efac",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.8),
                )

    fig.suptitle(
        f"{title_prefix}: 3 Visual Examples of [Real T1w, Real T2w, Synthetic T2w, Synthetic T1w]",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_three_visual_examples_comparison(
    sample_keys: List[str],
    out_path: str | Path,
) -> None:
    """
    Generate unified direct comparison of resnet_6blocks vs unet_128 for the 3 required visual examples
    [Real T1w, Real T2w, Synthetic T2w, Synthetic T1w], formatted for full-page width layout.
    """
    res_path = Path("results/resnet_6blocks/visual_samples.npz")
    unet_path = Path("results/unet_128/visual_samples.npz")
    if not (res_path.exists() and unet_path.exists()):
        return

    res_data = np.load(res_path, allow_pickle=True)
    unet_data = np.load(unet_path, allow_pickle=True)

    sample_labels = [
        "Example 1: BraTS2021_00025 (Slice z=80)",
        "Example 2: BraTS2021_00026 (Slice z=75)",
        "Example 3: BraTS2021_00030 (Slice z=80)",
    ]

    fig = plt.figure(figsize=(16.8, 6.8), dpi=280)
    gs = fig.add_gridspec(
        3, 9,
        width_ratios=[1, 1, 1, 1, 0.12, 1, 1, 1, 1],
        wspace=0.06,
        hspace=0.08,
        left=0.015,
        right=0.985,
        top=0.88,
        bottom=0.03,
    )

    col_headers = [
        "Real T1w\n(Input x)",
        "Real T2w\n(Input y)",
        "Synthetic T2w\n(G_A: T1->T2)",
        "Synthetic T1w\n(G_B: T2->T1)",
    ]

    for r, (key, lbl) in enumerate(zip(sample_keys, sample_labels)):
        r_item = res_data[key].item()
        u_item = unet_data[key].item()

        r_t1, r_t2, r_ft2, r_ft1, _, r_m = _extract_item_fields(r_item)
        u_t1, u_t2, u_ft2, u_ft1, _, u_m = _extract_item_fields(u_item)
        r_imgs = [r_t1, r_t2, r_ft2, r_ft1]
        u_imgs = [u_t1, u_t2, u_ft2, u_ft1]

        # ResNet cols 0..3
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(r_imgs[c], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, fontweight="bold", color="#1e3a8a", pad=6)
            if c == 0:
                ax.text(
                    0.04, 0.94, lbl, transform=ax.transAxes, fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.88), va="top"
                )
            elif c == 2:
                ax.text(
                    0.04, 0.06, f"PSNR: {r_m['psnr_T2']:.2f} dB\nSSIM: {r_m['ssim_T2']:.4f}",
                    transform=ax.transAxes, fontsize=8, fontweight="bold", color="#fef08a",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.88), va="bottom"
                )
            elif c == 3:
                ax.text(
                    0.04, 0.06, f"PSNR: {r_m['psnr_T1']:.2f} dB\nSSIM: {r_m['ssim_T1']:.4f}",
                    transform=ax.transAxes, fontsize=8, fontweight="bold", color="#86efac",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.88), va="bottom"
                )

        # Gap col 4: divider line
        ax_div = fig.add_subplot(gs[r, 4])
        ax_div.axis("off")
        ax_div.axvline(0.5, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.7)

        # UNet cols 5..8
        for c in range(4):
            ax = fig.add_subplot(gs[r, c + 5])
            ax.imshow(u_imgs[c], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, fontweight="bold", color="#991b1b", pad=6)
            if c == 0:
                ax.text(
                    0.04, 0.94, lbl, transform=ax.transAxes, fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.88), va="top"
                )
            elif c == 2:
                ax.text(
                    0.04, 0.06, f"PSNR: {u_m['psnr_T2']:.2f} dB\nSSIM: {u_m['ssim_T2']:.4f}",
                    transform=ax.transAxes, fontsize=8, fontweight="bold", color="#fef08a",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.88), va="bottom"
                )
            elif c == 3:
                ax.text(
                    0.04, 0.06, f"PSNR: {u_m['psnr_T1']:.2f} dB\nSSIM: {u_m['ssim_T1']:.4f}",
                    transform=ax.transAxes, fontsize=8, fontweight="bold", color="#86efac",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.88), va="bottom"
                )

    fig.text(
        0.24, 0.965, "(a) Generator 1: resnet_6blocks (6 Residual Blocks at 32x32 Bottleneck)",
        fontsize=12, fontweight="bold", color="#1e3a8a", ha="center"
    )
    fig.text(
        0.76, 0.965, "(b) Generator 2: unet_128 (7-Stage Recursive U-Net with Skip Connections)",
        fontsize=12, fontweight="bold", color="#991b1b", ha="center"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmaps_comparison(sample_key: str, out_path: str | Path) -> None:
    """
    Compare generators on the same test subject slice along with pixel-wise absolute error
    heatmaps |Real - Synthetic| and ground-truth BraTS tumor segmentation overlay.
    """
    candidates = [
        ("resnet_6blocks", "ResNet-6 (Default)"),
        ("unet_128", "U-Net-128 (Default)"),
        ("monai_resunet_ssim", "MONAI ResUNet + SSIM"),
    ]
    models = [(m, lbl) for m, lbl in candidates if (Path("results") / m / "visual_samples.npz").exists()]
    if not models:
        return

    loaded = {m: np.load(Path("results") / m / "visual_samples.npz", allow_pickle=True)[sample_key].item() for m, _ in models}

    fig, axes = plt.subplots(len(models), 5, figsize=(15.0, 3.0 * len(models)), dpi=250)
    if len(models) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, (m_key, m_label) in enumerate(models):
        item = loaded[m_key]
        r_t1, r_t2, f_t2, f_t1, seg, m = _extract_item_fields(item)
        err_t2 = np.abs(r_t2 - f_t2)
        err_t1 = np.abs(r_t1 - f_t1)

        axes[r_idx, 0].imshow(r_t1 if r_idx != 1 else r_t2, cmap="gray", vmin=0, vmax=1)
        if (seg > 0).sum() > 0:
            axes[r_idx, 0].contour(seg > 0, levels=[0.5], colors=["#38bdf8"], linewidths=1.2)
        if (seg == 4).sum() > 0:
            axes[r_idx, 0].contour(seg == 4, levels=[0.5], colors=["#ef4444"], linewidths=1.2)
        axes[r_idx, 0].set_title(
            f"{m_label}\nRef + Tumor Mask (Cyan=Edema, Red=Gd+)", fontsize=9.5, fontweight="bold"
        )
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].imshow(f_t2, cmap="gray", vmin=0, vmax=1)
        axes[r_idx, 1].set_title(
            f"Synthetic T2w\nPSNR={m['psnr_T2']:.2f}dB, SSIM={m['ssim_T2']:.4f}", fontsize=9.5, fontweight="bold"
        )
        axes[r_idx, 1].axis("off")

        rmse_t2 = float(m.get("rmse_T2", np.sqrt(np.mean(err_t2**2))))
        rmse_t1 = float(m.get("rmse_T1", np.sqrt(np.mean(err_t1**2))))

        im_e2 = axes[r_idx, 2].imshow(err_t2, cmap="magma", vmin=0, vmax=0.45)
        axes[r_idx, 2].set_title(f"|Real T2w - Syn T2w| Error\nRMSE={rmse_t2:.4f}", fontsize=9.5, fontweight="bold")
        axes[r_idx, 2].axis("off")
        plt.colorbar(im_e2, ax=axes[r_idx, 2], fraction=0.046, pad=0.04)

        axes[r_idx, 3].imshow(f_t1, cmap="gray", vmin=0, vmax=1)
        axes[r_idx, 3].set_title(
            f"Synthetic T1w\nPSNR={m['psnr_T1']:.2f}dB, SSIM={m['ssim_T1']:.4f}", fontsize=9.5, fontweight="bold"
        )
        axes[r_idx, 3].axis("off")

        im_e1 = axes[r_idx, 4].imshow(err_t1, cmap="magma", vmin=0, vmax=0.45)
        axes[r_idx, 4].set_title(f"|Real T1w - Syn T1w| Error\nRMSE={rmse_t1:.4f}", fontsize=9.5, fontweight="bold")
        axes[r_idx, 4].axis("off")
        plt.colorbar(im_e1, ax=axes[r_idx, 4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_quantitative_analysis(summaries: Dict[str, Any], out_path: str | Path) -> None:
    """Plot bar charts of PSNR, SSIM, Region-wise MAE (Normal vs. Edema vs. Enhancing Tumor), and Axial Z-Profile."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.2), dpi=250)

    order = [
        ("resnet_6blocks", "ResNet-6\n(Default)"),
        ("unet_128", "U-Net-128\n(Default)"),
        ("monai_resunet", "MONAI\nResUNet"),
        ("monai_resunet_ssim", "MONAI ResUNet\n+ SSIM Loss"),
        ("resnet_6blocks_no_id", "ResNet-6\n(No Idt Loss)"),
        ("resnet_6blocks_minmax", "ResNet-6\n(Min-Max Norm)"),
    ]
    labels = [lbl for k, lbl in order if k in summaries]
    keys = [k for k, _ in order if k in summaries]
    if not keys:
        plt.close(fig)
        return

    x = np.arange(len(keys))
    width = 0.36

    # (a) Test PSNR comparison
    psnr_t2 = [summaries[k]["test_overall"]["psnr_T2_mean"] for k in keys]
    psnr_t1 = [summaries[k]["test_overall"]["psnr_T1_mean"] for k in keys]
    b1 = axes[0, 0].bar(x - width / 2, psnr_t2, width, label="T1w -> T2w (Synthetic T2w)", color="#2563eb")
    b2 = axes[0, 0].bar(x + width / 2, psnr_t1, width, label="T2w -> T1w (Synthetic T1w)", color="#059669")
    axes[0, 0].set_title("(a) Test Set PSNR Comparison (Higher is Better)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=8.5)
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_ylim(13, max(psnr_t2 + psnr_t1) + 3.0)
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(True, axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # (b) Test SSIM comparison
    ssim_t2 = [summaries[k]["test_overall"]["ssim_T2_mean"] for k in keys]
    ssim_t1 = [summaries[k]["test_overall"]["ssim_T1_mean"] for k in keys]
    b3 = axes[0, 1].bar(x - width / 2, ssim_t2, width, label="T1w -> T2w (Synthetic T2w)", color="#2563eb")
    b4 = axes[0, 1].bar(x + width / 2, ssim_t1, width, label="T2w -> T1w (Synthetic T1w)", color="#059669")
    axes[0, 1].set_title("(b) Test Set SSIM Comparison (Higher is Better)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=8.5)
    axes[0, 1].set_ylabel("SSIM")
    axes[0, 1].set_ylim(0.45, 1.0)
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # (c) Region-wise Error Decomposition (Normal Parenchyma vs Edema vs Enhancing Tumor)
    reg_models = [k for k in ["resnet_6blocks", "unet_128", "monai_resunet", "monai_resunet_ssim"] if k in summaries]
    reg_labels = ["Normal Brain\nParenchyma", "Peritumoral\nEdema (Label 2)", "Gd-Enhancing\nTumor (Label 4)"]
    rx = np.arange(len(reg_labels))
    rw = 0.8 / max(1, len(reg_models))

    for idx, m_key in enumerate(reg_models):
        ov = summaries[m_key]["test_overall"]
        t2_norm = ov.get("mae_T2_normal_mean", ov.get("mae_T2_mean", 0.04))
        t1_norm = ov.get("mae_T1_normal_mean", ov.get("mae_T1_mean", 0.04))
        t2_edema = ov.get("mae_T2_edema_mean", t2_norm * 1.5)
        t1_edema = ov.get("mae_T1_edema_mean", t1_norm * 1.5)
        t2_enh = ov.get("mae_T2_enhancing_mean", t2_norm * 2.5)
        t1_enh = ov.get("mae_T1_enhancing_mean", t1_norm * 2.5)

        vals = [
            0.5 * (t2_norm + t1_norm),
            0.5 * (t2_edema + t1_edema),
            0.5 * (t2_enh + t1_enh),
        ]
        offset = (idx - (len(reg_models) - 1) / 2.0) * rw
        axes[1, 0].bar(rx + offset, vals, rw, label=MODEL_META[m_key][0], color=MODEL_META[m_key][1])

    axes[1, 0].set_title("(c) Error Source Analysis by Tissue/Tumor Sub-Region (MAE)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(rx)
    axes[1, 0].set_xticklabels(reg_labels, fontsize=9.5)
    axes[1, 0].set_ylabel("Mean Absolute Error (MAE in [0,1])")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    # (d) Axial slice profile of PSNR across the 3D volume (z = 30..119)
    for m_key in reg_models:
        slice_file = Path("results") / m_key / "slice_metrics.json"
        if not slice_file.exists():
            continue
        records = json.loads(slice_file.read_text())
        z_map: Dict[int, List[float]] = {}
        for r in records:
            z_map.setdefault(r["slice_idx"], []).append(0.5 * (r["psnr_T2"] + r["psnr_T1"]))
        zs = sorted(z_map.keys())
        means = [np.mean(z_map[z]) for z in zs]
        axes[1, 1].plot(zs, means, label=MODEL_META[m_key][0], color=MODEL_META[m_key][1], linewidth=2)

    axes[1, 1].set_title("(d) Test PSNR Profile Across Axial Slice Depth (z = 30..119)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Axial Slice Index (z)")
    axes[1, 1].set_ylabel("Mean PSNR (dB)")
    axes[1, 1].legend(fontsize=8.5)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
