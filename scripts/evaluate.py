#!/usr/bin/env python3
"""
Evaluation CLI for Multi-Parametric Brain MRI Generation with CycleGAN & MONAI.

Evaluates trained model checkpoints on the fixed held-out 4-subject test cohort:
- BraTS2021_00025, BraTS2021_00026, BraTS2021_00028, BraTS2021_00030 (360 slices total)
Computes:
- PSNR (T2w, T1w)
- SSIM (T2w, T1w)
- RMSE (T2w, T1w)
- MAE (T2w, T1w)
- Region-wise error decomposition (Normal Parenchyma, Peritumoral Edema, Enhancing Tumor Core)
- Saves visual sample predictions to visual_samples.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cyclegan import CycleGANConfig, CycleGANModel
from src.dataset import TEST_SUBJECTS, build_dataloaders
from src.metrics import MedicalTranslationEvaluator, compute_psnr_explicit, compute_ssim_explicit, to_unit_range


TARGET_VISUAL_SAMPLES = [
    ("BraTS2021_00025", 65),
    ("BraTS2021_00025", 80),
    ("BraTS2021_00026", 60),
    ("BraTS2021_00026", 75),
    ("BraTS2021_00028", 70),
    ("BraTS2021_00028", 85),
    ("BraTS2021_00030", 65),
    ("BraTS2021_00030", 80),
]


@torch.no_grad()
def evaluate_checkpoint(
    model_name: str,
    ckpt_path: Path,
    data_root: str = "AI4MIA_Assignment4_Dataset",
    results_dir: str = "results",
    batch_size: int = 12,
    norm_mode: str = "percentile",
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f" Evaluating: {model_name}")
    print(f" Checkpoint: {ckpt_path}")
    print(f" Norm Mode : {norm_mode}")
    print(f"{'='*70}\n")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    _, _, test_loader, _ = build_dataloaders(
        data_root=data_root, batch_size=batch_size, norm_mode=norm_mode, num_workers=4
    )

    # Detect generator architecture
    netG = "resnet_6blocks"
    lambda_ssim = 0.0
    if "unet_128" in model_name:
        netG = "unet_128"
    elif "monai_resunet" in model_name:
        netG = "monai_resunet"
        if "ssim" in model_name:
            lambda_ssim = 2.0

    cfg = CycleGANConfig(netG=netG, lambda_ssim=lambda_ssim)
    model = CycleGANModel(cfg)
    model.load_checkpoint(str(ckpt_path))
    model.netG_A.eval()
    model.netG_B.eval()

    evaluator = MedicalTranslationEvaluator()
    all_metrics: Dict[str, List[np.ndarray]] = {}
    subj_metrics: Dict[str, Dict[str, List[float]]] = {s: {} for s in TEST_SUBJECTS}
    slice_records: List[Dict[str, Any]] = []
    saved_samples: Dict[str, Dict[str, np.ndarray]] = {}

    target_set = set(TARGET_VISUAL_SAMPLES)

    for batch in test_loader:
        real_A = batch["A"].to(model.device)
        real_B = batch["B"].to(model.device)
        seg = batch["seg"].to(model.device)
        subjects = batch["subject"]
        slice_idxs = batch["slice_idx"]

        fake_B = model.netG_A(real_A)
        rec_A = model.netG_B(fake_B)
        fake_A = model.netG_B(real_B)
        rec_B = model.netG_A(fake_A)

        bm = evaluator.evaluate_batch(real_A, real_B, fake_B, fake_A, rec_A, rec_B, seg)
        rA_01, rB_01 = to_unit_range(real_A), to_unit_range(real_B)
        fB_01, fA_01 = to_unit_range(fake_B), to_unit_range(fake_A)

        for k, arr in bm.items():
            all_metrics.setdefault(k, []).append(arr)

        B = real_A.size(0)
        for i in range(B):
            subj = subjects[i]
            z = int(slice_idxs[i])
            rec = {"subject": subj, "slice_idx": z}
            for k in bm.keys():
                val = float(bm[k][i])
                rec[k] = val
                subj_metrics[subj].setdefault(k, []).append(val)
            slice_records.append(rec)

            if (subj, z) in target_set:
                key = f"{subj}_z{z:03d}"
                saved_samples[key] = {
                    "real_A": rA_01[i, 0].cpu().numpy(),
                    "real_B": rB_01[i, 0].cpu().numpy(),
                    "fake_B": fB_01[i, 0].cpu().numpy(),
                    "fake_A": fA_01[i, 0].cpu().numpy(),
                    "rec_A": to_unit_range(rec_A)[i, 0].cpu().numpy(),
                    "rec_B": to_unit_range(rec_B)[i, 0].cpu().numpy(),
                    "seg": seg[i, 0].cpu().numpy(),
                    "psnr_T2": float(bm["psnr_T2"][i]),
                    "ssim_T2": float(bm["ssim_T2"][i]),
                    "psnr_T1": float(bm["psnr_T1"][i]),
                    "ssim_T1": float(bm["ssim_T1"][i]),
                }

    overall: Dict[str, float] = {}
    for k, vlist in all_metrics.items():
        cat = np.concatenate(vlist, axis=0)
        overall[f"{k}_mean"] = float(np.mean(cat))
        overall[f"{k}_std"] = float(np.std(cat))

    per_subject: Dict[str, Dict[str, float]] = {}
    for subj, d in subj_metrics.items():
        per_subject[subj] = {f"{k}_mean": float(np.mean(vals)) for k, vals in d.items() if len(vals) > 0}

    out_dir = Path(results_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "summary.json"
    summary = {}
    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
        except Exception:
            pass

    summary.update({
        "exp_name": model_name,
        "checkpoint": str(ckpt_path),
        "test_overall": overall,
        "test_per_subject": per_subject,
    })

    summary_file.write_text(json.dumps(summary, indent=2))
    (out_dir / "slice_metrics.json").write_text(json.dumps(slice_records, indent=2))

    if saved_samples:
        np.savez_compressed(out_dir / "visual_samples.npz", **saved_samples)
        print(f"Saved {len(saved_samples)} visual sample slices to {out_dir / 'visual_samples.npz'}")

    print(f"\n--- [Evaluation Results: {model_name}] ---")
    print(f"  T1ce -> T2w (Synthetic T2w):")
    print(f"    PSNR : {overall['psnr_T2_mean']:.2f} ± {overall['psnr_T2_std']:.2f} dB")
    print(f"    SSIM : {overall['ssim_T2_mean']:.4f} ± {overall['ssim_T2_std']:.4f}")
    print(f"    RMSE : {overall['rmse_T2_mean']:.4f}")
    print(f"    MAE  : {overall['mae_T2_mean']:.4f}")
    print(f"  T2w -> T1ce (Synthetic T1ce):")
    print(f"    PSNR : {overall['psnr_T1_mean']:.2f} ± {overall['psnr_T1_std']:.2f} dB")
    print(f"    SSIM : {overall['ssim_T1_mean']:.4f} ± {overall['ssim_T1_std']:.4f}")
    print(f"    RMSE : {overall['rmse_T1_mean']:.4f}")
    print(f"    MAE  : {overall['mae_T1_mean']:.4f}")
    if "mae_edema" in overall:
        print(f"  Tissue Sub-Region Decomposition:")
        print(f"    Healthy Parenchyma MAE: {overall.get('mae_normal_mean', 0.0):.4f}")
        print(f"    Vasogenic Edema MAE   : {overall.get('mae_edema_mean', 0.0):.4f}")
        print(f"    Enhancing Core MAE    : {overall.get('mae_core_mean', 0.0):.4f}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN checkpoints on held-out test subjects")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_6blocks",
        choices=["resnet_6blocks", "unet_128", "monai_resunet", "monai_resunet_ssim", "resnet_6blocks_minmax", "all"],
        help="Model to evaluate",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to checkpoint .pth file")
    parser.add_argument("--data-root", type=str, default="AI4MIA_Assignment4_Dataset", help="BraTS dataset root")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=12, help="Mini-batch size")

    args = parser.parse_args()

    models = (
        ["resnet_6blocks", "unet_128", "monai_resunet", "monai_resunet_ssim", "resnet_6blocks_minmax"]
        if args.model == "all"
        else [args.model]
    )

    all_summaries = {}
    master_summary_path = Path(args.results_dir) / "all_experiments_summary.json"
    if master_summary_path.exists():
        all_summaries = json.loads(master_summary_path.read_text())

    for m in models:
        norm_mode = "minmax" if "minmax" in m else "percentile"
        ckpt = Path(args.checkpoint) if args.checkpoint else Path("checkpoints") / m / "best_model.pth"
        if not ckpt.exists():
            ckpt = Path("checkpoints") / m / "final_model.pth"
        summ = evaluate_checkpoint(
            m,
            ckpt,
            data_root=args.data_root,
            results_dir=args.results_dir,
            batch_size=args.batch_size,
            norm_mode=norm_mode,
        )
        if m in all_summaries and isinstance(all_summaries[m], dict):
            all_summaries[m].update(summ)
        else:
            all_summaries[m] = summ

    master_summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nAll evaluations updated at: {master_summary_path}")


if __name__ == "__main__":
    main()
