#!/usr/bin/env python3
"""
Training CLI for Multi-Parametric Brain MRI Generation with CycleGAN & MONAI.

Supports:
- resnet_6blocks: ResNet generator with 6 residual blocks at 32x32 resolution.
- unet_128: 7-stage U-Net with skip connections down to 1x1 bottleneck.
- monai_resunet: MONAI Residual U-Net (4 stages, 2 residual units per stage).
- monai_resunet_ssim: MONAI Residual U-Net with structural SSIM cycle regularization.
- resnet_6blocks_minmax: Ablation model using naive Min-Max normalization.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cyclegan import CycleGANConfig, CycleGANModel
from src.dataset import (
    TEST_SUBJECTS,
    VAL_SUBJECTS,
    build_dataloaders,
)
from src.metrics import MedicalTranslationEvaluator


EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "resnet_6blocks": {
        "netG": "resnet_6blocks",
        "norm_mode": "percentile",
        "lambda_ssim": 0.0,
        "description": "Standard CycleGAN with 6-block ResNet generator & 99.5% percentile scaling",
    },
    "unet_128": {
        "netG": "unet_128",
        "norm_mode": "percentile",
        "lambda_ssim": 0.0,
        "description": "Standard CycleGAN with 7-level U-Net-128 generator & 99.5% percentile scaling",
    },
    "monai_resunet": {
        "netG": "monai_resunet",
        "norm_mode": "percentile",
        "lambda_ssim": 0.0,
        "description": "MONAI Residual U-Net (4 stages, 2 resunits) with standard CycleGAN loss",
    },
    "monai_resunet_ssim": {
        "netG": "monai_resunet",
        "norm_mode": "percentile",
        "lambda_ssim": 2.0,
        "description": "MONAI Residual U-Net with MONAI SSIMLoss structural cycle regularization",
    },
    "resnet_6blocks_minmax": {
        "netG": "resnet_6blocks",
        "norm_mode": "minmax",
        "lambda_ssim": 0.0,
        "description": "Ablation: ResNet-6 with naive Min-Max scaling (exposing Gadolinium outlier crush)",
    },
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def train_single_model(
    exp_name: str,
    data_root: str = "AI4MIA_Assignment4_Dataset",
    n_epochs: int = 20,
    n_epochs_decay: int = 20,
    batch_size: int = 12,
    lr: float = 0.0002,
    seed: int = 42,
    ckpt_root: str = "checkpoints",
    results_root: str = "results",
) -> Dict[str, Any]:
    if exp_name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment: {exp_name}. Choose from {list(EXPERIMENT_PRESETS.keys())}")

    preset = EXPERIMENT_PRESETS[exp_name]
    set_seed(seed)

    ckpt_dir = Path(ckpt_root) / exp_name
    exp_dir = Path(results_root) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" Starting Experiment: {exp_name}")
    print(f" Description: {preset['description']}")
    print(f" Architecture: netG={preset['netG']} | Norm: {preset['norm_mode']} | SSIM Loss: {preset['lambda_ssim']}")
    print(f"{'='*70}\n")

    train_loader, val_loader, _, _ = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        norm_mode=preset["norm_mode"],
        num_workers=4,
    )

    cfg = CycleGANConfig(
        netG=preset["netG"],
        lambda_ssim=preset["lambda_ssim"],
        n_epochs=n_epochs,
        n_epochs_decay=n_epochs_decay,
        lr=lr,
    )

    model = CycleGANModel(cfg)
    evaluator = MedicalTranslationEvaluator()

    total_epochs = n_epochs + n_epochs_decay
    start_epoch = 1
    best_val_ssim = -1.0
    history: list[dict[str, Any]] = []

    # Resume from checkpoint if available
    latest_ckpt = ckpt_dir / "latest_checkpoint.pth"
    if latest_ckpt.exists():
        print(f"Found existing checkpoint at {latest_ckpt}, resuming training...")
        chk = torch.load(latest_ckpt, map_location=model.device, weights_only=False)
        start_epoch = chk["epoch"] + 1
        model.netG_A.load_state_dict(chk["netG_A"])
        model.netG_B.load_state_dict(chk["netG_B"])
        model.netD_A.load_state_dict(chk["netD_A"])
        model.netD_B.load_state_dict(chk["netD_B"])
        model.optimizer_G.load_state_dict(chk["opt_G"])
        model.optimizer_D.load_state_dict(chk["opt_D"])
        model.scheduler_G.load_state_dict(chk["sched_G"])
        model.scheduler_D.load_state_dict(chk["sched_D"])
        best_val_ssim = chk.get("best_val_ssim", -1.0)
        history = chk.get("history", [])
        print(f"Resumed successfully from epoch {chk['epoch']} (best validation SSIM: {best_val_ssim:.4f})")

    t_start = time.time()
    for epoch in range(start_epoch, total_epochs + 1):
        ep_t0 = time.time()
        # Strictly unpaired shuffle across domains
        train_loader.dataset.shuffle_domains()
        model.train()

        loss_accum: dict[str, float] = {}
        n_batches = 0

        for batch in train_loader:
            losses = model.optimize_step(batch["A"], batch["B"])
            for k, v in losses.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + float(v)
            n_batches += 1

        model.update_learning_rate()
        ep_time = time.time() - ep_t0
        avg_losses = {k: v / max(1, n_batches) for k, v in loss_accum.items()}

        # Validation evaluation
        model.netG_A.eval()
        model.netG_B.eval()
        val_metrics: dict[str, list[float]] = {"psnr_T2": [], "ssim_T2": [], "psnr_T1": [], "ssim_T1": []}

        with torch.no_grad():
            for batch in val_loader:
                real_A = batch["A"].to(model.device)
                real_B = batch["B"].to(model.device)
                fake_B = model.netG_A(real_A)
                rec_A = model.netG_B(fake_B)
                fake_A = model.netG_B(real_B)
                rec_B = model.netG_A(fake_A)
                bm = evaluator.evaluate_batch(real_A, real_B, fake_B, fake_A, rec_A, rec_B)
                for k in val_metrics:
                    val_metrics[k].extend(bm[k].tolist())

        val_summary = {k: float(np.mean(v)) for k, v in val_metrics.items()}
        val_score = 0.5 * (val_summary["ssim_T2"] + val_summary["ssim_T1"])

        ep_record = {
            "epoch": epoch,
            "time_s": ep_time,
            "lr": float(model.optimizer_G.param_groups[0]["lr"]),
            **avg_losses,
            **{f"val_{k}": v for k, v in val_summary.items()},
        }
        history.append(ep_record)

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] {ep_time:.1f}s | "
            f"Loss_G: {avg_losses.get('loss_G', 0.0):.3f} | "
            f"Loss_D: {avg_losses.get('loss_D', 0.0):.3f} | "
            f"Cyc: {avg_losses.get('loss_cycle', 0.0):.3f} | "
            f"Val PSNR(T2): {val_summary['psnr_T2']:.2f}dB | "
            f"Val SSIM(T2): {val_summary['ssim_T2']:.4f} | "
            f"Val SSIM(T1): {val_summary['ssim_T1']:.4f}"
        )

        if val_score > best_val_ssim:
            best_val_ssim = val_score
            model.save_checkpoint(str(ckpt_dir / "best_model.pth"))
            print(f"  --> Saved new best model checkpoint (Val SSIM: {best_val_ssim:.4f})")

        torch.save(
            {
                "epoch": epoch,
                "netG_A": model.netG_A.state_dict(),
                "netG_B": model.netG_B.state_dict(),
                "netD_A": model.netD_A.state_dict(),
                "netD_B": model.netD_B.state_dict(),
                "opt_G": model.optimizer_G.state_dict(),
                "opt_D": model.optimizer_D.state_dict(),
                "sched_G": model.scheduler_G.state_dict(),
                "sched_D": model.scheduler_D.state_dict(),
                "best_val_ssim": best_val_ssim,
                "history": history,
            },
            ckpt_dir / "latest_checkpoint.pth",
        )

    model.save_checkpoint(str(ckpt_dir / "final_model.pth"))
    total_time = time.time() - t_start
    print(f"\nExperiment {exp_name} finished in {total_time/60.0:.1f} minutes!")

    # Save history
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"exp_name": exp_name, "best_val_ssim": best_val_ssim, "total_time_s": total_time}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CycleGAN models for Brain MRI translation")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_6blocks",
        choices=list(EXPERIMENT_PRESETS.keys()) + ["all"],
        help="Model architecture / experiment preset to train",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs at initial learning rate")
    parser.add_argument("--epochs-decay", type=int, default=20, help="Epochs linearly decaying learning rate to 0")
    parser.add_argument("--batch-size", type=int, default=12, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-root", type=str, default="AI4MIA_Assignment4_Dataset", help="BraTS dataset root")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation results")

    args = parser.parse_args()

    if args.model == "all":
        models_to_run = list(EXPERIMENT_PRESETS.keys())
    else:
        models_to_run = [args.model]

    for m in models_to_run:
        train_single_model(
            exp_name=m,
            data_root=args.data_root,
            n_epochs=args.epochs,
            n_epochs_decay=args.epochs_decay,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            ckpt_root=args.checkpoint_dir,
            results_root=args.results_dir,
        )


if __name__ == "__main__":
    main()
