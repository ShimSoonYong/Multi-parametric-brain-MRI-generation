"""
Main Experiment Runner for AI4MIA Assignment 4 (Multi-parametric Brain MRI Generation).

Trains and evaluates 4 systematic experiments on the BraTS 2021 dataset:
1. `resnet_6blocks`: Default CycleGAN with 6-block ResNet generator & MONAI 99.5% percentile scaling.
2. `unet_128`: Default CycleGAN with 7-level U-Net-128 generator & MONAI 99.5% percentile scaling.
3. `monai_resunet_ssim`: MONAI Residual U-Net (`monai.networks.nets.UNet`) + MONAI `SSIMLoss` cycle regularization.
4. `resnet_6blocks_minmax`: Ablation with naive Min-Max scaling (analyzing Gadolinium spike error source).

Saves:
- Per-epoch training losses and validation PSNR/SSIM histories (`results/<exp>/history.json`)
- Comprehensive test-set quantitative metrics across all 4 fixed test subjects (`00025`, `00026`, `00028`, `00030`),
  including PSNR, SSIM, RMSE, MAE, and region-wise error decomposition (Normal vs. Edema vs. Enhancing Tumor)
- Visual sample predictions (`[Real T1w, Real T2w, Synthetic T2w, Synthetic T1w, Rec T1w, Rec T2w, Seg]`)
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.cyclegan import CycleGANConfig, CycleGANModel
from src.dataset import TEST_SUBJECTS, build_dataloaders
from src.metrics import MedicalTranslationEvaluator, compute_psnr_explicit, compute_ssim_explicit, to_unit_range


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_detailed_test_set(
    model: CycleGANModel,
    test_loader: torch.utils.data.DataLoader,
    evaluator: MedicalTranslationEvaluator,
    save_dir: Path,
    target_samples: List[tuple[str, int]],
) -> Dict[str, Any]:
    """
    Run full evaluation on the fixed 4-subject test set (00025, 00026, 00028, 00030),
    collect overall + per-subject + per-slice metrics, and save visual examples.
    """
    model.netG_A.eval()
    model.netG_B.eval()

    all_metrics: Dict[str, List[np.ndarray]] = {}
    subj_metrics: Dict[str, Dict[str, List[float]]] = {s: {} for s in TEST_SUBJECTS}
    slice_records: List[Dict[str, Any]] = []
    saved_samples: Dict[str, Dict[str, np.ndarray]] = {}

    target_set = set(target_samples)

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

        # Also verify explicit PSNR/SSIM parity
        rA_01, rB_01 = to_unit_range(real_A), to_unit_range(real_B)
        fB_01, fA_01 = to_unit_range(fake_B), to_unit_range(fake_A)
        bm["psnr_T2_explicit"] = compute_psnr_explicit(fB_01, rB_01).cpu().numpy()
        bm["ssim_T2_explicit"] = compute_ssim_explicit(fB_01, rB_01).cpu().numpy()
        bm["psnr_T1_explicit"] = compute_psnr_explicit(fA_01, rA_01).cpu().numpy()
        bm["ssim_T1_explicit"] = compute_ssim_explicit(fA_01, rA_01).cpu().numpy()

        for k, v in bm.items():
            all_metrics.setdefault(k, []).append(v)

        bsz = real_A.shape[0]
        for i in range(bsz):
            s_name = str(subjects[i])
            z_idx = int(slice_idxs[i].item() if isinstance(slice_idxs[i], torch.Tensor) else slice_idxs[i])

            for k, v in bm.items():
                if not np.isnan(v[i]):
                    subj_metrics[s_name].setdefault(k, []).append(float(v[i]))

            slice_records.append({
                "subject": s_name,
                "slice_idx": z_idx,
                "psnr_T2": float(bm["psnr_T2"][i]),
                "ssim_T2": float(bm["ssim_T2"][i]),
                "rmse_T2": float(bm["rmse_T2"][i]),
                "mae_T2": float(bm["mae_T2"][i]),
                "psnr_T1": float(bm["psnr_T1"][i]),
                "ssim_T1": float(bm["ssim_T1"][i]),
                "rmse_T1": float(bm["rmse_T1"][i]),
                "mae_T1": float(bm["mae_T1"][i]),
                "tumor_pixels": int((seg[i] > 0).sum().item()),
                "enhancing_pixels": int((seg[i] == 4).sum().item()),
            })

            if (s_name, z_idx) in target_set:
                key = f"{s_name}_z{z_idx:03d}"
                saved_samples[key] = {
                    "real_T1": rA_01[i, 0].cpu().numpy(),
                    "real_T2": rB_01[i, 0].cpu().numpy(),
                    "fake_T2": fB_01[i, 0].cpu().numpy(),
                    "fake_T1": fA_01[i, 0].cpu().numpy(),
                    "rec_T1": to_unit_range(rec_A)[i, 0].cpu().numpy(),
                    "rec_T2": to_unit_range(rec_B)[i, 0].cpu().numpy(),
                    "seg": seg[i, 0].cpu().numpy(),
                    "metrics": {k: float(bm[k][i]) for k in ["psnr_T2", "ssim_T2", "psnr_T1", "ssim_T1", "rmse_T2", "rmse_T1"]},
                }

    overall_summary: Dict[str, float] = {}
    for k, arr_list in all_metrics.items():
        concat = np.concatenate(arr_list, axis=0)
        overall_summary[f"{k}_mean"] = float(np.nanmean(concat))
        overall_summary[f"{k}_std"] = float(np.nanstd(concat))

    per_subject_summary: Dict[str, Dict[str, float]] = {}
    for s_name, m_dict in subj_metrics.items():
        per_subject_summary[s_name] = {}
        for k, vals in m_dict.items():
            per_subject_summary[s_name][f"{k}_mean"] = float(np.mean(vals))
            per_subject_summary[s_name][f"{k}_std"] = float(np.std(vals))

    np.savez_compressed(save_dir / "visual_samples.npz", **{k: v for k, v in saved_samples.items()})
    with open(save_dir / "slice_metrics.json", "w") as f:
        json.dump(slice_records, f, indent=2)

    return {
        "overall": overall_summary,
        "per_subject": per_subject_summary,
    }


def run_single_experiment(
    exp_name: str,
    cfg: CycleGANConfig,
    norm_mode: str,
    batch_size: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate a single CycleGAN configuration end-to-end."""
    set_seed(seed)
    exp_dir = Path("results") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints") / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[Experiment: {exp_name}] netG={cfg.netG} | norm={norm_mode} | lambda_ssim={cfg.lambda_ssim}")
    print(f"{'='*70}")

    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        batch_size=batch_size, norm_mode=norm_mode, num_workers=4, seed=seed
    )
    evaluator = MedicalTranslationEvaluator(max_val=1.0)
    model = CycleGANModel(cfg)

    n_params_G = sum(p.numel() for p in model.netG_A.parameters())
    n_params_D = sum(p.numel() for p in model.netD_A.parameters())
    total_epochs = cfg.n_epochs + cfg.n_epochs_decay

    history: List[Dict[str, float]] = []
    best_val_ssim = -1.0
    start_epoch = 1
    t_start = time.time()

    latest_ckpt = ckpt_dir / "latest_checkpoint.pth"
    if latest_ckpt.exists():
        try:
            loaded_state = torch.load(latest_ckpt, map_location=model.device, weights_only=False)
            model.netG_A.load_state_dict(loaded_state["netG_A"])
            model.netG_B.load_state_dict(loaded_state["netG_B"])
            model.netD_A.load_state_dict(loaded_state["netD_A"])
            model.netD_B.load_state_dict(loaded_state["netD_B"])
            model.optimizer_G.load_state_dict(loaded_state["opt_G"])
            model.optimizer_D.load_state_dict(loaded_state["opt_D"])
            model.scheduler_G.load_state_dict(loaded_state["sched_G"])
            model.scheduler_D.load_state_dict(loaded_state["sched_D"])
            start_epoch = loaded_state["epoch"] + 1
            history = loaded_state.get("history", [])
            best_val_ssim = loaded_state.get("best_val_ssim", -1.0)
            print(f"Resuming {exp_name} from epoch {start_epoch}/{total_epochs}...")
        except Exception as e:
            print(f"Warning: could not load latest_checkpoint.pth: {e}")
    elif (ckpt_dir / "best_model.pth").exists() and exp_name == "unet_128":
        try:
            model.load_checkpoint(str(ckpt_dir / "best_model.pth"))
            start_epoch = 95
            for _ in range(94):
                model.step_schedulers()
            print(f"Resuming unet_128 from epoch 95/{total_epochs} using saved best_model weights...")
        except Exception as e:
            print(f"Warning: could not resume unet_128 best_model: {e}")

    for epoch in range(start_epoch, total_epochs + 1):
        ep_t0 = time.time()
        # Reshuffle unpaired domains at the start of every epoch
        train_ds.shuffle_domains()

        running_losses: Dict[str, List[float]] = {}
        for batch in train_loader:
            step_losses = model.optimize_step(batch["A"], batch["B"])
            for k, v in step_losses.items():
                running_losses.setdefault(k, []).append(v)

        model.step_schedulers()
        current_lr = float(model.optimizer_G.param_groups[0]["lr"])

        ep_record: Dict[str, float] = {
            "epoch": epoch,
            "lr": current_lr,
            "epoch_time_s": float(time.time() - ep_t0),
        }
        for k, v_list in running_losses.items():
            ep_record[f"train_{k}"] = float(np.mean(v_list))

        # Validate every 2 epochs (and final epoch)
        if epoch % 2 == 0 or epoch == total_epochs or epoch == 1:
            val_metrics = model.evaluate(val_loader, evaluator)
            for k in ["psnr_T2_mean", "ssim_T2_mean", "psnr_T1_mean", "ssim_T1_mean", "rmse_T2_mean", "rmse_T1_mean"]:
                ep_record[f"val_{k}"] = val_metrics[k]

            mean_val_ssim = 0.5 * (val_metrics["ssim_T2_mean"] + val_metrics["ssim_T1_mean"])
            if mean_val_ssim > best_val_ssim:
                best_val_ssim = mean_val_ssim
                model.save_checkpoint(str(ckpt_dir / "best_model.pth"))

            print(
                f"Epoch [{epoch:02d}/{total_epochs:02d}] ({ep_record['epoch_time_s']:.1f}s) "
                f"G={ep_record['train_G_total']:.3f} Cyc={ep_record['train_cycle_A']+ep_record['train_cycle_B']:.3f} "
                f"D={ep_record['train_D_A']+ep_record['train_D_B']:.3f} | "
                f"Val T2 PSNR={val_metrics['psnr_T2_mean']:.2f}dB SSIM={val_metrics['ssim_T2_mean']:.4f} | "
                f"Val T1 PSNR={val_metrics['psnr_T1_mean']:.2f}dB SSIM={val_metrics['ssim_T1_mean']:.4f}"
            )

        history.append(ep_record)

        # Save latest_checkpoint after every epoch for seamless recovery
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

    total_train_time = float(time.time() - t_start)
    model.save_checkpoint(str(ckpt_dir / "final_model.pth"))

    # Load best validation checkpoint for test evaluation
    model.load_checkpoint(str(ckpt_dir / "best_model.pth"))

    # Representative axial slices across the 4 test subjects (including tumor & anatomical landmarks)
    target_samples = [
        ("BraTS2021_00025", 65),
        ("BraTS2021_00025", 80),
        ("BraTS2021_00026", 60),
        ("BraTS2021_00026", 75),
        ("BraTS2021_00028", 70),
        ("BraTS2021_00028", 85),
        ("BraTS2021_00030", 65),
        ("BraTS2021_00030", 80),
    ]
    test_results = evaluate_detailed_test_set(model, test_loader, evaluator, exp_dir, target_samples)

    exp_summary = {
        "exp_name": exp_name,
        "netG": cfg.netG,
        "norm_mode": norm_mode,
        "lambda_ssim": cfg.lambda_ssim,
        "G_params": n_params_G,
        "D_params": n_params_D,
        "total_epochs": total_epochs,
        "total_train_time_s": total_train_time,
        "test_overall": test_results["overall"],
        "test_per_subject": test_results["per_subject"],
        "history": history,
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(exp_summary, f, indent=2)

    ov = test_results["overall"]
    print(f"\n---> [Test Results: {exp_name}] <---")
    print(
        f"  T1w -> T2w (Synthetic T2w): PSNR = {ov['psnr_T2_mean']:.2f} ± {ov['psnr_T2_std']:.2f} dB | "
        f"SSIM = {ov['ssim_T2_mean']:.4f} ± {ov['ssim_T2_std']:.4f} | "
        f"RMSE = {ov['rmse_T2_mean']:.4f} | MAE = {ov['mae_T2_mean']:.4f}"
    )
    print(
        f"  T2w -> T1w (Synthetic T1w): PSNR = {ov['psnr_T1_mean']:.2f} ± {ov['psnr_T1_std']:.2f} dB | "
        f"SSIM = {ov['ssim_T1_mean']:.4f} ± {ov['ssim_T1_std']:.4f} | "
        f"RMSE = {ov['rmse_T1_mean']:.4f} | MAE = {ov['mae_T1_mean']:.4f}"
    )
    return exp_summary


def main() -> None:
    os.makedirs("results", exist_ok=True)
    experiments = [
        (
            "resnet_6blocks",
            CycleGANConfig(netG="resnet_6blocks", n_epochs=20, n_epochs_decay=20, lambda_ssim=0.0),
            "percentile",
        ),
        (
            "unet_128",
            CycleGANConfig(netG="unet_128", n_epochs=20, n_epochs_decay=20, lambda_ssim=0.0),
            "percentile",
        ),
        (
            "monai_resunet_ssim",
            CycleGANConfig(netG="monai_resunet", n_epochs=20, n_epochs_decay=20, lambda_ssim=2.0),
            "percentile",
        ),
        (
            "resnet_6blocks_minmax",
            CycleGANConfig(netG="resnet_6blocks", n_epochs=20, n_epochs_decay=20, lambda_ssim=0.0),
            "minmax",
        ),
    ]

    all_summaries: Dict[str, Any] = {}
    master_summary_path = Path("results/all_experiments_summary.json")
    if master_summary_path.exists():
        with open(master_summary_path, "r") as f:
            all_summaries = json.load(f)

    for exp_name, cfg, norm_mode in experiments:
        exp_summary_file = Path("results") / exp_name / "summary.json"
        if exp_summary_file.exists():
            print(f"Skipping {exp_name} (already completed at {exp_summary_file})")
            with open(exp_summary_file, "r") as f:
                all_summaries[exp_name] = json.load(f)
            continue

        summary = run_single_experiment(exp_name, cfg, norm_mode=norm_mode, batch_size=12, seed=42)
        all_summaries[exp_name] = summary
        with open(master_summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

    print("\nAll 4 experiments completed successfully!")


if __name__ == "__main__":
    main()
