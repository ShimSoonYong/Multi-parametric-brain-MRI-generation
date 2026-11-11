"""
Quantitative Evaluation Metrics for Medical Image Translation (MONAI + Explicit Formulas).

Implements:
1. PSNR (Peak Signal-to-Noise Ratio) & SSIM (Structural Similarity Index Measure)
   using both MONAI (`PSNRMetric`, `SSIMMetric`) and explicit mathematical definitions
   from AI4MIA Week 10/11 lecture notes.
2. Extra quantitative metrics: RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
3. Anatomical & Pathological Region-wise Error Analysis using BraTS segmentation masks
   (Healthy Brain Parenchyma vs. Peritumoral Edema vs. Gadolinium-Enhancing Tumor).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from monai.metrics import MAEMetric, PSNRMetric, RMSEMetric, SSIMMetric


def to_unit_range(img: torch.Tensor) -> torch.Tensor:
    """Rescale tensor from generator range [-1, 1] to [0, 1]."""
    return ((img.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def compute_psnr_explicit(pred_01: torch.Tensor, target_01: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Explicit PSNR formula (Week 10 Slide 3):
        PSNR = 10 * log10(MAX_I^2 / MSE)
    """
    mse = torch.mean((pred_01 - target_01) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10((max_val**2) / mse)


def compute_ssim_explicit(
    pred_01: torch.Tensor,
    target_01: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
) -> torch.Tensor:
    """
    Explicit 2D Gaussian-windowed SSIM formula (Week 10 Slide 4):
        SSIM(x, y) = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)*(sigma_x^2 + sigma_y^2 + C2))
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    coords = torch.arange(win_size, dtype=pred_01.dtype, device=pred_01.device) - win_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    pad = win_size // 2
    mu_x = F.conv2d(pred_01, window, padding=pad)
    mu_y = F.conv2d(target_01, window, padding=pad)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred_01 * pred_01, window, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(target_01 * target_01, window, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(pred_01 * target_01, window, padding=pad) - mu_xy

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


class MedicalTranslationEvaluator:
    """
    Comprehensive evaluator using MONAI metrics (PSNRMetric, SSIMMetric, RMSEMetric, MAEMetric)
    along with region-specific error decomposition on BraTS 2021 test subjects.
    """

    def __init__(self, max_val: float = 1.0) -> None:
        self.max_val = max_val
        self.psnr_metric = PSNRMetric(max_val=max_val)
        self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=max_val)
        self.rmse_metric = RMSEMetric()
        self.mae_metric = MAEMetric()

    @torch.no_grad()
    def evaluate_batch(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        fake_A: torch.Tensor,
        rec_A: torch.Tensor,
        rec_B: torch.Tensor,
        seg: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Compute per-slice metrics for a batch of paired (T1ce, T2) slices."""
        rA = to_unit_range(real_A)
        rB = to_unit_range(real_B)
        fB = to_unit_range(fake_B)
        fA = to_unit_range(fake_A)
        cA = to_unit_range(rec_A)
        cB = to_unit_range(rec_B)

        # MONAI metrics for T1w -> T2w (Fake B vs Real B)
        psnr_B = self.psnr_metric(y_pred=fB, y=rB).squeeze(-1).cpu().numpy()
        ssim_B = self.ssim_metric(y_pred=fB, y=rB).squeeze(-1).cpu().numpy()
        rmse_B = self.rmse_metric(y_pred=fB, y=rB).squeeze(-1).cpu().numpy()
        mae_B = self.mae_metric(y_pred=fB, y=rB).squeeze(-1).cpu().numpy()

        # MONAI metrics for T2w -> T1w (Fake A vs Real A)
        psnr_A = self.psnr_metric(y_pred=fA, y=rA).squeeze(-1).cpu().numpy()
        ssim_A = self.ssim_metric(y_pred=fA, y=rA).squeeze(-1).cpu().numpy()
        rmse_A = self.rmse_metric(y_pred=fA, y=rA).squeeze(-1).cpu().numpy()
        mae_A = self.mae_metric(y_pred=fA, y=rA).squeeze(-1).cpu().numpy()

        # Cycle reconstruction PSNR (Rec A vs Real A, Rec B vs Real B)
        psnr_rec_A = self.psnr_metric(y_pred=cA, y=rA).squeeze(-1).cpu().numpy()
        psnr_rec_B = self.psnr_metric(y_pred=cB, y=rB).squeeze(-1).cpu().numpy()

        # Region-specific MAE analysis (Normal Brain vs Tumor vs Enhancing Core)
        seg_dev = seg.to(rA.device)
        brain_mask = (rA > 0.02) | (rB > 0.02)
        normal_mask = brain_mask & (seg_dev == 0)
        tumor_mask = seg_dev > 0
        edema_mask = seg_dev == 2
        enhancing_mask = seg_dev == 4

        abs_err_B = torch.abs(fB - rB)
        abs_err_A = torch.abs(fA - rA)

        def masked_mae(err: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
            out = []
            for i in range(err.shape[0]):
                m = mask[i]
                if m.sum() > 10:
                    out.append(float(err[i][m].mean().item()))
                else:
                    out.append(np.nan)
            return np.array(out, dtype=np.float64)

        return {
            "psnr_T2": psnr_B,
            "ssim_T2": ssim_B,
            "rmse_T2": rmse_B,
            "mae_T2": mae_B,
            "psnr_T1": psnr_A,
            "ssim_T1": ssim_A,
            "rmse_T1": rmse_A,
            "mae_T1": mae_A,
            "psnr_rec_T1": psnr_rec_A,
            "psnr_rec_T2": psnr_rec_B,
            "mae_T2_normal": masked_mae(abs_err_B, normal_mask),
            "mae_T2_tumor": masked_mae(abs_err_B, tumor_mask),
            "mae_T2_edema": masked_mae(abs_err_B, edema_mask),
            "mae_T2_enhancing": masked_mae(abs_err_B, enhancing_mask),
            "mae_T1_normal": masked_mae(abs_err_A, normal_mask),
            "mae_T1_tumor": masked_mae(abs_err_A, tumor_mask),
            "mae_T1_edema": masked_mae(abs_err_A, edema_mask),
            "mae_T1_enhancing": masked_mae(abs_err_A, enhancing_mask),
        }
