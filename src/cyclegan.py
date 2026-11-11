"""
CycleGAN Model and Training Engine for Unpaired Brain MRI Translation.

Follows default hyperparameter settings from `junyanz/pytorch-CycleGAN-and-pix2pix`:
- Adversarial objective: LSGAN (MSELoss)
- Cycle-consistency loss: L1Loss with lambda_A = 10.0, lambda_B = 10.0
- Identity mapping loss: L1Loss with lambda_identity = 0.5
- Optimizer: Adam (lr = 0.0002, beta1 = 0.5, beta2 = 0.999)
- LR Scheduler: Constant for `n_epochs`, then linear decay to 0 over `n_epochs_decay`
- ImagePool buffer size: 50
- Optional MONAI SSIMLoss regularization for structural boundary preservation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from monai.losses import SSIMLoss

from src.metrics import MedicalTranslationEvaluator, to_unit_range
from src.networks import GANLoss, ImagePool, define_D, define_G


@dataclass
class CycleGANConfig:
    """Configuration matching default settings of pytorch-CycleGAN-and-pix2pix."""

    netG: str = "resnet_6blocks"  # 'resnet_6blocks' | 'unet_128' | 'monai_resunet'
    ngf: int = 64
    ndf: int = 64
    n_layers_D: int = 3
    norm: str = "instance"
    use_dropout: bool = False
    init_type: str = "normal"
    init_gain: float = 0.02
    gan_mode: str = "lsgan"
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_identity: float = 0.5
    lambda_ssim: float = 0.0  # > 0 enables MONAI SSIM structural cycle loss
    pool_size: int = 50
    lr: float = 0.0002
    beta1: float = 0.5
    n_epochs: int = 25
    n_epochs_decay: int = 25
    device: str = "cuda"


class CycleGANModel:
    """Complete CycleGAN model encapsulating G_A, G_B, D_A, D_B, losses, and optimizers."""

    def __init__(self, cfg: CycleGANConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # Generators: G_A (T1w -> T2w), G_B (T2w -> T1w)
        self.netG_A = define_G(
            cfg.netG, 1, 1, cfg.ngf, cfg.norm, cfg.use_dropout, cfg.init_type, cfg.init_gain, self.device
        )
        self.netG_B = define_G(
            cfg.netG, 1, 1, cfg.ngf, cfg.norm, cfg.use_dropout, cfg.init_type, cfg.init_gain, self.device
        )

        # Discriminators: D_A (evaluates T2w), D_B (evaluates T1w)
        self.netD_A = define_D(1, cfg.ndf, cfg.n_layers_D, cfg.norm, cfg.init_type, cfg.init_gain, self.device)
        self.netD_B = define_D(1, cfg.ndf, cfg.n_layers_D, cfg.norm, cfg.init_type, cfg.init_gain, self.device)

        # Image history buffers
        self.fake_A_pool = ImagePool(cfg.pool_size)
        self.fake_B_pool = ImagePool(cfg.pool_size)

        # Loss functions
        self.criterionGAN = GANLoss(cfg.gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
        self.criterionSSIM = SSIMLoss(spatial_dims=2, data_range=1.0) if cfg.lambda_ssim > 0 else None

        # Optimizers (default CycleGAN Adam settings)
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            lr=cfg.lr,
            betas=(cfg.beta1, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr=cfg.lr,
            betas=(cfg.beta1, 0.999),
        )

        # Linear decay schedulers
        def lambda_rule(epoch: int) -> float:
            lr_l = 1.0 - max(0, epoch - cfg.n_epochs) / float(cfg.n_epochs_decay + 1)
            return max(0.0, lr_l)

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)

    @staticmethod
    def set_requires_grad(nets: List[nn.Module], requires_grad: bool = False) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def optimize_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> Dict[str, float]:
        """Run one forward + backward update for Generators and Discriminators."""
        real_A = real_A.to(self.device, non_blocking=True)
        real_B = real_B.to(self.device, non_blocking=True)

        # Forward cycle: A -> B -> A and B -> A -> B
        fake_B = self.netG_A(real_A)  # G_A(A): Synthetic T2w
        rec_A = self.netG_B(fake_B)   # G_B(G_A(A)): Reconstructed T1w
        fake_A = self.netG_B(real_B)  # G_B(B): Synthetic T1w
        rec_B = self.netG_A(fake_A)   # G_A(G_B(B)): Reconstructed T2w

        # -------------------
        # 1. Update G_A and G_B
        # -------------------
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()

        # Identity loss: G_A(B) should be B, G_B(A) should be A
        if self.cfg.lambda_identity > 0:
            idt_A = self.netG_A(real_B)
            loss_idt_A = self.criterionIdt(idt_A, real_B) * self.cfg.lambda_B * self.cfg.lambda_identity
            idt_B = self.netG_B(real_A)
            loss_idt_B = self.criterionIdt(idt_B, real_A) * self.cfg.lambda_A * self.cfg.lambda_identity
        else:
            loss_idt_A = torch.tensor(0.0, device=self.device)
            loss_idt_B = torch.tensor(0.0, device=self.device)

        # GAN loss: D_A(G_A(A)) and D_B(G_B(B))
        loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)

        # Cycle consistency loss
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.cfg.lambda_A
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.cfg.lambda_B

        # Optional MONAI SSIM structural loss on cycle reconstructions
        if self.criterionSSIM is not None and self.cfg.lambda_ssim > 0:
            ssim_A = self.criterionSSIM(to_unit_range(rec_A), to_unit_range(real_A))
            ssim_B = self.criterionSSIM(to_unit_range(rec_B), to_unit_range(real_B))
            loss_ssim = (ssim_A + ssim_B) * self.cfg.lambda_ssim
        else:
            loss_ssim = torch.tensor(0.0, device=self.device)

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_ssim
        loss_G.backward()
        self.optimizer_G.step()

        # -------------------
        # 2. Update D_A and D_B
        # -------------------
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()

        # D_A loss (T2w domain)
        fake_B_pooled = self.fake_B_pool.query(fake_B.detach())
        loss_D_A = 0.5 * (
            self.criterionGAN(self.netD_A(real_B), True)
            + self.criterionGAN(self.netD_A(fake_B_pooled), False)
        )
        loss_D_A.backward()

        # D_B loss (T1w domain)
        fake_A_pooled = self.fake_A_pool.query(fake_A.detach())
        loss_D_B = 0.5 * (
            self.criterionGAN(self.netD_B(real_A), True)
            + self.criterionGAN(self.netD_B(fake_A_pooled), False)
        )
        loss_D_B.backward()
        self.optimizer_D.step()

        return {
            "G_total": float(loss_G.item()),
            "G_A": float(loss_G_A.item()),
            "G_B": float(loss_G_B.item()),
            "cycle_A": float(loss_cycle_A.item()),
            "cycle_B": float(loss_cycle_B.item()),
            "idt": float((loss_idt_A + loss_idt_B).item()),
            "ssim_loss": float(loss_ssim.item()),
            "D_A": float(loss_D_A.item()),
            "D_B": float(loss_D_B.item()),
        }

    def step_schedulers(self) -> None:
        self.scheduler_G.step()
        self.scheduler_D.step()

    @torch.no_grad()
    def evaluate(
        self, dataloader: torch.utils.data.DataLoader, evaluator: MedicalTranslationEvaluator
    ) -> Dict[str, float]:
        """Evaluate model on paired validation/test DataLoader."""
        self.netG_A.eval()
        self.netG_B.eval()

        agg: Dict[str, List[np.ndarray]] = {}
        for batch in dataloader:
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            seg = batch["seg"].to(self.device)

            fake_B = self.netG_A(real_A)
            rec_A = self.netG_B(fake_B)
            fake_A = self.netG_B(real_B)
            rec_B = self.netG_A(fake_A)

            batch_metrics = evaluator.evaluate_batch(real_A, real_B, fake_B, fake_A, rec_A, rec_B, seg)
            for k, v in batch_metrics.items():
                agg.setdefault(k, []).append(v)

        self.netG_A.train()
        self.netG_B.train()

        summary: Dict[str, float] = {}
        for k, arr_list in agg.items():
            concat = np.concatenate(arr_list, axis=0)
            summary[f"{k}_mean"] = float(np.nanmean(concat))
            summary[f"{k}_std"] = float(np.nanstd(concat))
        return summary

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "netG_A": self.netG_A.state_dict(),
                "netG_B": self.netG_B.state_dict(),
                "netD_A": self.netD_A.state_dict(),
                "netD_B": self.netD_B.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.netG_A.load_state_dict(ckpt["netG_A"])
        self.netG_B.load_state_dict(ckpt["netG_B"])
        self.netD_A.load_state_dict(ckpt["netD_A"])
        self.netD_B.load_state_dict(ckpt["netD_B"])
