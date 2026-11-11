"""
MONAI-based Dataset and Preprocessing Pipeline for BraTS 2021 T1ce <-> T2 Translation.

Implements:
1. Strict subject-level separation into Train (15), Validation (3), and Fixed Test (4) sets.
2. MONAI transforms for NIfTI loading, outlier-robust percentile intensity scaling (or min-max),
   foreground center cropping, and spatial resizing to 128x128.
3. Unpaired random shuffling of T1ce and T2 slices during training (as required by assignment).
4. Paired slice alignment for validation and test quantitative evaluation.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from monai.data import DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    Resized,
)

# Fixed test set mandated by Week 11 Slide 20
TEST_SUBJECTS: List[str] = [
    "BraTS2021_00025",
    "BraTS2021_00026",
    "BraTS2021_00028",
    "BraTS2021_00030",
]

# Validation set (3 subjects separated from non-test data)
VAL_SUBJECTS: List[str] = [
    "BraTS2021_00021",
    "BraTS2021_00022",
    "BraTS2021_00024",
]

# Training set (remaining 15 subjects)
TRAIN_SUBJECTS: List[str] = [
    "BraTS2021_00000",
    "BraTS2021_00002",
    "BraTS2021_00003",
    "BraTS2021_00005",
    "BraTS2021_00006",
    "BraTS2021_00008",
    "BraTS2021_00009",
    "BraTS2021_00011",
    "BraTS2021_00012",
    "BraTS2021_00014",
    "BraTS2021_00016",
    "BraTS2021_00017",
    "BraTS2021_00018",
    "BraTS2021_00019",
    "BraTS2021_00020",
]

# Axial slice range containing brain parenchyma & tumor structures (90 slices/subject)
DEFAULT_SLICE_RANGE: Tuple[int, int] = (30, 120)


def normalize_volume(
    vol: np.ndarray,
    mode: str = "percentile",
    upper_pct: float = 99.5,
) -> np.ndarray:
    """
    Normalize 3D MRI volume intensity to [-1.0, 1.0] for Tanh generator output.

    Args:
        vol: 3D numpy array of MRI intensities.
        mode: 'percentile' (clips hyper-intense outliers at 99.5th percentile)
              or 'minmax' (naive min-max scaling sensitive to contrast spikes).
        upper_pct: Upper percentile cutoff when mode == 'percentile'.
    """
    vol = vol.astype(np.float32)
    vmin = 0.0
    if mode == "percentile":
        nonzero = vol[vol > 0]
        vmax = float(np.percentile(nonzero, upper_pct)) if nonzero.size > 0 else float(vol.max())
    elif mode == "minmax":
        vmax = float(vol.max())
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    if vmax <= vmin:
        return np.full_like(vol, -1.0, dtype=np.float32)

    clipped = np.clip(vol, vmin, vmax)
    scaled_01 = (clipped - vmin) / (vmax - vmin)  # [0, 1]
    return scaled_01 * 2.0 - 1.0  # [-1, 1]


def load_subject_slices(
    data_root: str | Path,
    subjects: List[str],
    slice_range: Tuple[int, int] = DEFAULT_SLICE_RANGE,
    norm_mode: str = "percentile",
    image_size: int = 128,
    crop_size: int = 192,
) -> List[Dict[str, torch.Tensor | str | int]]:
    """
    Load 3D NIfTI volumes via MONAI LoadImaged, normalize intensities, extract 2D axial
    slices, and apply MONAI spatial center-cropping and resizing to (1, image_size, image_size).
    """
    data_root = Path(data_root)
    vol_loader = Compose([
        LoadImaged(keys=["t1", "t2", "seg"], image_only=True),
        EnsureChannelFirstd(keys=["t1", "t2", "seg"], channel_dim="no_channel"),
    ])
    spatial_transform = Compose([
        CenterSpatialCropd(keys=["t1", "t2", "seg"], roi_size=(crop_size, crop_size)),
        Resized(
            keys=["t1", "t2", "seg"],
            spatial_size=(image_size, image_size),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        EnsureTyped(keys=["t1", "t2", "seg"], dtype=torch.float32),
    ])

    slices_list: List[Dict[str, torch.Tensor | str | int]] = []
    z_start, z_end = slice_range

    for subj in subjects:
        subj_dir = data_root / subj
        vol_dict = {
            "t1": str(subj_dir / f"{subj}_t1ce.nii.gz"),
            "t2": str(subj_dir / f"{subj}_t2.nii.gz"),
            "seg": str(subj_dir / f"{subj}_seg.nii.gz"),
        }
        loaded = vol_loader(vol_dict)
        # Convert to numpy (1, H, W, D) -> normalize 3D volume
        t1_vol = normalize_volume(np.asarray(loaded["t1"][0]), mode=norm_mode)
        t2_vol = normalize_volume(np.asarray(loaded["t2"][0]), mode=norm_mode)
        seg_vol = np.asarray(loaded["seg"][0], dtype=np.float32)

        for z in range(z_start, z_end):
            # Skip nearly empty slices
            if (t1_vol[:, :, z] > -0.95).sum() < 400:
                continue
            # Rotate 90 degrees for standard radiological axial orientation
            s_t1 = np.rot90(t1_vol[:, :, z])[np.newaxis, ...].copy()
            s_t2 = np.rot90(t2_vol[:, :, z])[np.newaxis, ...].copy()
            s_seg = np.rot90(seg_vol[:, :, z])[np.newaxis, ...].copy()

            transformed = spatial_transform({"t1": s_t1, "t2": s_t2, "seg": s_seg})
            slices_list.append({
                "t1": transformed["t1"].clamp(-1.0, 1.0),
                "t2": transformed["t2"].clamp(-1.0, 1.0),
                "seg": transformed["seg"].long(),
                "subject": subj,
                "slice_idx": z,
            })

    return slices_list


class UnpairedBraTSDataset(TorchDataset):
    """
    Unpaired training dataset for CycleGAN using MONAI augmentations.
    Shuffles Domain A (T1ce) and Domain B (T2) independently so that training pairs
    (real_A, real_B) are strictly unpaired across subjects and axial slice positions.
    """

    def __init__(
        self,
        slices_data: List[Dict[str, torch.Tensor | str | int]],
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.t1_slices = [item["t1"] for item in slices_data]
        self.t2_slices = [item["t2"] for item in slices_data]
        self.t1_meta = [(item["subject"], item["slice_idx"]) for item in slices_data]
        self.t2_meta = [(item["subject"], item["slice_idx"]) for item in slices_data]

        self.rng = random.Random(seed)
        self.perm_A = list(range(len(self.t1_slices)))
        self.perm_B = list(range(len(self.t2_slices)))
        self.shuffle_domains()

        self.augment = augment
        self.aug_A = RandFlipd(keys=["img"], prob=0.5, spatial_axis=1)
        self.aug_B = RandFlipd(keys=["img"], prob=0.5, spatial_axis=1)

    def shuffle_domains(self) -> None:
        """Independently shuffle indices of Domain A (T1ce) and Domain B (T2)."""
        self.rng.shuffle(self.perm_A)
        self.rng.shuffle(self.perm_B)

    def __len__(self) -> int:
        return len(self.t1_slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        idx_A = self.perm_A[idx % len(self.perm_A)]
        idx_B = self.perm_B[idx % len(self.perm_B)]
        # Ensure we do not accidentally pick the exact same slice index
        if idx_A == idx_B and len(self.perm_B) > 1:
            idx_B = self.perm_B[(idx + 1) % len(self.perm_B)]

        real_A = self.t1_slices[idx_A]
        real_B = self.t2_slices[idx_B]

        if self.augment:
            real_A = self.aug_A({"img": real_A})["img"]
            real_B = self.aug_B({"img": real_B})["img"]

        return {
            "A": real_A,
            "B": real_B,
            "subject_A": self.t1_meta[idx_A][0],
            "slice_A": self.t1_meta[idx_A][1],
            "subject_B": self.t2_meta[idx_B][0],
            "slice_B": self.t2_meta[idx_B][1],
        }


class PairedBraTSDataset(TorchDataset):
    """
    Paired dataset for validation and test evaluation.
    Preserves exact anatomical correspondence between Real T1w (A) and Real T2w (B)
    to compute ground-truth PSNR, SSIM, RMSE, and MAE metrics.
    """

    def __init__(self, slices_data: List[Dict[str, torch.Tensor | str | int]]) -> None:
        super().__init__()
        self.slices_data = slices_data

    def __len__(self) -> int:
        return len(self.slices_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        item = self.slices_data[idx]
        return {
            "A": item["t1"],
            "B": item["t2"],
            "seg": item["seg"],
            "subject": item["subject"],
            "slice_idx": item["slice_idx"],
        }


def build_dataloaders(
    data_root: str | Path = "AI4MIA_Assignment4_Dataset",
    batch_size: int = 8,
    norm_mode: str = "percentile",
    image_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, UnpairedBraTSDataset]:
    """Build MONAI DataLoaders for train (unpaired), validation (paired), and test (paired)."""
    train_slices = load_subject_slices(
        data_root, TRAIN_SUBJECTS, norm_mode=norm_mode, image_size=image_size
    )
    val_slices = load_subject_slices(
        data_root, VAL_SUBJECTS, norm_mode=norm_mode, image_size=image_size
    )
    test_slices = load_subject_slices(
        data_root, TEST_SUBJECTS, norm_mode=norm_mode, image_size=image_size
    )

    train_ds = UnpairedBraTSDataset(train_slices, augment=True, seed=seed)
    val_ds = PairedBraTSDataset(val_slices)
    test_ds = PairedBraTSDataset(test_slices)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_ds
