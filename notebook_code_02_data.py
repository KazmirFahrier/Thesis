"""Data loading utilities for NIfTI volumes and dataset splits."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset


def read_nifti_file(filepath: Path) -> np.ndarray:
    """Load a NIfTI volume as a float32 NumPy array."""

    scan = nib.load(str(filepath))
    data = scan.get_fdata().astype(np.float32)
    return data


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-volume z-score normalization with epsilon for stability."""

    mean = volume.mean()
    std = volume.std()
    if std < eps:
        return volume - mean
    return (volume - mean) / (std + eps)


def discover_files(root: Path, extensions: Sequence[str] = (".nii", ".nii.gz")) -> List[Path]:
    """Return all NIfTI files nested one directory below the root."""

    files: List[Path] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ext in extensions:
            files.extend(sorted(class_dir.glob(f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found under {root}")
    return files


def build_class_mapping(files: Sequence[Path]) -> dict[str, int]:
    """Create a deterministic label-to-index mapping from file paths."""

    classes = sorted({path.parent.name for path in files})
    return {cls: idx for idx, cls in enumerate(classes)}


def stratified_split(
    files: Sequence[Path],
    class_to_idx: dict[str, int],
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split files into train/val/test sets while preserving class balance."""

    labels = [class_to_idx[path.parent.name] for path in files]
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(splitter.split(files, labels))

    temp_files = [files[i] for i in temp_idx]
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = val_size / (1.0 - train_size)
    temp_splitter = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
    val_idx, test_idx = next(temp_splitter.split(temp_files, temp_labels))

    train_files = [files[i] for i in train_idx]
    val_files = [temp_files[i] for i in val_idx]
    test_files = [temp_files[i] for i in test_idx]
    return train_files, val_files, test_files


class MRIDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapping NIfTI volumes and integer labels."""

    def __init__(self, files: Sequence[Path], class_to_idx: dict[str, int]):
        self.files = list(files)
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        volume = read_nifti_file(path)
        volume = zscore_normalize(volume)
        tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)

        label_idx = self.class_to_idx[path.parent.name]
        label = torch.tensor(label_idx, dtype=torch.long)
        return tensor, label
