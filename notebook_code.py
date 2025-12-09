"""Cleaned training notebook code for 3D fMRI classification.

This script is designed to be run inside a notebook (or as a plain Python
script) on Kaggle. It focuses on a deterministic data pipeline, sensible
normalization, and a compact 3D CNN suited for small-to-medium datasets.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------
# Reproducibility helpers
# -----------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# Data pipeline
# -----------------------

def read_nifti_file(filepath: Path) -> np.ndarray:
    """Load a NIfTI volume as a float32 NumPy array."""
    scan = nib.load(str(filepath))
    data = scan.get_fdata().astype(np.float32)
    return data


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = volume.mean()
    std = volume.std()
    if std < eps:
        return volume - mean
    return (volume - mean) / (std + eps)


def discover_files(root: Path, extensions: Sequence[str] = (".nii", ".nii.gz")) -> List[Path]:
    files: List[Path] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ext in extensions:
            files.extend(sorted(class_dir.glob(f"*{ext}")))
    if not files:
        raise FileNotFoundError(f"No NIfTI files found under {root}")
    return files


def build_class_mapping(files: Sequence[Path]) -> dict[str, int]:
    classes = sorted({path.parent.name for path in files})
    return {cls: idx for idx, cls in enumerate(classes)}


def stratified_split(
    files: Sequence[Path],
    class_to_idx: dict[str, int],
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
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


# -----------------------
# Model
# -----------------------


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# -----------------------
# Training & evaluation
# -----------------------


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def mcc_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    return float(matthews_corrcoef(y_true, preds))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_mcc = 0.0
    criterion = nn.CrossEntropyLoss()

    for data, targets in tqdm(loader, desc="Train", leave=False):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_acc += accuracy_from_logits(logits, targets) * data.size(0)
        total_mcc += mcc_from_logits(logits, targets) * data.size(0)

    count = len(loader.dataset)
    return total_loss / count, total_acc / count, total_mcc / count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_mcc = 0.0

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Eval", leave=False):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = criterion(logits, targets)
            total_loss += loss.item() * data.size(0)
            total_acc += accuracy_from_logits(logits, targets) * data.size(0)
            total_mcc += mcc_from_logits(logits, targets) * data.size(0)

    count = len(loader.dataset)
    return total_loss / count, total_acc / count, total_mcc / count


# -----------------------
# Config & entry point
# -----------------------


@dataclass
class TrainConfig:
    data_dir: Path
    batch_size: int = 4
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2
    train_size: float = 0.7
    val_size: float = 0.15



def create_loaders(config: TrainConfig, files: Sequence[Path], class_to_idx: dict[str, int]):
    train_files, val_files, test_files = stratified_split(
        files, class_to_idx, train_size=config.train_size, val_size=config.val_size, seed=config.seed
    )

    train_ds = MRIDataset(train_files, class_to_idx)
    val_ds = MRIDataset(val_files, class_to_idx)
    test_ds = MRIDataset(test_files, class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple 3D CNN on fMRI volumes")
    parser.add_argument("data_dir", type=Path, help="Root directory containing class subfolders with NIfTI files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = discover_files(config.data_dir)
    class_to_idx = build_class_mapping(files)
    train_loader, val_loader, test_loader = create_loaders(config, files, class_to_idx)

    model = Simple3DCNN(num_classes=len(class_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_mcc = -math.inf
    best_state = None

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")
        train_loss, train_acc, train_mcc = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_mcc = evaluate(model, val_loader, device)

        print(
            f"Train loss {train_loss:.4f} | acc {train_acc:.4f} | mcc {train_mcc:.4f} | "
            f"Val loss {val_loss:.4f} | acc {val_acc:.4f} | mcc {val_mcc:.4f}"
        )

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_state = {"model": model.state_dict(), "class_to_idx": class_to_idx}

    if best_state is not None:
        torch.save(best_state, "best_model.pt")
        print(f"Saved best checkpoint with MCC {best_val_mcc:.4f} to best_model.pt")

    test_loss, test_acc, test_mcc = evaluate(model, test_loader, device)
    print(f"Test loss {test_loss:.4f} | acc {test_acc:.4f} | mcc {test_mcc:.4f}")


if __name__ == "__main__":
    main()
