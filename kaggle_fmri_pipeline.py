"""
Minimal, Kaggle-friendly training script for 3D fMRI classification.
- Deterministic class indexing and stratified splits.
- Integer labels matched to CrossEntropyLoss (no in-model softmax).
- Single, concise 3D CNN backbone with optional dropout.

Usage (inside Kaggle notebook/script):
    !python kaggle_fmri_pipeline.py \
        --data-root /kaggle/input/your_dataset \
        --epochs 5 --batch-size 2 --lr 3e-4

Expected directory structure:
/data_root/
    class_a/
        sample1.nii.gz
        sample2.nii.gz
    class_b/
        ...

The script prints train/val metrics and saves the best model weights
(`best_model.pt`) based on validation MCC.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ----------------------------- Reproducibility ----------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------ Data handling ------------------------------ #
class ZScoreNormalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std().clamp(min=1e-6)
        return (tensor - mean) / std


def load_nifti(path: str) -> torch.Tensor:
    """Load a NIfTI volume as a float32 tensor (D, H, W)."""
    volume = nib.load(path).get_fdata().astype(np.float32)
    return torch.from_numpy(volume)


class FMriDataset(Dataset):
    def __init__(
        self,
        file_paths: Sequence[str],
        labels: Sequence[int],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        label = self.labels[idx]
        volume = load_nifti(path)
        if self.transform:
            volume = self.transform(volume)
        # add channel dimension for Conv3d
        volume = volume.unsqueeze(0)
        return volume, label


def index_dataset(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    """Return file paths, integer labels, and ordered class names."""
    class_names = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    class_names.sort()
    file_paths: List[str] = []
    labels: List[int] = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_root, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".nii", ".nii.gz")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)
    if not file_paths:
        raise ValueError(f"No NIfTI files found under {data_root}.")
    return file_paths, labels, class_names


def build_splits(
    file_paths: Sequence[str],
    labels: Sequence[int],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        labels,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


# --------------------------------- Model ---------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-3)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        return self.dropout(x)


class FMriCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.stem = ConvBlock(in_channels, 32, kernel=5, stride=2, dropout=dropout)
        self.block1 = ConvBlock(32, 64, kernel=5, stride=2, dropout=dropout)
        self.block2 = ConvBlock(64, 128, kernel=3, stride=2, dropout=dropout)
        self.block3 = ConvBlock(128, 256, kernel=3, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


# ----------------------------- Train / Evaluate --------------------------- #
@dataclass
class TrainConfig:
    data_root: str
    epochs: int = 5
    batch_size: int = 2
    lr: float = 3e-4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def mcc(preds: torch.Tensor, labels: torch.Tensor) -> float:
    # simple, non-vectorized Matthew's correlation coefficient
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / (denom ** 0.5)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    running_acc = 0.0
    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels = labels.to(device)
        logits = model(volumes)
        loss = criterion(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * volumes.size(0)
        preds = torch.argmax(logits, dim=1)
        running_acc += accuracy(preds, labels) * volumes.size(0)
    total = len(loader.dataset)
    return running_loss / total, running_acc / total


def evaluate_mcc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds_all: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device)
            logits = model(volumes)
            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.tolist())
    preds_tensor = torch.tensor(preds_all)
    labels_tensor = torch.tensor(labels_all)
    return mcc(preds_tensor, labels_tensor)


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    file_paths, labels, class_names = index_dataset(config.data_root)
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = build_splits(
        file_paths, labels, config.val_ratio, config.test_ratio, config.seed
    )

    transform = ZScoreNormalize()
    train_ds = FMriDataset(train_paths, train_labels, transform)
    val_ds = FMriDataset(val_paths, val_labels, transform)
    test_ds = FMriDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    device = torch.device(config.device)
    model = FMriCNN(in_channels=1, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    best_mcc = -1.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        val_mcc = evaluate_mcc(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_mcc={val_mcc:.3f}"
        )
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save({"model_state": model.state_dict(), "class_names": class_names}, "best_model.pt")
            print(f"Saved new best model with val_mcc={best_mcc:.3f}")

    # Final test evaluation using the best checkpoint
    if os.path.exists("best_model.pt"):
        checkpoint = torch.load("best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print("Loaded best validation checkpoint for test evaluation.")
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    test_mcc = evaluate_mcc(model, test_loader, device)
    print(
        f"Test metrics: loss={test_loss:.4f} accuracy={test_acc:.3f} mcc={test_mcc:.3f} across {len(test_ds)} samples"
    )


# ----------------------------------- CLI ---------------------------------- #
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a lightweight 3D CNN for fMRI classification on Kaggle.")
    parser.add_argument("--data-root", required=True, help="Path to dataset root with one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu). Defaults to cuda if available.",
    )
    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
