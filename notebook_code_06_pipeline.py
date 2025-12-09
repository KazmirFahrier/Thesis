"""Pipeline orchestration for training the fMRI classifier."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from notebook_code_01_seed import set_seed
from notebook_code_02_data import MRIDataset, build_class_mapping, discover_files, stratified_split
from notebook_code_03_model import Simple3DCNN
from notebook_code_05_loops import evaluate, run_epoch


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


def train_pipeline(
    data_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_workers: int = 2,
    seed: int = 42,
) -> dict[str, float]:
    """Convenience wrapper for running training from a notebook."""

    config = TrainConfig(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        seed=seed,
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

    return {"test_loss": test_loss, "test_acc": test_acc, "test_mcc": test_mcc}


def cli_main() -> None:
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
