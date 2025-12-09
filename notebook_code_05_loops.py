"""Training and evaluation loops for the fMRI classifier."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from notebook_code_04_metrics import accuracy_from_logits, mcc_from_logits


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
