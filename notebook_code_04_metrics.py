"""Metric helpers for evaluating classifier performance."""

from __future__ import annotations

import torch
from sklearn.metrics import matthews_corrcoef


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def mcc_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    return float(matthews_corrcoef(y_true, preds))
