"""Modular fMRI training notebook code with numbered helper modules.

This file re-exports the key pieces so it can still be imported or run
as `python notebook_code.py`, while the logic lives in the numbered
helpers for easier debugging.
"""

from __future__ import annotations

from notebook_code_01_seed import set_seed
from notebook_code_02_data import (
    MRIDataset,
    build_class_mapping,
    discover_files,
    read_nifti_file,
    stratified_split,
    zscore_normalize,
)
from notebook_code_03_model import Simple3DCNN
from notebook_code_04_metrics import accuracy_from_logits, mcc_from_logits
from notebook_code_05_loops import evaluate, run_epoch
from notebook_code_06_pipeline import TrainConfig, cli_main, create_loaders, train_pipeline

__all__ = [
    "TrainConfig",
    "MRIDataset",
    "Simple3DCNN",
    "accuracy_from_logits",
    "build_class_mapping",
    "cli_main",
    "create_loaders",
    "discover_files",
    "evaluate",
    "mcc_from_logits",
    "read_nifti_file",
    "run_epoch",
    "set_seed",
    "stratified_split",
    "train_pipeline",
    "zscore_normalize",
]


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
