# Code Review: `thesis-work.ipynb`

## Summary
I reviewed the extracted Python code from `thesis-work.ipynb` (saved as `notebook_code.py`) to identify correctness, reliability, and maintainability issues. Below are the most pressing findings and suggested fixes.

## Findings
1. **Loss/label mismatch and double softmax**
   * The dataset builds one-hot labels, but the training loop feeds them to `nn.CrossEntropyLoss`, which expects integer class indices. Labels are also cast to `float()` and predictions are softmaxed inside the model, so the loss sees already-normalized probabilities instead of logits. This combination produces incorrect gradients and unstable training. Consider switching labels to integer indices (no one-hot), removing the model’s softmax, and using `nn.CrossEntropyLoss`; or keep one-hot targets but move to `nn.BCEWithLogitsLoss`/`nn.MultiLabelSoftMarginLoss` with raw logits. 【F:notebook_code.py†L64-L139】【F:notebook_code.py†L321-L377】【F:notebook_code.py†L575-L665】

2. **Conflicting model definitions**
   * `CNN3D` is defined twice; the second declaration overwrites the first at import time, making the earlier attention-enabled version dead code. This duplication obscures intent and makes it easy to call the wrong architecture. Remove or rename one of the definitions and keep a single, clearly documented model entry point. 【F:notebook_code.py†L321-L377】【F:notebook_code.py†L411-L439】

3. **Non-deterministic class mapping and split leakage risk**
   * Class IDs derive from `os.listdir` ordering, which is platform-dependent and changes run to run. The split logic also shuffles the full file list before generating train/val/test subsets without stratifying by class, which can skew label distribution. Sort `class_dir`, build `dict_class` deterministically, and use a stratified split (e.g., `train_test_split(..., stratify=labels)`) to preserve per-class balance. 【F:notebook_code.py†L45-L146】【F:notebook_code.py†L141-L150】

## Suggested Next Steps
* Refactor the training pipeline to align labels, model outputs, and loss functions.
* Consolidate the CNN architecture into a single, well-tested class definition.
* Make dataset construction deterministic and stratified to improve reproducibility and evaluation quality.

## Execution Note
I have not executed the notebook code in this environment because the dataset paths are not available in the repository and GPU resources are not configured. I can run sanity checks or small CPU-only tests if provided with accessible sample data and clear run scripts, but full training or evaluation is currently blocked by the missing inputs and hardware constraints.
