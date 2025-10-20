# 🧠 Advanced Task-Based fMRI Classification — Hybrid 3D-CNN + Transformer with Attention (PyTorch)

This project investigates how **hybrid deep learning architectures** can enhance task-based fMRI classification by integrating **3D Convolutional Neural Networks (CNNs)** for local feature extraction with **Transformer encoders** for capturing long-range temporal dependencies. The work bridges deep learning and neuroscience to support **functional brain mapping**, **neuro-rehabilitation**, and **brain-computer interface (BCI)** applications.

---

## 🔍 Problem

Functional MRI (fMRI) data presents a unique challenge — it is **high-dimensional, noisy, and temporally complex**. Traditional CNNs can extract spatial features effectively but fail to capture temporal context across brain regions. The goal was to build a **scalable and interpretable model** that preserves spatial precision while understanding global temporal relationships in brain activation patterns.

---

## ⚙️ Action

* Developed a **hybrid deep learning architecture** combining 3D-CNNs with Transformer encoders using **multi-head self-attention** to model spatial-temporal dependencies.
* Implemented a **complete data pipeline** for 4D fMRI volumes (100×100×100×232) including zero-padding, normalization, voxel-wise BOLD analysis, and temporal alignment.
* Introduced **DropConnect regularization (p=0.17)** and optimized architectural design, reducing computation cost by over **40%** (FLOPs 262G vs. 465G).
* Conducted an **ablation study** comparing CNN-only, Transformer-only, and hybrid models to validate the efficiency of the proposed approach.
* Tracked performance using custom visualization scripts (ROC-AUC, PRC-AUC, F1, MCC) and built attention heatmaps to interpret learned spatial features.

---

## 🚀 Result

* **Accuracy:** 85.22% | **MCC:** 0.8055 | **ROC-AUC:** 0.95 | **PRC-AUC:** 0.88
* Outperformed baseline (DeepBrain) accuracy of 82.9% while reducing compute overhead by 40%.
* Attention visualizations revealed activation clusters aligned with **motor and sensory cortical regions**, confirming the model’s neuroscientific validity.
* Established a reproducible, open-source workflow for fMRI analysis — improving transparency and accelerating future neuroimaging research.

---

## 🧠 Challenges & Lessons Learned

1. **High Dimensionality:** Addressed memory bottlenecks via batch-wise loading, data compression, and optimized tensor operations.
2. **Noisy Data:** Integrated voxel-level normalization and temporal smoothing to reduce signal drift.
3. **Limited Samples:** Mitigated overfitting with DropConnect, weighted loss functions, and careful model pruning.
4. **Interpretability:** Developed visualization utilities for feature attribution and attention mapping to improve explainability.

---

## 🧩 Key Takeaways

This project demonstrates that combining **CNN-based spatial encoding** and **Transformer-based temporal modeling** can meaningfully improve both accuracy and interpretability in fMRI classification. The hybrid approach offers a path toward **real-time brain state decoding** with applications in **neuroimaging, clinical diagnostics, and cognitive science**.

---

## 🧮 Tools & Libraries

* **Frameworks:** PyTorch, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Data Handling:** NiBabel, TorchIO, pandas
* **Visualization:** ROC, PRC, saliency, and attention heatmaps
* **Environment:** CUDA 12.2 (NVIDIA RTX GPU)

---

## 📊 Repository Structure

```
├── data_preprocessing/        # fMRI data loading, normalization, and signal alignment
├── models/                    # CNN, Transformer, and hybrid architectures
├── training/                  # Training scripts, evaluation metrics, early stopping
├── visualization/              # Attention maps, ROC, and PRC visualizations
├── notebooks/                 # Experiment notebooks
└── README.md                  # This documentation
```

---

## 🧩 Future Work

* Implement **contrastive pretraining** for unsupervised representation learning.
* Explore **cross-dataset generalization** to improve robustness.
* Integrate results into real-time **BCI decoding frameworks**.

---

## 💬 Citation & Acknowledgments

If you reference this repository, please cite the original work.
This research was independently developed to advance **interpretable and resource-efficient** fMRI classification models.

---

⭐ **Keywords:** fMRI • 3D-CNN • Transformer • Self-Attention • Deep Learning • Neuroimaging • PyTorch • Brain Mapping • Data Preprocessing • Interpretability
