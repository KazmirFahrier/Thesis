# 🧠 Hybrid 3D CNN + Transformer Model for Task-Based fMRI Classification

This repository contains the official implementation of a hybrid deep learning architecture designed to classify task-based fMRI data using a combination of **3D Convolutional Neural Networks (3D CNNs)** and **Transformer encoders** with attention mechanisms.

The model, developed as part of a master's thesis, demonstrates superior performance in decoding brain activity patterns associated with somatotopic motor tasks. This work has been submitted as a journal paper titled:  
**"Advanced Task-Based fMRI Classification Using a Hybrid 3D CNN and Transformer Model with Attention Mechanisms"**

---

## 🧠 Project Summary

Functional MRI (fMRI) data, while rich in temporal and spatial information, poses significant challenges for classification due to its high dimensionality and variability. To address these, this project introduces a lightweight, scalable architecture that leverages:

- **3D CNNs** to capture fine-grained spatial features from voxel-based inputs.
- **3D Self-Attention Blocks** to enhance local feature representation.
- **Transformer Encoders** to model long-range dependencies and contextual relevance.
- **DropConnect & Dropout** for enhanced regularization and generalization.

The model achieves an impressive **85.22% accuracy**, surpassing state-of-the-art baselines such as DeepBrain.

---

## 🔍 Dataset and Classification Task

The model was trained and evaluated on a subset of publicly available task-fMRI data focused on somatotopic motor responses. The classification task involved identifying one of four targeted classes:

- **Left Leg**
- **Right Leg**
- **Upper Arm**
- **Forearm**

Preprocessing, slicing, and one-hot encoding techniques were applied to create consistent voxel-based representations suitable for deep learning.

---

## 🏗️ Model Architecture Highlights

- **Input Layer**: 4D fMRI volume
- **3D CNN Stack**: For local spatial extraction with Instance Normalization
- **Self-Attention Block**: 3D Query-Key-Value with multi-head attention
- **Transformer Encoder**: 6-layer sequence processing with dropout
- **Classification Head**: Fully connected + Softmax

The design emphasizes both **local texture learning** and **global context modeling**, resulting in a high-performance hybrid pipeline.

---

## 📈 Performance

| Metric   | Result  |
|----------|---------|
| Accuracy | 85.22%  |
| Stability | Robust across folds |
| Comparison | Outperformed DeepBrain and other baseline models |

Additional metrics (F1 score, ROC-AUC, confusion matrix) are included in the Jupyter notebook for detailed evaluation.

---

