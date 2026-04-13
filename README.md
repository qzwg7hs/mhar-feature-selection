# Automated Feature Selection for Multimodal Human Activity Recognition Using Metaheuristic Optimization Algorithms

---

## Overview

This repository contains the full implementation of a metaheuristic-based automated feature selection framework for **Multimodal Human Activity Recognition (MHAR)**. The framework wraps 14 nature-inspired optimization algorithms around a neural network classifier to automatically identify near-optimal feature subsets — without requiring a manually predefined feature count.

The framework is evaluated on two benchmark MHAR datasets across **50+ experimental configurations**, including full multimodal, single-modality, multi-modality subset, and simulated data corruption settings. All experiments follow a strict **Leave-One-Subject-Out (LOSO)** cross-validation protocol to avoid data leakage.

---

## Key Results

### Main Results — Full Feature Set, All Modalities

| Method | UTD-MHAD V1 | UTD-MHAD V2 | CZU-MHAD V1 | CZU-MHAD V2 |
|---|---|---|---|---|
| Baseline (no selection) | 85.71 | 93.39 | 91.26 | 94.16 |
| Mutual Information | 74.89 | 93.97 | 93.77 | 96.30 |
| RFE | 91.41 | 94.55 | 93.23 | 95.14 |
| Lasso / L1 | 81.76 | 87.01 | 89.12 | 92.35 |
| **Combined Metaheuristic** | **92.21** | **96.99** | **94.82** | **95.80** |

*V1 = statistical features; V2 = extended features. All results are mean LOSO test accuracy (%).*

### Two-Layer Feature Selection (Best Results)

The two-layer cascaded approach (per-modality selection → global selection) achieves the best overall accuracy-compactness trade-off:

| Configuration | Accuracy (%) | Feature Retention (%) |
|---|---|---|
| UTD-MHAD V2 — Combined Meta | **97.45** | 20.03 |
| CZU-MHAD V1 — Combined Meta | 95.01 | 21.08 |

### Cross-Dataset Generalization

Models trained on UTD-MHAD and tested on unseen datasets **without any retraining or tuning**:

| Transfer | Baseline | Best Metaheuristic | Features Used |
|---|---|---|---|
| UTD-MHAD → CZU-MHAD | 55.06% | 65.19% (SCA) | 11.1% |
| UTD-MHAD → MSR-Action3D | 42.34% | 57.66% (WOA) | 25.0% |

---

## Repository Structure

```
.
├── UTD-MHAD/
│   ├── Feature extraction scripts and notebooks
│   ├── Training and evaluation notebooks (.ipynb)
│   └── Extracted feature files
│
├── CZU-MHAD/
│   ├── Feature extraction scripts and notebooks
│   ├── Training and evaluation notebooks (.ipynb)
│   └── Extracted feature files
│
└── MSR-Action3D/
    ├── Feature extraction scripts and notebooks
    ├── Training and evaluation notebooks (.ipynb)
    └── Extracted feature files
```

---

## Datasets

| Dataset | Modalities | Subjects | Action Classes |
|---|---|---|---|
| [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) | RGB, Depth, Skeleton, Inertial | 8 | 27 |
| [CZU-MHAD](https://github.com/yujmo/CZU_MHAD) | Depth, Skeleton, Inertial | 5 | 22 |
| [MSR-Action3D](https://sites.google.com/view/wanqingli/data-sets/msr-action3d) | Depth, Skeleton | 10 | 20 |

Datasets are **not included** in this repository due to size and licensing. Download them from the links above and place them in the corresponding folders before running feature extraction.

---

## Method

The proposed framework treats feature selection as a **binary combinatorial optimization problem**. Each candidate solution is a binary mask over the full feature vector, evaluated using a fitness function that trades classification accuracy against feature compactness:

```
F(sol) = 0.95 × (1 − val_accuracy) + 0.05 × (|selected| / |total|)
```

**14 metaheuristic algorithms** from the [EvoloPy](https://github.com/7ossam81/EvoloPy) framework are evaluated:

`BAT · CS · DE · FFA · GA · GWO · HHO · JAYA · MFO · MVO · PSO · SCA · SSA · WOA`

The classifier is an **adaptive MLP** whose hidden layer widths scale with the number of selected features. Feature selection and model training are performed independently within each LOSO fold to prevent data leakage.

---

## Setup

**Main dependencies:** Python 3.8+, PyTorch, scikit-learn, NumPy, Pandas, EvoloPy

---

This project is released for academic and research use. Please contact the author for any other use cases.
