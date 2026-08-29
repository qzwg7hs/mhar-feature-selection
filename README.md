# MHAR Feature Selection

Code for the paper **"Adaptive Feature Selection for Robust and Generalizable Multimodal Human Activity Recognition Using Metaheuristic Optimization Algorithms"** — Aruay Amangeldi, Tayfun Kucukyilmaz, Adnan Yazici (preprint submitted to Elsevier).

The framework casts feature selection as a binary combinatorial optimization problem and, rather than committing to a single optimizer, selects the best-performing metaheuristic **per leave-one-subject-out (LOSO) fold** from 14 nature-inspired algorithms in the [EvoloPy](https://github.com/7ossam81/EvoloPy) toolkit (BAT, CS, DE, FFA, GA, GWO, HHO, JAYA, MFO, MVO, PSO, SCA, SSA, WOA).

## Datasets

| Folder      | Dataset    | Modalities used                    |
|-------------|------------|------------------------------------|
| `czu-mhad/` | CZU-MHAD   | depth, skeleton, inertial          |
| `mmfit/`    | MM-Fit     | inertial, skeleton                 |
| `utd-mhad/` | UTD-MHAD   | depth, skeleton, inertial, video   |

Each folder contains Jupyter notebooks for feature extraction, feature selection, fusion, and robustness experiments (simulated sensor noise / signal loss at 10–50%), all evaluated under a strict LOSO protocol.

## Main results

Mean LOSO test performance with all modalities, feature selection applied to the full concatenated feature vector. Reported as mean ± std over 8 / 5 / 10 folds for UTD-MHAD / CZU-MHAD / MM-Fit. *Combined Meta* is the proposed per-fold metaheuristic selection.

### UTD-MHAD

| Method                            | Accuracy (%)   | F1 (%)          | Retention (%)  |
|-----------------------------------|----------------|-----------------|----------------|
| No-selection baseline             | 93.39 ± 4.09   | 92.30 ± 5.08    | 100            |
| Mutual Information                 | 94.54 ± 3.50   | 93.74 ± 4.17    | 50             |
| RFE                               | 95.59 ± 2.51   | 94.94 ± 3.00    | 50             |
| Lasso / L1                        | 83.29 ± 9.61   | 81.23 ± 10.19   | 50             |
| Best single metaheuristic (MFO)   | 95.59 ± 1.54   | 95.19 ± 1.72    | 49.96          |
| **Combined Meta (proposed)**      | **96.99 ± 1.96** | **96.63 ± 2.27** | 49.13 ± 0.60 |

### CZU-MHAD

| Method                            | Accuracy (%)   | F1 (%)          | Retention (%)  |
|-----------------------------------|----------------|-----------------|----------------|
| No-selection baseline             | 96.18 ± 2.95   | 96.10 ± 3.06    | 100            |
| Mutual Information                 | 94.75 ± 3.73   | 94.13 ± 4.50    | 50             |
| RFE                               | 94.83 ± 3.12   | 94.60 ± 3.12    | 50             |
| Lasso / L1                        | 96.42 ± 3.44   | 96.43 ± 3.36    | 50             |
| Best single metaheuristic (JAYA)  | 97.22 ± 2.72   | 97.22 ± 2.71    | 45.70          |
| **Combined Meta (proposed)**      | **97.91 ± 2.22** | **97.91 ± 2.21** | 39.39 ± 16.53 |

### MM-Fit

| Method                            | Accuracy (%)   | F1 (%)          | Retention (%)  |
|-----------------------------------|----------------|-----------------|----------------|
| No-selection baseline             | 97.62 ± 3.84   | 97.21 ± 4.74    | 100            |
| Mutual Information                 | 99.27 ± 1.41   | 99.28 ± 1.41    | 50             |
| RFE                               | 96.71 ± 4.21   | 96.54 ± 4.69    | 50             |
| Lasso / L1                        | 97.46 ± 3.55   | 97.04 ± 4.50    | 50             |
| Best single metaheuristic (SSA)   | 98.33 ± 2.49   | 98.28 ± 2.72    | 47.84          |
| **Combined Meta (proposed)**      | **98.93 ± 2.46** | **98.82 ± 2.74** | 43.45 ± 15.24 |

Additional findings:

- Feature retention drops to ~40–50% of the full set, cutting model parameters, on-disk size, and GPU memory by roughly half with no increase in inference latency.
- Improvement over the no-selection baseline is statistically significant on UTD-MHAD and CZU-MHAD (p < 0.05, Cohen's d = 1.11 and 1.18); MM-Fit is near-ceiling.
- The two-layer cascade pushes retention to ~22–26% while maintaining accuracy.
- Gains persist with a modality-token Transformer classifier, and cross-dataset transfer (UTD-MHAD → CZU-MHAD / MSR-Action3D, no retraining) improves by up to ~15 percentage points over the unselected baseline.

## Not included in this repository

Generated artifacts are excluded via `.gitignore` and must be produced locally by running the notebooks:

- `features*/` — extracted feature arrays (`.npy`)
- `results*/` — experiment outputs (`.json`, `.png`)
- model checkpoints (`.pt`, `.pth`, `.pkl`)
- the raw datasets themselves (download from their official sources)
- `.venv/` — local virtual environment

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install numpy pandas scikit-learn torch matplotlib jupyter
```

Then open the notebooks with `jupyter lab` or in VS Code.
