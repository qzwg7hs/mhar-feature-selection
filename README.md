# MHAR Feature Selection

Feature-selection experiments for multimodal human activity recognition (MHAR)
across three datasets:

| Folder      | Dataset    | Modalities used                     |
|-------------|------------|-------------------------------------|
| `czu-mhad/` | CZU-MHAD   | depth, inertial, skeleton           |
| `mmfit/`    | MM-Fit     | inertial / pose                     |
| `utd-mhad/` | UTD-MHAD   | inertial, skeleton, video           |

Each folder contains Jupyter notebooks for feature extraction, feature
selection, fusion, and robustness experiments (added sensor noise / signal
loss at 10–50%).

## Not included in this repository

Generated artifacts are excluded via `.gitignore` and must be produced locally
by running the notebooks:

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
