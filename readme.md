# TIØ4900 Master's thesis

Minimal replication code for the master's thesis.

## Purpose
Provide utilities and a notebook to reproduce the expanding-window forecasting (y_{t+1} = beta x_t) experiments with KR yields and FRED‑MD features.

## Quick start
1. Put required data files in `data/` (e.g. `2026-01-MD.csv`, yields CSV).
2. Open and run `init.ipynb` (VS Code or Jupyter).
3. Inspect/modify models in the notebook (`LassoModel`, `LinearModel`, `NNModel`).

## Files
- `init.ipynb` — main notebook to run experiments
- `utils/base_utils.py` — data loaders, forward/excess return functions
- `utils/window_utils.py` — expanding-window OOS framework and metrics
- `models/` — model implementations plus temporary storage (wip)
- `data/` — (place datasets here)

## Notes
- Features X_t predict target y_{t+1} (no look‑ahead).
- Standardize / apply PCA inside model.fit() to avoid leakage.