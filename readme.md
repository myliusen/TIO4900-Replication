# TIØ4900 Master's Thesis: Bond Return Forecasting Replication

Minimal replication codebase for the master's thesis, focusing on forecasting bond excess returns using expanding-window OOS frameworks.

## Purpose
This repository provides utilities, models, and Jupyter notebooks to reproduce out-of-sample expanding-window forecasting experiments ($y_{t+1} = \beta x_t$) for bond excess returns using yields (e.g., KR yields) and macroeconomic factors (e.g., FRED-MD). The project explicitly implements deep learning ensembles and standard baseline models found in macro-finance literature (e.g., Ludvigson and Ng).

## Quick Start
1. Ensure your required data files are placed within the `data/` directory (e.g., `2026-01-MD.csv`, `kr_yields.csv`, `gsw_yields.csv`).
2. Install the necessary dependencies (consider creating a virtual environment and using `pip install -r requirements.txt`).
3. Open and run `notebooks/init.ipynb` via your preferred Jupyter environment (e.g., VS Code, JupyterLab) as an entry point for data generation, expanding windows, and results visualization.

## Directory Structure
- **`bianchi_replication/`** — Original code templates and NN ensembling prototypes replicated from macroeconomic bond forecasting research.
- **`data/`** — Subdirectory for raw and generated datasets. Ensure ALFRED / FRED-MD snapshots and yield curves (KR, GSW, LW) are stored here.
- **`models/`** — Source code for the actual model wrappers.
  - Classical (Random Walk, Historical Mean, Cochrane-Piazzesi)
  - Linear (PCA, Ridge, PCR)
  - Tree-based (Random Forest, Extra Trees)
  - Deep Learning (`pytorch_mlp.py`, `ann.py`, etc., including PyTorch-based ensemble models with automated early stopping)
- **`notebooks/`** — Jupyter Notebooks for exploratory data analysis (`eda.ipynb`), individual models (`gbt.ipynb`, `lasso_pcr.ipynb`), and the main testing environment (`init.ipynb`).
- **`utils/`** — Core helpers:
  - `base_utils.py` — Data loaders, returns calculus, forward rate generation, plotting functions.
  - `window_utils.py` — Expanding-window OOS execution framework and $R^2_{OOS}$ eval metrics.
  - `macro_grouping.py` / `shap_utils.py` — Data structuring and explainability components.

## Notes & Best Practices
- **Data Leakage:** All standardization, scaling, dimensionality reduction (e.g., PCA) or hyperparameter grids are optimized strictly within the `fit()` steps on the respective in-sample partition to ensure no forward-looking bias. 
- **Ensemble Stability:** The built-in MLP/Neural Network wrappers support `n_mc` (Monte Carlo runs) and `n_avg` (top-N validation selection), making it straightforward to match complex ensembling rules while maintaining reproducibility via seeds.