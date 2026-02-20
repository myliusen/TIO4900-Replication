import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Suggested architecture grids (for CV in notebooks) ──────────────────────
#
# These are *plausible* tree specifications in the spirit of Bianchi–Büchner–
# Tamoni (shallow → medium → deeper trees, more estimators + stronger
# regularization).  They are not calibrated to exactly match the paper’s
# Internet Appendix, but provide a reasonable search space for expanding-
# window CV in the notebooks.

XGB_ARCH_GRID = [
    # Very shallow / stable
    {
        "name": "xgb_shallow",
        "max_depth": 2,
        "n_estimators": 200,
        "learning_rate": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    # Baseline
    {
        "name": "xgb_medium",
        "max_depth": 3,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    # Slightly deeper / more regularized
    {
        "name": "xgb_deep",
        "max_depth": 4,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    },
]


LGBM_ARCH_GRID = [
    # Shallow trees, few leaves
    {
        "name": "lgbm_shallow",
        "num_leaves": 15,
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_data_in_leaf": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    # Baseline
    {
        "name": "lgbm_medium",
        "num_leaves": 31,
        "max_depth": -1,  # let num_leaves control complexity
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_data_in_leaf": 30,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    # More trees / stronger regularization
    {
        "name": "lgbm_deep",
        "num_leaves": 63,
        "max_depth": -1,
        "n_estimators": 600,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_data_in_leaf": 50,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
]


class XGBoostModel:
    """XGBoost regressor with optional per-group feature engineering.

    Parameters
    ----------
    features : dict or None
        Per-group feature config.  Keys are column group names in the
        multi-level DataFrame ('fred', 'forward', 'yields').
        Each value is a dict with:
            'method': 'raw' | 'pca'
            'n_components': int  (required when method='pca')
        Groups not listed are dropped.
        If None (default), all features are used as-is (raw).

        Example — mimic LN-BH inputs::

            features={
                'fred':    {'method': 'pca', 'n_components': 8},
                'yields':  {'method': 'pca', 'n_components': 3},
                'forward': {'method': 'raw'},
            }

    All remaining keyword arguments are forwarded to XGBRegressor.
    """

    def __init__(self, features=None, n_estimators=100, max_depth=3,
                 learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.0, reg_lambda=1.0, random_state=42):
        self.features = features
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'reg:squarederror',
        }
        self.model = None
        # Fitted transformers per group (populated in fit)
        self._scalers = {}
        self._pcas = {}

    # ── internal helpers ─────────────────────────────────────────────

    def _build_features(self, X, *, fit: bool) -> np.ndarray:
        """Extract / transform features from X.

        When fit=True the scalers and PCA objects are fit on X (training).
        When fit=False the previously-fit transformers are applied (predict).
        """
        if self.features is None:
            # No feature config → use everything raw
            if isinstance(X, pd.DataFrame):
                return X.values.astype(np.float64)
            return np.asarray(X, dtype=np.float64)

        blocks = []
        for group, cfg in self.features.items():
            X_group = X[group].values.astype(np.float64)
            method = cfg.get('method', 'raw')

            if method == 'pca':
                n_comp = cfg['n_components']
                if fit:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_group)
                    pca = PCA(n_components=n_comp)
                    X_out = pca.fit_transform(X_scaled)
                    self._scalers[group] = scaler
                    self._pcas[group] = pca
                else:
                    X_scaled = self._scalers[group].transform(X_group)
                    X_out = self._pcas[group].transform(X_scaled)
                blocks.append(X_out)

            elif method == 'raw':
                blocks.append(X_group)

            else:
                raise ValueError(f"Unknown method '{method}' for group '{group}'")

        return np.hstack(blocks)

    # ── public interface ─────────────────────────────────────────────

    def fit(self, X, y):
        X_np = self._build_features(X, fit=True)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X_np, y)

    def predict(self, X):
        X_np = self._build_features(X, fit=False)
        return self.model.predict(X_np)


class LightGBMModel:
    """LightGBM regressor with optional per-group feature engineering.

    Interface mirrors ``XGBoostModel`` so it can be dropped into the same
    expanding-window code.

    Parameters
    ----------
    features : dict or None
        Same convention as ``XGBoostModel`` (see above).

    All remaining keyword arguments are forwarded to ``lgb.LGBMRegressor``.
    """

    def __init__(self, features=None,
                 num_leaves=31, max_depth=-1,
                 learning_rate=0.05, n_estimators=400,
                 subsample=0.8, colsample_bytree=0.8,
                 min_data_in_leaf=30,
                 reg_alpha=0.0, reg_lambda=0.0,
                 random_state=42):
        self.features = features
        self.params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_data_in_leaf": min_data_in_leaf,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "regression",
        }
        self.model = None
        # Fitted transformers per group (populated in fit)
        self._scalers = {}
        self._pcas = {}

    # ── internal helpers ─────────────────────────────────────────────

    def _build_features(self, X, *, fit: bool) -> np.ndarray:
        """Extract / transform features from X.

        When fit=True the scalers and PCA objects are fit on X (training).
        When fit=False the previously-fit transformers are applied (predict).
        """
        if self.features is None:
            # No feature config → use everything raw
            if isinstance(X, pd.DataFrame):
                return X.values.astype(np.float64)
            return np.asarray(X, dtype=np.float64)

        blocks = []
        for group, cfg in self.features.items():
            X_group = X[group].values.astype(np.float64)
            method = cfg.get("method", "raw")

            if method == "pca":
                n_comp = cfg["n_components"]
                if fit:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_group)
                    pca = PCA(n_components=n_comp)
                    X_out = pca.fit_transform(X_scaled)
                    self._scalers[group] = scaler
                    self._pcas[group] = pca
                else:
                    X_scaled = self._scalers[group].transform(X_group)
                    X_out = self._pcas[group].transform(X_scaled)
                blocks.append(X_out)

            elif method == "raw":
                blocks.append(X_group)

            else:
                raise ValueError(f"Unknown method '{method}' for group '{group}'")

        return np.hstack(blocks)

    # ── public interface ─────────────────────────────────────────────

    def fit(self, X, y):
        X_np = self._build_features(X, fit=True)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_np, y)

    def predict(self, X):
        X_np = self._build_features(X, fit=False)
        return self.model.predict(X_np)
