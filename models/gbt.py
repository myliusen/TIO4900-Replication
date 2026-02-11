import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
