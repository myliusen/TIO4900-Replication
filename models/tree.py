import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import product
import sklearn.ensemble

class ForwardTreeEnsembleModel:
    """
    Tree ensemble model using ONLY the forward block of X.
    Supports:
      - estimator: 'ef' (ExtraTrees) or 'rf' (RandomForest)
      - optional scaling
      - optional PCA on forward rates
      - 85/15 temporal split for internal validation
      - grid search over model hyperparameters
      - tune_every: only retune hyperparameters every N fit() calls
    """

    def __init__(
        self,
        estimator="ef",
        scale=True,
        use_pca=False,
        n_components=3,
        param_grid=None,
        tune_every=60,
        random_state=42,
    ):
        self.estimator = estimator.lower()
        if self.estimator not in {"ef", "rf"}:
            raise ValueError("estimator must be 'ef' or 'rf'")

        self.scale = scale
        self.use_pca = use_pca
        self.n_components = n_components
        self.tune_every = tune_every
        self.random_state = random_state

        self.scaler = None
        self.pca = None
        self.model = None
        self.best_params_ = None
        self._fit_calls = 0

        if param_grid is None:
            if self.estimator == "ef":
                self.param_grid = {
                    "n_estimators": [200, 500, 1000],
                    "max_depth": [2, 3, 5],
                    "min_samples_leaf": [1, 5],
                }
            else:  # rf
                self.param_grid = {
                    "n_estimators": [200, 500, 1000],
                    "max_depth": [2, 3, 5],
                    "min_samples_leaf": [1, 5],
                }
        else:
            self.param_grid = param_grid

    def _get_forward_block(self, X):
        if not hasattr(X, "columns"):
            return np.array(X)

        if hasattr(X.columns, "levels"):  # MultiIndex columns
            if "forward" not in X.columns.get_level_values(0):
                raise ValueError("Expected 'forward' block in level-0 columns.")
            return X["forward"].values

        # fallback for non-MultiIndex DataFrame: try column name
        if "forward" in X.columns:
            vals = X["forward"]
            return vals.values.reshape(-1, 1) if vals.ndim == 1 else vals.values

        raise ValueError("Could not locate forward features in X.")

    def _transform_forward(self, X_forward, fit=False):
        Z = X_forward

        if self.scale:
            if fit or self.scaler is None:
                self.scaler = StandardScaler()
                Z = self.scaler.fit_transform(Z)
            else:
                Z = self.scaler.transform(Z)

        if self.use_pca:
            n_comp = min(self.n_components, Z.shape[1])
            if fit or self.pca is None:
                self.pca = PCA(n_components=n_comp)
                Z = self.pca.fit_transform(Z)
            else:
                Z = self.pca.transform(Z)

        return Z

    def _make_estimator(self, params):
        if self.estimator == "ef":
            return ExtraTreesRegressor(random_state=self.random_state, **params)
        return RandomForestRegressor(random_state=self.random_state, **params)

    def _should_tune(self):
        if self.best_params_ is None:
            return True
        if self.tune_every is None or self.tune_every <= 1:
            return True
        return (self._fit_calls % self.tune_every) == 0

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        X_forward = self._get_forward_block(X)
        X_feat = self._transform_forward(X_forward, fit=True)

        n = len(y)
        split = int(n * 0.85)
        default_params = {k: v[0] for k, v in self.param_grid.items()}

        # Not enough data for split: fit with best/default params
        if split < 10 or (n - split) < 3:
            params = self.best_params_ if self.best_params_ is not None else default_params
            self.model = self._make_estimator(params)
            self.model.fit(X_feat, y)
            self.best_params_ = params
            self._fit_calls += 1
            return

        do_tune = self._should_tune()

        if do_tune:
            print("Tuning hyperparameters...")
            X_subtrain, X_val = X_feat[:split], X_feat[split:]
            y_subtrain, y_val = y[:split], y[split:]

            param_names = list(self.param_grid.keys())
            param_values = list(self.param_grid.values())

            best_mse = np.inf
            best_params = default_params

            for combo in product(*param_values):
                params = dict(zip(param_names, combo))
                m = self._make_estimator(params)
                m.fit(X_subtrain, y_subtrain)
                pred = m.predict(X_val)
                mse = np.mean((y_val - pred) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_params = params

            self.best_params_ = best_params

        # Refit on full train with current best params
        self.model = self._make_estimator(self.best_params_)
        self.model.fit(X_feat, y)
        self._fit_calls += 1

    def predict(self, X):
        X_forward = self._get_forward_block(X)
        X_feat = self._transform_forward(X_forward, fit=False)
        return self.model.predict(X_feat)


class ExtraTreesForwardModel(ForwardTreeEnsembleModel):
    def __init__(self, **kwargs):
        super().__init__(estimator="ef", **kwargs)


class RandomForestForwardModel(ForwardTreeEnsembleModel):
    def __init__(self, **kwargs):
        super().__init__(estimator="rf", **kwargs)


class ExtraTreesModel:
    """
    Extra Trees Regressor with internal time-series-safe hyperparameter tuning.
    
    At each expanding window step, fit() receives X_train, y_train.
    Within fit():
      - Optionally apply PCA / feature selection via `features` config
      - Use the first 85% as sub-train, last 15% as validation
      - Grid search over hyperparameters on sub-train → evaluate on validation
      - Refit on full X_train with best hyperparameters
    """
    
    def __init__(self, features=None, param_grid=None, random_state=42):
        """
        Parameters
        ----------
        features : dict or None
            Feature configuration, same format as XGBoostModel.
            E.g. {'forward': {'method': 'raw'}, 'fred': {'method': 'pca', 'n_components': 5}}
            If None, uses all columns raw.
        param_grid : dict or None
            Grid of hyperparameters to search over.
            If None, uses a sensible default grid.
        random_state : int
            Random seed for reproducibility.
        """
        self.features = features
        self.random_state = random_state
        self.model = None
        self.best_params_ = None
        
        # Feature transformation state
        self._scalers = {}
        self._pcas = {}
        
        if param_grid is None:
            self.param_grid = {
            'n_estimators': [200],
            'max_depth': [3, 5, 10],
            'min_samples_leaf': [1, 5],
            }
        else:
            self.param_grid = param_grid
    
    def _build_features(self, X, fit=True):
        """Build feature matrix from config, fitting scalers/PCA only when fit=True."""
        if self.features is None:
            arr = X.values if hasattr(X, 'values') else np.array(X)
            if fit:
                self._global_scaler = StandardScaler()
                return self._global_scaler.fit_transform(arr)
            else:
                return self._global_scaler.transform(arr)
        
        parts = []
        for group, cfg in self.features.items():
            # Extract the group columns from the MultiIndex DataFrame
            if group in X.columns.get_level_values(0):
                X_group = X[group].values
            else:
                continue
            
            method = cfg.get('method', 'raw')
            
            if fit:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_group)
                self._scalers[group] = scaler
            else:
                X_scaled = self._scalers[group].transform(X_group)
            
            if method == 'pca':
                n_comp = cfg.get('n_components', 3)
                if fit:
                    pca = PCA(n_components=n_comp)
                    X_out = pca.fit_transform(X_scaled)
                    self._pcas[group] = pca
                else:
                    X_out = self._pcas[group].transform(X_scaled)
                parts.append(X_out)
            else:
                parts.append(X_scaled)
        
        return np.hstack(parts)
    
    def fit(self, X, y):
        X_transformed = self._build_features(X, fit=True)
        n = len(y)
        
        split = int(n * 0.85)
        
        # If too little data, use defaults on full set
        if split < 10 or (n - split) < 3:
            self.best_params_ = {k: v[0] for k, v in self.param_grid.items()}
            self.model = ExtraTreesRegressor(
                random_state=self.random_state, **self.best_params_)
            self.model.fit(X_transformed, y)
            return
        
        X_subtrain, X_val = X_transformed[:split], X_transformed[split:]
        y_subtrain, y_val = y[:split], y[split:]
        
        # Grid search
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        best_mse = np.inf
        best_params = {k: v[0] for k, v in self.param_grid.items()}
        
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            m = ExtraTreesRegressor(random_state=self.random_state, **params)
            m.fit(X_subtrain, y_subtrain)
            preds = m.predict(X_val)
            mse = np.mean((y_val - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_params = params
        
        self.best_params_ = best_params
        
        # Refit on full training set with best params
        self.model = ExtraTreesRegressor(
            random_state=self.random_state, **self.best_params_)
        self.model.fit(X_transformed, y)
    
    def predict(self, X):
        X_transformed = self._build_features(X, fit=False)
        return self.model.predict(X_transformed)
    

class GroupPCARandomForest:
    def __init__(self, components=3, series='yields', macro_pcs=1, rf_kwargs=None):
        self.components = components
        self.series = series
        self.macro_pcs = macro_pcs
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.fred_pcas = {}  # category: PCA object
        self.rf_kwargs = rf_kwargs if rf_kwargs is not None else {}
        self.model = sklearn.ensemble.RandomForestRegressor(**self.rf_kwargs)

    def fit(self, X, y):
        # PCA on yields/forwards
        yields = X[self.series]
        pca_scores = self.pca.fit_transform(yields)

        # PCA on each macro category in 'fred'
        fred = X['fred']
        macro_cat_pcs = []
        self.fred_pcas = {}
        for cat in fred.columns.get_level_values(0).unique():
            cat_df = fred[cat]
            pca = sklearn.decomposition.PCA(n_components=self.macro_pcs)
            pcs = pca.fit_transform(cat_df)
            macro_cat_pcs.append(pcs)
            self.fred_pcas[cat] = pca

        macro_cat_pcs = np.hstack(macro_cat_pcs)  # shape (n_samples, n_cats * macro_pcs)

        # Concatenate all features
        features = np.concatenate([pca_scores, macro_cat_pcs], axis=1)
        self.model.fit(features, y)

    def predict(self, X):
        yields = X[self.series]
        pca_scores = self.pca.transform(yields)

        fred = X['fred']
        macro_cat_pcs = []
        for cat, pca in self.fred_pcas.items():
            cat_df = fred[cat]
            pcs = pca.transform(cat_df)
            macro_cat_pcs.append(pcs)
        macro_cat_pcs = np.hstack(macro_cat_pcs)

        features = np.concatenate([pca_scores, macro_cat_pcs], axis=1)
        return self.model.predict(features)