import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import product


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