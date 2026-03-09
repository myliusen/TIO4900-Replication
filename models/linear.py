import numpy as np
import sklearn.linear_model
import skglm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

class LassoModel:
    def __init__(self, alphas=None, series=None):
        self.alphas = alphas if alphas is not None else np.logspace(-5, 1, 30)
        self.series = series
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        # 1. Handle feature selection (subsetting columns)
        X_sub = X[self.series] if self.series else X
        
        n = len(y)
        split = int(n * 0.85)
        X_vals = X_sub.values
        X_subtrain, X_v = X_vals[:split], X_vals[split:]
        y_subtrain, y_v = y[:split], y[split:]

        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        # Optimization: only tune if we have enough validation data
        if len(y_v) >= 3:
            for alpha in self.alphas:
                m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
                m.fit(X_subtrain, y_subtrain)
                mse = np.mean((y_v - m.predict(X_v)) ** 2)
                if mse < best_mse:
                    best_mse, best_alpha = mse, alpha
        else:
            best_alpha = np.median(self.alphas)

        # 4. FINAL REFIT
        # Refit on subtrain + internal val
        X_final = np.vstack([X_subtrain, X_v])
        y_final = np.concatenate([y_subtrain, y_v])
        
        self.model = sklearn.linear_model.Lasso(alpha=best_alpha, max_iter=10000)
        self.model.fit(X_final, y_final)

    def predict(self, X):
        X_sub = X[self.series].values if self.series else X.values
        return self.model.predict(X_sub)

class RidgeModel:
    """
    Ridge with internal time-series-safe hyperparameter tuning.
    
    Same 85/15 temporal split approach as LassoModel.
    """
    
    def __init__(self, alphas=None, series='yields'):
        if alphas is None:
            self.alphas = np.logspace(-5, 5, 30)  # 1e-5 to 1e5
        else:
            self.alphas = alphas
        self.series = series
        self.best_alpha_ = None
        self.model = None
    
    def fit(self, X, y):
        X_sub = X[[self.series]].values if self.series else X.values
        n = len(y)
        
        # 85/15 temporal split — no shuffling
        split = int(n * 0.85)
        
        # Need enough data in both splits
        if split < 10 or (n - split) < 3:
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
            self.model.fit(X_sub, y)
            return
        
        X_subtrain, X_val = X_sub[:split], X_sub[split:]
        y_subtrain, y_val = y[:split], y[split:]
        
        # Grid search
        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        for alpha in self.alphas:
            m = sklearn.linear_model.Ridge(alpha=alpha)
            m.fit(X_subtrain, y_subtrain)
            preds = m.predict(X_val)
            mse = np.mean((y_val - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        
        self.best_alpha_ = best_alpha

        # Refit on full training set with best alpha
        self.model = sklearn.linear_model.Ridge(alpha=self.best_alpha_)
        self.model.fit(X_sub, y)
    
    def predict(self, X):
        X_sub = X[[self.series]].values if self.series else X.values
        return self.model.predict(X_sub)


class GroupLassoModel:
    """
    Group Lasso with internal 85/15 temporal validation split for alpha tuning.
    
    Uses StandardScaler on features. The `groups` parameter is a per-feature
    integer array (e.g., [0,0,0,1,1,2,2,2,...]) that maps each column to its group.
    Internally converted to group_sizes list for skglm.GroupLasso.
    """
    
    def __init__(self, alphas=None, groups=None):
        if alphas is None:
            self.alphas = np.logspace(-4, 1, 30)
        else:
            self.alphas = alphas
        self.groups = groups  # per-feature integer label array
        self.model = None
        self.scaler = None
        self.best_alpha_ = None

    def _get_group_sizes(self):
        """Convert per-feature group labels to ordered group sizes list for skglm."""
        _, counts = np.unique(self.groups, return_counts=True)
        return counts.tolist()

    def fit(self, X, y):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        y_vals = np.array(y).ravel()
        
        group_sizes = self._get_group_sizes()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_vals)
        
        n = len(y_vals)
        split = int(n * 0.85)
        
        # Fallback if not enough data for a proper split
        if split < 10 or (n - split) < 3:
            self.best_alpha_ = np.median(self.alphas)
            self.model = skglm.GroupLasso(alpha=self.best_alpha_, groups=group_sizes)
            self.model.fit(X_scaled, y_vals)
            return
        
        X_subtrain, X_val = X_scaled[:split], X_scaled[split:]
        y_subtrain, y_val = y_vals[:split], y_vals[split:]
        
        # Grid search over alpha
        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        for alpha in self.alphas:
            try:
                m = skglm.GroupLasso(alpha=alpha, groups=group_sizes)
                m.fit(X_subtrain, y_subtrain)
                preds = m.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = alpha
            except Exception:
                # Some alpha values may cause convergence issues; skip them
                continue
        
        self.best_alpha_ = best_alpha
        
        # Refit on full training set (subtrain + val) with best alpha
        self.model = skglm.GroupLasso(alpha=self.best_alpha_, groups=group_sizes)
        self.model.fit(X_scaled, y_vals)
    
    def predict(self, X):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        X_scaled = self.scaler.transform(X_vals)
        return self.model.predict(X_scaled)


class BianchiElasticNet:
    """
    Faithful reimplementation of Bianchi's ElasticNet_Exog_Plain,
    adapted to work with our expanding_window API.

    Key design choices matching Bianchi exactly:
      - StandardScaler on training data, applied to test
      - PredefinedSplit: last 15% of training as single validation fold
      - ElasticNetCV with l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]
      - max_iter=5000, random_state=42, n_jobs=-1
      - No refit on full training set after CV (ElasticNetCV uses its
        internal refit on the CV-selected alpha/l1_ratio)
    """

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            All features (macro + yields concatenated) for the training window.
        y : np.array, shape (n_samples,)
            Target (single maturity excess return).
        """
        X_train = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        y_train = np.array(y).ravel()

        # Scale inputs for training
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Construct validation sample as last 15% of training sample
        N_train = int(np.round(X_train_scaled.shape[0] * 0.85))
        N_val = X_train_scaled.shape[0] - N_train
        test_fold = np.concatenate((
            np.full(N_train, -1),   # -1 = always in training
            np.full(N_val, 0)       #  0 = validation fold
        ))
        ps = PredefinedSplit(test_fold.tolist())

        # Fit ElasticNetCV — exactly as Bianchi
        self.model = sklearn.linear_model.ElasticNetCV(
            cv=ps,
            max_iter=20000,
            n_jobs=-1,
            l1_ratio=[.1, .3, .5, .7, .9],
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X):
        """
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (1, n_features) or (n, n_features)
        
        Returns
        -------
        np.array of predictions
        """
        X_test = np.array(X) if not isinstance(X, np.ndarray) else X.copy()
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)