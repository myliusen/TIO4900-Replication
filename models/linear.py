import numpy as np
import sklearn.linear_model
import skglm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

class LassoModel:
    """
    Lasso with internal time-series-safe hyperparameter tuning.
    
    At each expanding window step, fit() receives X_train, y_train
    (all data up to time t). Within fit():
      - Use the first 85% as sub-train, last 15% as validation
      - Grid search over alpha on sub-train → evaluate on validation
      - Refit on full X_train with best alpha
    
    No future data is ever used: the validation set is always the
    most recent portion of the training window (temporal split).
    """
    
    def __init__(self, alphas=None, series='forward'):
        if alphas is None:
            self.alphas = np.logspace(-5, 1, 30)  # 1e-5 to 10
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
            # Too little data to tune — just use median alpha on full set
            self.best_alpha_ = np.median(self.alphas)
            self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
            self.model.fit(X_sub, y)
            return
        
        X_subtrain, X_val = X_sub[:split], X_sub[split:]
        y_subtrain, y_val = y[:split], y[split:]
        
        # Grid search
        best_alpha = self.alphas[0]
        best_mse = np.inf
        
        for alpha in self.alphas:
            m = sklearn.linear_model.Lasso(alpha=alpha, max_iter=10000)
            m.fit(X_subtrain, y_subtrain)
            preds = m.predict(X_val)
            mse = np.mean((y_val - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
        
        self.best_alpha_ = best_alpha

        # Refit on full training set with best alpha
        self.model = sklearn.linear_model.Lasso(alpha=self.best_alpha_, max_iter=10000)
        self.model.fit(X_sub, y)
    
    def predict(self, X):
        X_sub = X[[self.series]].values if self.series else X.values
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
    def __init__(self, alpha=0.01, groups=None):
        self.alpha = alpha
        self.groups = groups

    def fit(self, X, y):
        # Extract integer group codes from the MultiIndex
        groups = self.groups
        
        # skglm expects a list of group sizes (contiguous blocks),
        # but groups_as_array gives per-column integer labels.
        # skglm.GroupLasso accepts either format depending on version.
        # With skglm, pass the number of features per group as a list:
        import numpy as np
        _, counts = np.unique(groups, return_counts=True)
        group_sizes = counts.tolist()

        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        self.model = skglm.GroupLasso(alpha=self.alpha, groups=group_sizes)
        self.model.fit(X_vals, y)
    
    def predict(self, X):
        X_vals = X.values if hasattr(X, 'values') else np.array(X)
        return self.model.predict(X_vals)

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