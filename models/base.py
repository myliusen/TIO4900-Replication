import sklearn
import numpy as np

class LinearModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class RandomWalkModel:
    """Diagnostic only — predicts y[t] = y[t-1].
    
    With overlapping returns, this achieves high apparent R² due to
    mechanical autocorrelation (~11/12 months shared), NOT genuine
    predictive power. Use to check whether other models are simply
    mimicking this persistence.
    """
    def __init__(self):
        self.y_last = None

    def fit(self, X, y):
        # Store the last observed y value
        self.y_last = y[-1]
    
    def predict(self, X):
        # For Random Walk, always predict the last observed value
        n_pred = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.full(n_pred, self.y_last)
    
    
class HistoricalMeanModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean = np.mean(y)
    
    def predict(self, X):
        return np.array(self.mean)
    

class PCABaselineModel:
    def __init__(self, components=3, series='yields'):
        self.components = components
        self.series = series
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        # perform PCA on yields:
        yields = X[self.series]
        # Fit the PCA on the TRAINING set
        pca_scores = self.pca.fit_transform(yields)
        
        self.model.fit(pca_scores, y)
    
    def predict(self, X):
        yields = X[self.series]
        pca_scores = self.pca.transform(yields)
        return self.model.predict(pca_scores)
    

class PCABaselineModelPlusN:
    def __init__(self, components=3, series='yields', n_extra=1):
        self.components = components
        self.series = series
        self.n_extra = n_extra
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.fred_pca = sklearn.decomposition.PCA(n_components=self.n_extra)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        # PCA on yields/forwards
        yields = X[self.series]
        pca_scores = self.pca.fit_transform(yields)

        # PCA on 'fred'
        fred = X['fred']
        fred_pc1 = self.fred_pca.fit_transform(fred)  # shape (n_samples, 1)

        # Concatenate
        features = np.concatenate([pca_scores, fred_pc1], axis=1)

        self.model.fit(features, y)
    
    def predict(self, X):
        yields = X[self.series]
        pca_scores = self.pca.transform(yields)

        fred = X['fred']
        fred_pc1 = self.fred_pca.transform(fred)

        features = np.concatenate([pca_scores, fred_pc1], axis=1)
        return self.model.predict(features)
    

class PCABaselineModelMacroGroups:
    def __init__(self, components=3, series='yields', lasso=False, alpha=0.01):
        self.components = components
        self.series = series
        self.lasso = lasso
        self.alpha = alpha
        self.pca = sklearn.decomposition.PCA(n_components=components)
        if self.lasso:
            self.model = sklearn.linear_model.Lasso(alpha=self.alpha)
        else:
            self.model = sklearn.linear_model.LinearRegression()
        self.fred_pcas = {}  # Will hold one PCA per macro category

    def fit(self, X, y, X_val=None, y_val=None):
        # PCA on yields/forwards
        yields = X[self.series]
        pca_scores = self.pca.fit_transform(yields)

        # PCA on each macro category in 'fred'
        fred = X['fred']
        macro_cat_pc1s = []
        self.fred_pcas = {}
        for cat in fred.columns.get_level_values(0).unique():
            cat_df = fred[cat]
            pca = sklearn.decomposition.PCA(n_components=1)
            pc1 = pca.fit_transform(cat_df)
            macro_cat_pc1s.append(pc1)
            self.fred_pcas[cat] = pca

        macro_cat_pc1s = np.hstack(macro_cat_pc1s)  # shape (n_samples, n_cats)

        # Concatenate all features
        features = np.concatenate([pca_scores, macro_cat_pc1s], axis=1)
        self.model.fit(features, y)

    def predict(self, X):
        yields = X[self.series]
        pca_scores = self.pca.transform(yields)

        fred = X['fred']
        macro_cat_pc1s = []
        for cat, pca in self.fred_pcas.items():
            cat_df = fred[cat]
            pc1 = pca.transform(cat_df)
            macro_cat_pc1s.append(pc1)
        macro_cat_pc1s = np.hstack(macro_cat_pc1s)

        features = np.concatenate([pca_scores, macro_cat_pc1s], axis=1)
        return self.model.predict(features)