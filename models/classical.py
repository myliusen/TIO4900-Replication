import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CochranePiazzesiModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        # Selects the 'forwards' sub-frame
        X_forwards = X['forward']
        # Only use the first 5 forward rates (12m, 24m, 36m, 48m, 60m) as in CP (2005)
        X_forwards = X_forwards.iloc[:, :5]

        # 2. Run OLS to estimate the factor
        # CP (2005) show that unrestricted OLS on forward rates recovers the "tent-shaped" factor
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(X_forwards, y)
    
    def predict(self, X):
        X_forwards = X['forward']
        X_forwards = X_forwards.iloc[:, :5]
        
        return self.model.predict(X_forwards)
    
    
class LudvigsonNgModel:
    def __init__(self, n_factors=8):
        # We need enough components to get at least up to F8
        self.n_factors = max(n_factors, 8)
        self.pca = sklearn.decomposition.PCA(n_components=self.n_factors)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y):
        # 1. Prepare Macro Factors (F1...F8)
        X_macro = X['macro'] if isinstance(X, pd.DataFrame) and 'macro' in X.columns else X
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_macro)
        factors = self.pca.fit_transform(X_scaled) # Shape (T, n_factors)
        
        # 2. Construct CP Factor (Tent-shaped linear combination of forwards)
        # We must estimate CP internally to avoid data leakage.
        X_forwards = X['forward'].iloc[:, :5].values 
        
        # Estimate CP factor: regress y on 5 forwards
        # (Strictly: Regress average y on forwards, but regressing current y is common proxy)
        cp_ols = sklearn.linear_model.LinearRegression()
        cp_ols.fit(X_forwards, y)
        cp_factor = cp_ols.predict(X_forwards).reshape(-1, 1)

        # 3. Select Specific LN Factors
        # Indices are 0-based: F1->0, F3->2, F4->3, F8->7
        F1 = factors[:, 0].reshape(-1, 1)
        F3 = factors[:, 2].reshape(-1, 1)
        F4 = factors[:, 3].reshape(-1, 1)
        F8 = factors[:, 7].reshape(-1, 1)
        
        # Construct Non-linear term F1^3
        F1_cubic = F1 ** 3

        # 4. Form final design matrix [CP, F1, F1^3, F3, F4, F8]
        X_final = np.hstack([cp_factor, F1, F1_cubic, F3, F4, F8])

        # Store CP model for prediction step
        self.cp_model = cp_ols
        
        self.model.fit(X_final, y)
    
    def predict(self, X):
        # 1. Macro Factors
        X_macro = X['macro'] if isinstance(X, pd.DataFrame) and 'macro' in X.columns else X
        X_scaled = self.scaler.transform(X_macro)
        factors = self.pca.transform(X_scaled)
        
        # 2. CP Factor
        X_forwards = X['forward'].iloc[:, :5].values 
        cp_factor = self.cp_model.predict(X_forwards).reshape(-1, 1)

        # 3. Select Specific Factors
        F1 = factors[:, 0].reshape(-1, 1)
        F3 = factors[:, 2].reshape(-1, 1)
        F4 = factors[:, 3].reshape(-1, 1)
        F8 = factors[:, 7].reshape(-1, 1)
        F1_cubic = F1 ** 3
        
        X_final = np.hstack([cp_factor, F1, F1_cubic, F3, F4, F8])
        
        return self.model.predict(X_final)
    
class LudvigsonNgBauerHamiltonSpec:
    def __init__(self, n_factors=8):
        # We need enough components to get at least up to F8
        self.n_factors = max(n_factors, 8)
        self.pca_macro = sklearn.decomposition.PCA(n_components=self.n_factors)
        self.pca_yields = sklearn.decomposition.PCA(n_components=3)
        self.model = sklearn.linear_model.LinearRegression()
        self.macro_scaler = StandardScaler()

    def fit(self, X, y):
        # 1) Yield PCs (PC1-3) from yield panel
        X_yields = X['yields']
        yield_pcs = self.pca_yields.fit_transform(X_yields)

        # 2) Macro PCs (F1-F8) from macro panel
        X_macro = X['macro'] if 'macro' in X.columns else X['fred']
        X_macro_scaled = self.macro_scaler.fit_transform(X_macro)
        macro_factors = self.pca_macro.fit_transform(X_macro_scaled)

        # 3) Combine [PC1-3, F1-8]
        X_final = np.hstack([yield_pcs[:, :3], macro_factors[:, :8]])

        self.model.fit(X_final, y)

    def predict(self, X):
        X_yields = X['yields']
        yield_pcs = self.pca_yields.transform(X_yields)

        X_macro = X['macro'] if 'macro' in X.columns else X['fred']
        X_macro_scaled = self.macro_scaler.transform(X_macro)
        macro_factors = self.pca_macro.transform(X_macro_scaled)

        X_final = np.hstack([yield_pcs[:, :3], macro_factors[:, :8]])

        return self.model.predict(X_final)
