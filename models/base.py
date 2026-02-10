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
    def __init__(self):
        pass

    def fit(self, X, y):
        self.last_value = y[-1]
    
    def predict(self, X):
        # Return array of same length as X
        return np.full(X.shape[0], self.last_value)

    
class HistoricalMeanModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean = np.mean(y)
    
    def predict(self, X):
        return np.array(self.mean)
    

class PCABaselineModel:
    def __init__(self, components=3):
        self.components = components
        self.pca = sklearn.decomposition.PCA(n_components=components)
        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y):
        # perform PCA on yields:
        yields = X['yields']
        # Fit the PCA on the TRAINING set
        pca_scores = self.pca.fit_transform(yields)
        
        self.model.fit(pca_scores, y)
    
    def predict(self, X):
        yields = X['yields']
        pca_scores = self.pca.transform(yields)
        return self.model.predict(pca_scores)