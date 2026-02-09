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