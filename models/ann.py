import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class NNModel:
    def __init__(self, hidden_size=32, epochs=200, lr=0.001):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.scaler = None
        self.model = None

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_s = X.astype(np.float32)
        y_t = torch.tensor(y.astype(np.float32)).reshape(-1, 1)
        X_t = torch.tensor(X_s)

        input_size = X_s.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        # X_s = self.scaler.transform(X).astype(np.float32)
        X_t = torch.tensor(X.astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).numpy().flatten()
