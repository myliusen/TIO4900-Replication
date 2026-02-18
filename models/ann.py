import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

DEVICE = 'cpu'

class BianchiANN:
    """
    One-hidden-layer feedforward network matching Bianchi's design choices,
    but using forward rates as a flat input (no group ensembling).
    """

    def __init__(self, hidden_size=3, series='forward', epochs=500, lr=0.015,
                 momentum=0.9, weight_decay=0.01, batch_size=32, patience=20,
                 tune_every=48, seed=42):
        self.hidden_size = hidden_size
        self.series = series
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.tune_every = tune_every
        self.seed = seed

        self.param_grid = list(ParameterGrid({
            'dropout': [0.1, 0.3, 0.5],
            'l1l2': [0.01, 0.001],
        }))

        self.scaler_ = None
        self.model_ = None
        self._fit_count = 0
        self._best_params = {'dropout': 0.3, 'l1l2': 0.001}

    def _build_net(self, n_in, n_out, dropout, l1l2):
        model = nn.Sequential(
            nn.Linear(n_in, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, n_out),
        ).to(DEVICE)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        return model

    def _train(self, model, X_train, y_train, X_val, y_val,
               l1l2, epochs=None):
        if epochs is None:
            epochs = self.epochs

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        criterion = nn.MSELoss()

        best_val = np.inf
        best_state = deepcopy(model.state_dict())
        wait = 0
        n = X_train.shape[0]

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:  # BatchNorm needs at least 2 samples
                    continue
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()
                loss = criterion(model(xb), yb)

                if l1l2 > 0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1l2 * l1

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        model.load_state_dict(best_state)
        return best_val

    def _select_features(self, X):
        if hasattr(X, 'columns') and isinstance(X.columns, pd.MultiIndex):
            return X[self.series].values.astype(np.float32)
        return np.array(X, dtype=np.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_np = self._select_features(X)
        y_np = np.array(y, dtype=np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        self.scaler_ = MinMaxScaler(feature_range=(-1, 1))
        X_np = self.scaler_.fit_transform(X_np)

        n = X_np.shape[0]
        n_train = int(n * 0.85)

        X_train = torch.tensor(X_np[:n_train], device=DEVICE)
        y_train = torch.tensor(y_np[:n_train], device=DEVICE)
        X_val = torch.tensor(X_np[n_train:], device=DEVICE)
        y_val = torch.tensor(y_np[n_train:], device=DEVICE)

        n_in = X_np.shape[1]
        n_out = y_np.shape[1]

        self._fit_count += 1
        if self._fit_count % self.tune_every == 1 or self._fit_count == 1:
            best_score = np.inf
            for params in self.param_grid:
                torch.manual_seed(self.seed)
                net = self._build_net(n_in, n_out, params['dropout'], params['l1l2'])
                val_loss = self._train(net, X_train, y_train, X_val, y_val,
                                       l1l2=params['l1l2'])
                if val_loss < best_score:
                    best_score = val_loss
                    self._best_params = params

        torch.manual_seed(self.seed)
        self.model_ = self._build_net(n_in, n_out,
                                       self._best_params['dropout'],
                                       self._best_params['l1l2'])
        self._train(self.model_, X_train, y_train, X_val, y_val,
                    l1l2=self._best_params['l1l2'])

    def predict(self, X):
        X_np = self._select_features(X)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
        X_np = self.scaler_.transform(X_np)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.tensor(X_np, device=DEVICE)).cpu().numpy()
        if pred.shape[1] == 1:
            return pred.flatten()
        return pred