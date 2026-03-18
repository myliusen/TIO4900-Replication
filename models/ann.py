import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class EarlyStopping:
    """
    Simple and reusable early stopping module to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class _MLPNetwork(nn.Module):
    """
    The underlying PyTorch neural network module.
    Constructs a simple feedforward architecture based on the `archi` tuple.
    """
    def __init__(self, input_dim, archi, output_dim):
        super(_MLPNetwork, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(archi):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU()) # Standard ReLU activation
            
            # Apply Batch Normalization to the activations after the last ReLU layer
            if i == len(archi) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            current_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Pack layers into a sequential block
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PyTorchMLPWrapper:
    """
    A scikit-learn style wrapper for the PyTorch MLP.
    Designed to fit exactly into the API used in classical.py and window_utils.py.
    Forward-rate only network.
    """
    def __init__(self, archi=(3,), lr=0.01, epochs=100, 
                 seed=42, momentum=0.9, param_grid=None, tune_every=60, patience=10,
                 n_mc=1, n_avg=1, seeds=None):
        self.archi = archi
        self.lr = lr
        self.epochs = epochs
        self.random_state = seed # Base seed for single model mode
        self.momentum = momentum
        self.param_grid = param_grid if param_grid is not None else {'penalty': [0.001, 0.0001]}
        self.tune_every = tune_every
        self.patience = patience
        
        # Ensemble parameters
        self.n_mc = n_mc
        self.n_avg = n_avg
        # Default seeds if not provided
        self.seeds = seeds if seeds is not None else [seed + i for i in range(n_mc)]
        
        # Internal state
        self.models = [] # List storing the trained models
        self.criterion = nn.MSELoss() # Standard MSE for regression
        self.best_params_ = None
        self._fit_calls = 0
        
        # Scalers
        self.x_scaler = None
        self.y_scaler = None

    def _set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
    def _extract_array(self, data, is_X=True):
        """Helper to extract pure numpy arrays from potentially complex input structures."""
        if is_X and isinstance(data, pd.DataFrame) and 'forward' in data:
            data = data['forward']
            
        if hasattr(data, 'values'):
            arr = data.values
        else:
            arr = np.array(data)
            
        if not is_X and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            
        return arr

    def _should_tune(self):
        if self.best_params_ is None:
            return True
        if self.tune_every is None or self.tune_every <= 1:
            return True
        return (self._fit_calls % self.tune_every) == 0

    def fit(self, X, y):
        """
        Fits the neural network. 
        """
        X_arr = self._extract_array(X, is_X=True)
        y_arr = self._extract_array(y, is_X=False)
        
        # Always refit scalers on the current expanding window's training set
        if self.x_scaler is None:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            
        X_scaled = self.x_scaler.fit_transform(X_arr)
        y_scaled = self.y_scaler.fit_transform(y_arr)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        input_dim = X_tensor.shape[1]
        output_dim = y_tensor.shape[1]
        n_samples = X_tensor.shape[0]

        split = int(n_samples * 0.85)
        
        # Determine whether to tune and get best params this cycle
        tuning_needed = self._should_tune() and split >= 10 and (n_samples - split) >= 3
        
        self.models = [] # Reset models for new fit
        run_results = [] # To store (val_loss, model_state, best_penalty, current_epochs) for each seed

        # Loop over requested number of MC runs
        for run_idx in range(self.n_mc):
            curr_seed = self.seeds[run_idx] if len(self.seeds) > run_idx else self.random_state + run_idx
            self._set_seed(curr_seed)
            
            best_penalty = self.param_grid['penalty'][0]
            current_epochs = self.epochs
            val_loss_final = float('inf')

            # 2. Hyperparameter tuning loop
            if tuning_needed:
                X_subtrain, X_val = X_tensor[:split], X_tensor[split:]
                y_subtrain, y_val = y_tensor[:split], y_tensor[split:]
                
                best_mse = float('inf')
                
                for penalty in self.param_grid['penalty']:
                    self._set_seed(curr_seed) # Reset seed for consistent evaluation across grid
                    temp_model = _MLPNetwork(input_dim=input_dim, archi=self.archi, output_dim=output_dim)
                    temp_optimizer = optim.SGD(
                        temp_model.parameters(), lr=self.lr, momentum=self.momentum, 
                        nesterov=True, weight_decay=penalty
                    )
                    
                    early_stopper = EarlyStopping(patience=self.patience)
                    
                    for epoch in range(self.epochs):
                        temp_model.train()
                        temp_optimizer.zero_grad()
                        preds = temp_model(X_subtrain)
                        loss = self.criterion(preds, y_subtrain)
                        if penalty > 0:
                            l1_penalty = sum(p.abs().sum() for p in temp_model.parameters())
                            loss += penalty * l1_penalty
                        loss.backward()
                        temp_optimizer.step()
                    
                        # Early Stopping Check against validation set every epoch
                        temp_model.eval()
                        with torch.no_grad():
                            val_preds = temp_model(X_val)
                            val_mse = self.criterion(val_preds, y_val).item()
                            
                        early_stopper(val_mse, epoch)
                        if early_stopper.early_stop:
                            break
                        
                    if early_stopper.best_loss < best_mse:
                        best_mse = early_stopper.best_loss
                        best_penalty = penalty
                        current_epochs = early_stopper.best_epoch + 1 # +1 since 0-indexed
                
                val_loss_final = best_mse
                if run_idx == 0: # Store best params from the first seed for continuity
                    self.best_params_ = {'penalty': best_penalty, 'epochs': current_epochs}
            
            else:
                # If not tuning, fallback to best params or defaults
                if self.best_params_ is None:
                    self.best_params_ = {'penalty': self.param_grid['penalty'][0], 'epochs': self.epochs}
                best_penalty = self.best_params_['penalty']
                current_epochs = self.best_params_['epochs']
                
                # Without tuning, calculate a quick val loss on a split block 
                # (to allow fair sorting for the ensemble) 
                if split >= 10:
                    val_loss_final = self._simulate_val_loss(X_tensor, y_tensor, split, best_penalty, current_epochs, curr_seed)
                else:
                    val_loss_final = 0.0

            # 3. Initialize or re-initialize the model for full-dataset training
            self._set_seed(curr_seed)
            model = _MLPNetwork(input_dim=input_dim, archi=self.archi, output_dim=output_dim)
            optimizer = optim.SGD(
                model.parameters(), 
                lr=self.lr, 
                momentum=self.momentum, 
                nesterov=True, 
                weight_decay=best_penalty
            )

            # 4. Training Loop on full dataset
            model.train()
            for epoch in range(current_epochs):
                optimizer.zero_grad()
                predictions = model(X_tensor)
                loss = self.criterion(predictions, y_tensor)
                
                # Application of L1 penalty using the tuned parameter
                if best_penalty > 0:
                    l1_penalty = sum(p.abs().sum() for p in model.parameters())
                    loss += best_penalty * l1_penalty
                
                loss.backward()
                optimizer.step()
                
            run_results.append((val_loss_final, model))
            
        # Optional Ensembling Selection Logic: 
        # Sort models by validation loss and pick the best n_avg
        run_results.sort(key=lambda x: x[0]) 
        self.models = [m[1] for m in run_results[:self.n_avg]]

        self._fit_calls += 1
        return self
        
    def _simulate_val_loss(self, X_tensor, y_tensor, split, penalty, epochs, seed):
        """Helper to get a validation metric when tune_every prevents a full grid search."""
        self._set_seed(seed)
        model = _MLPNetwork(input_dim=X_tensor.shape[1], archi=self.archi, output_dim=y_tensor.shape[1])
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True, weight_decay=penalty)
        
        X_subtrain, X_val = X_tensor[:split], X_tensor[split:]
        y_subtrain, y_val = y_tensor[:split], y_tensor[split:]
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(X_subtrain)
            loss = self.criterion(preds, y_subtrain)
            if penalty > 0:
                loss += penalty * sum(p.abs().sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            return self.criterion(model(X_val), y_val).item()

    def predict(self, X):
        if not self.models:
            raise ValueError("This model instance is not fitted yet. Call 'fit' before 'predict'.")
            
        X_arr = self._extract_array(X, is_X=True)
        X_scaled = self.x_scaler.transform(X_arr)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Ensembling prediction: average outputs across best models
        preds_scaled_accum = None
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).numpy()
                if preds_scaled_accum is None:
                    preds_scaled_accum = pred
                else:
                    preds_scaled_accum += pred
                    
        preds_scaled = preds_scaled_accum / len(self.models)
            
        # Inverse transform the predictions to return back to raw scale
        preds = self.y_scaler.inverse_transform(preds_scaled)
            
        if preds.shape[1] == 1:
            return preds.flatten()
        return preds