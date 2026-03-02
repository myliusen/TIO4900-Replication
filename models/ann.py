import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from torch.utils.data import TensorDataset, DataLoader

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

# ---------- small utilities ----------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # store cpu clones of params to avoid device issues
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _compute_regularization(model, l1l2_macro, l1l2_fwd):
    """
    Applies Elastic Net (L1 + L2) regularization.
    l1l2_macro: scalar lambda for the macro tower
    l1l2_fwd:   scalar lambda for the fwd tower
    """
    l1_loss = 0.0
    l2_loss = 0.0
    
    for name, p in model.named_parameters():
        if 'weight' not in name:
            continue
            
        # Determine which penalty to use based on layer name
        is_macro = 'macro' in name.lower()
        reg_lambda = l1l2_macro if is_macro else l1l2_fwd
        
        # L1: Sum of absolute values
        l1_loss += reg_lambda * p.abs().sum()
        # L2: Sum of squares (Ridge / Weight Decay)
        l2_loss += reg_lambda * p.pow(2).sum()
        
    return l1_loss + l2_loss


def _make_dataloader(X_list, y_tensor, batch_size, shuffle):
    """
    X_list: list of torch tensors (all same length)
    returns: DataLoader that yields (xs_list, y)
    """
    dataset = TensorDataset(*X_list, y_tensor)
    # drop_last to avoid singleton batch BN issues
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ---------- simple reusable MLP builder ----------
def build_mlp(in_dim, out_dim, archi=(), dropout=0.0, final_activation=None, use_batchnorm=False):
    layers = []
    in_d = in_dim
    for h in archi:
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_d, h))
        layers.append(nn.ReLU(inplace=True))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        in_d = h
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_d, out_dim))
    if final_activation:
        layers.append(final_activation)
    return nn.Sequential(*layers)


# ---------- generic trainer used by all public wrappers ----------
def train_model(model, X_train_list, y_train, X_val_list, y_val,
                epochs=500, lr=0.015, momentum=0.9, 
                batch_size=32, patience=20, 
                l1l2_macro=1e-4, l1l2_fwd=1e-4, seed=42):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(DEVICE)

    # weight_decay=0 because we apply Custom Elastic Net in the loop
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=0, nesterov=True)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)

    train_tensors = [t.to(DEVICE) for t in X_train_list]
    val_tensors = [t.to(DEVICE) for t in X_val_list]
    y_train, y_val = y_train.to(DEVICE), y_val.to(DEVICE)

    train_dl = _make_dataloader(train_tensors, y_train, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            *batch_x, batch_y = batch
            optimizer.zero_grad()
            pred = model(*batch_x)
            
            mse = criterion(pred, batch_y)
            reg = _compute_regularization(model, l1l2_macro, l1l2_fwd)
            loss = mse + reg
            
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(*val_tensors)
            val_loss = float(criterion(val_pred, y_val).item())

        if es.step(val_loss, model): break

    es.restore(model)
    return es.best_loss


def _split_and_scale(X_arrays, y, val_frac=0.15):
    n = len(y)
    split = int(n * (1.0 - val_frac))

    scalers = []
    X_train_tensors = []
    X_val_tensors = []

    for X_arr in X_arrays:
        scaler = StandardScaler()
        # Fit on all available history to prevent stale scaling in expanding windows
        scaler.fit(X_arr) 
        
        X_tr = scaler.transform(X_arr[:split])
        X_va = scaler.transform(X_arr[split:])
        scalers.append(scaler)
        
        X_train_tensors.append(torch.tensor(X_tr, dtype=torch.float32))
        X_val_tensors.append(torch.tensor(X_va, dtype=torch.float32))

    # Reshape and Scale Y to prevent L1 regularization from zeroing out weights
    y_arr = y if y.ndim == 2 else y.reshape(-1, 1)
    
    y_scaler = StandardScaler()
    y_scaler.fit(y_arr)
    
    y_tr = y_scaler.transform(y_arr[:split])
    y_va = y_scaler.transform(y_arr[split:])
    
    y_train_t = torch.tensor(y_tr, dtype=torch.float32)
    y_val_t = torch.tensor(y_va, dtype=torch.float32)

    return X_train_tensors, y_train_t, X_val_tensors, y_val_t, scalers, y_scaler


def grid_search(build_fn, X_arrays, y_array, param_grid, n_out, seed=42, **train_kwargs):
    grid = list(ParameterGrid(param_grid))
    best = (np.inf, None, None, None, None)
    
    for params in grid:
        # Standardize parameter extraction (handles both single and dual grids)
        dm = params.get('Dropout_Macro', params.get('Dropout', 0.0))
        df = params.get('Dropout_Fwd', params.get('Dropout', 0.0))
        lm = params.get('L1L2_Macro', params.get('l1l2', 1e-4))
        lf = params.get('L1L2_Fwd', params.get('l1l2', 1e-4))

        X_tr, y_tr, X_va, y_va, scalers, y_scaler = _split_and_scale(X_arrays, y_array)
        
        # build_fn now expects (dims, out, drop_m, drop_f)
        model = build_fn([arr.shape[1] for arr in X_arrays], n_out, dm, df)
        
        val_loss = train_model(model, X_tr, y_tr, X_va, y_va, 
                               l1l2_macro=lm, l1l2_fwd=lf, seed=seed, **train_kwargs)
        
        if val_loss < best[0]:
            best = (val_loss, deepcopy(model), scalers, y_scaler, params)

    # Refit logic
    best_loss, _, best_scalers, best_y_scaler, best_params = best
    dm = best_params.get('Dropout_Macro', best_params.get('Dropout', 0.0))
    df = best_params.get('Dropout_Fwd', best_params.get('Dropout', 0.0))
    lm = best_params.get('L1L2_Macro', best_params.get('l1l2', 1e-4))
    lf = best_params.get('L1L2_Fwd', best_params.get('l1l2', 1e-4))
    
    X_tr, y_tr, X_va, y_va, _, _ = _split_and_scale(X_arrays, y_array)
    model = build_fn([arr.shape[1] for arr in X_arrays], n_out, dm, df)
    train_model(model, X_tr, y_tr, X_va, y_va, l1l2_macro=lm, l1l2_fwd=lf, seed=seed, **train_kwargs)
    
    return model, best_scalers, best_y_scaler, val_loss, best_params


# ---------- compact architectures using builder ----------
class ForwardRateNet(nn.Module):
    def __init__(self, n_in, n_out, archi=(3,), dropout_fwd=0.0, use_bn=False):
        super().__init__()
        # archi_fwd in the name helps _compute_regularization
        self.fwd_net = build_mlp(n_in, n_out, archi=archi, dropout=dropout_fwd, use_batchnorm=use_bn)
    def forward(self, x):
        return self.fwd_net(x)


class MacroFwdHybridNet(nn.Module):
    def __init__(self, n_macro, n_fwd, n_out, archi_macro=(32, 16, 8), archi_fwd=(3,), 
                 dropout_macro=0.1, dropout_fwd=0.0, use_bn=True):
        super().__init__()
        self.macro_tower = build_mlp(n_macro, archi_macro[-1], archi=archi_macro[:-1], 
                                     dropout=dropout_macro, use_batchnorm=False)
        self.fwd_tower = build_mlp(n_fwd, archi_fwd[-1], archi=archi_fwd[:-1], 
                                   dropout=dropout_fwd, use_batchnorm=False)
        merge_dim = archi_macro[-1] + archi_fwd[-1]
        self.output_layer = nn.Linear(merge_dim, n_out)
        self.bn_merge = nn.BatchNorm1d(merge_dim) if use_bn else nn.Identity()

    def forward(self, x_macro, x_fwd):
        merged = torch.cat([self.macro_tower(x_macro), self.fwd_tower(x_fwd)], dim=-1)
        return self.output_layer(self.bn_merge(merged))


class GroupEnsembleNet(nn.Module):
    def __init__(self, group_sizes, n_fwd, n_out, archi_group=(1,), archi_fwd=(3,), 
                 dropout_macro=0.1, dropout_fwd=0.0, use_bn=True):
        super().__init__()
        self.macro_towers = nn.ModuleList([
            build_mlp(g, archi_group[-1], archi=archi_group[:-1], 
                      dropout=dropout_macro, use_batchnorm=False) for g in group_sizes
        ])
        self.fwd_tower = build_mlp(n_fwd, archi_fwd[-1], archi=archi_fwd[:-1], 
                                   dropout=dropout_fwd, use_batchnorm=False)
        merge_dim = (len(group_sizes) * archi_group[-1]) + archi_fwd[-1]
        self.output_layer = nn.Linear(merge_dim, n_out)
        self.bn_merge = nn.BatchNorm1d(merge_dim) if use_bn else nn.Identity()

    def forward(self, *inputs):
        """
        inputs: a tuple of tensors (group1, group2, ..., groupK, fwd_rates)
        """
        # The last input is always the forward rates
        macro_inputs = inputs[:-1]
        fwd_input = inputs[-1]
        
        # Pass each macro group through its respective tower
        macro_outs = [tower(x) for tower, x in zip(self.macro_towers, macro_inputs)]
        
        # Pass fwd rates through its tower
        fwd_out = self.fwd_tower(fwd_input)
        
        # Concatenate everything
        merged = torch.cat(macro_outs + [fwd_out], dim=-1)
        
        return self.output_layer(self.bn_merge(merged))


# ---------- public wrappers ----------
class ForwardRateANN:
    def __init__(self, archi=(3,), series='forward', do_grid_search=True, tune_every=60, **kwargs):
        self.archi = archi
        self.series = series
        self.do_grid_search = do_grid_search
        self.tune_every = tune_every
        self.param_grid = kwargs.get('param_grid', {
            'Dropout': [0.0, 0.1, 0.3], 
            'l1l2': [1e-5, 1e-4]
        })
        self.train_params = kwargs
        self._fit_count = 0
        self._best_params = None
        self._model = None
        self._scalers = None
        self._y_scaler = None

    def fit(self, X, y):
        X_arr = X[self.series].values
        y_arr = np.array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self._fit_count += 1
        
        def build_fn(dims, out, drop_m, drop_f): 
            return ForwardRateNet(dims[0], out, archi=self.archi, dropout_fwd=drop_f)

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            self._model, self._scalers, self._y_scaler, _, self._best_params = grid_search(
                build_fn, [X_arr], y_arr, self.param_grid, y_arr.shape[1], **self.train_params)
        else:
            lm = self._best_params.get('l1l2', 1e-4)
            df = self._best_params.get('Dropout', 0.0)
            X_tr, y_tr, X_va, y_va, self._scalers, self._y_scaler = _split_and_scale([X_arr], y_arr)
            self._model = build_fn([X_arr.shape[1]], y_arr.shape[1], 0.0, df)
            train_model(self._model, X_tr, y_tr, X_va, y_va, l1l2_fwd=lm, l1l2_macro=lm, **self.train_params)
    
    def predict(self, X):
        X_scaled = self._scalers[0].transform(X[self.series].values)
        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            return self._y_scaler.inverse_transform(pred_scaled).squeeze()


class HybridANN:
    DEFAULT_GRID = {
        'Dropout_Macro': [0.1, 0.3, 0.5],
        'Dropout_Fwd':   [0.0],
        'L1L2_Macro':    [0.01, 0.001],
        'L1L2_Fwd':      [0.0001],
    }

    def __init__(self, archi_macro=(32, 16, 8), archi_fwd=(3,), 
                 do_grid_search=True, tune_every=60, **kwargs):
        self.archi_macro = archi_macro
        self.archi_fwd = archi_fwd
        self.do_grid_search = do_grid_search
        self.tune_every = tune_every
        
        self.param_grid = kwargs.get('param_grid', self.DEFAULT_GRID)
        self.train_params = kwargs # lr, momentum, epochs, etc.
        
        self._model = None
        self._scalers = None
        self._y_scaler = None
        self._fit_count = 0
        self._best_params = None

    def _select_features(self, X):
        """Standardizes input into [macro_array, fwd_array]"""
        return [X['fred'].values, X['forward'].values]

    def fit(self, X, y):
        inputs = self._select_features(X)
        y_arr = np.array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self._fit_count += 1
        
        def build_fn(dims, out, drop_m, drop_f):
            return MacroFwdHybridNet(dims[0], dims[1], out, 
                                     self.archi_macro, self.archi_fwd, 
                                     drop_m, drop_f)

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            self._model, self._scalers, self._y_scaler, _, self._best_params = grid_search(
                build_fn, inputs, y_arr, self.param_grid, y_arr.shape[1], **self.train_params
            )
        else:
            # Use found best params or fallback to kwargs
            lm = self._best_params.get('L1L2_Macro', 1e-3)
            lf = self._best_params.get('L1L2_Fwd', 1e-4)
            dm = self._best_params.get('Dropout_Macro', 0.1)
            df = self._best_params.get('Dropout_Fwd', 0.0)
            
            X_tr, y_tr, X_va, y_va, self._scalers, self._y_scaler = _split_and_scale(inputs, y_arr)
            self._model = build_fn([a.shape[1] for a in inputs], y_arr.shape[1], dm, df)
            
            train_model(self._model, X_tr, y_tr, X_va, y_va, 
                        l1l2_macro=lm, l1l2_fwd=lf, **self.train_params)

    def predict(self, X):
        raw = self._select_features(X)
        scaled_macro = torch.tensor(self._scalers[0].transform(raw[0]), dtype=torch.float32).to(DEVICE)
        scaled_fwd = torch.tensor(self._scalers[1].transform(raw[1]), dtype=torch.float32).to(DEVICE)
        
        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(scaled_macro, scaled_fwd).cpu().numpy()
            return self._y_scaler.inverse_transform(pred_scaled).squeeze()

    def __repr__(self):
        status = "Fitted" if self._model is not None else "Not Fitted"
        return f"--- HybridANN ({status}) ---\nArchi Macro: {self.archi_macro}\nArchi Fwd: {self.archi_fwd}\n{str(self._model) if self._model else ''}"
    

class GroupEnsembleANN:
    BIANCHI_GRID = {
        'Dropout_Macro': [0.1, 0.3, 0.5],
        'Dropout_Fwd':   [0.0],
        'L1L2_Macro':    [0.01, 0.001],
        'L1L2_Fwd':      [0.0001],
    }

    def __init__(self, archi_group=(1,), archi_fwd=(3,), do_grid_search=True, tune_every=60, **kwargs):
        self.archi_group = archi_group
        self.archi_fwd = archi_fwd
        self.do_grid_search = do_grid_search
        self.tune_every = tune_every
        self.param_grid = kwargs.get('param_grid', self.BIANCHI_GRID)
        self.train_params = kwargs
        self._fit_count = 0
        self._best_params = None
        self._group_names = None
        self._model = None
        self._scalers = None
        self._y_scaler = None

    def _select_features(self, X):
        if self._group_names is None:
            self._group_names = X['fred'].columns.get_level_values(0).unique().tolist()
        return [X['fred'][gn].values for gn in self._group_names] + [X['forward'].values]

    def fit(self, X, y):
        inputs = self._select_features(X)
        y_arr = np.array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self._fit_count += 1
        
        def build_fn(dims, out, drop_m, drop_f):
            return GroupEnsembleNet(dims[:-1], dims[-1], out, self.archi_group, self.archi_fwd, drop_m, drop_f)

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            self._model, self._scalers, self._y_scaler, _, self._best_params = grid_search(
                build_fn, inputs, y_arr, self.param_grid, y_arr.shape[1], **self.train_params)
        else:
            lm, lf = self._best_params.get('L1L2_Macro', 1e-3), self._best_params.get('L1L2_Fwd', 1e-4)
            dm, df = self._best_params.get('Dropout_Macro', 0.1), self._best_params.get('Dropout_Fwd', 0.0)
            X_tr, y_tr, X_va, y_va, self._scalers, self._y_scaler = _split_and_scale(inputs, y_arr)
            self._model = build_fn([a.shape[1] for a in inputs], y_arr.shape[1], dm, df)
            train_model(self._model, X_tr, y_tr, X_va, y_va, l1l2_macro=lm, l1l2_fwd=lf, **self.train_params)

    def predict(self, X):
        raw = self._select_features(X)
        scaled = [torch.tensor(s.transform(r), dtype=torch.float32).to(DEVICE) for s, r in zip(self._scalers, raw)]
        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(*scaled).cpu().numpy()
            return self._y_scaler.inverse_transform(pred_scaled).squeeze()

# WGNN

# ---------- 1. Custom Volatility Scaler ----------
class VolatilityScaler:
    """
    Scales target variables purely by their standard deviation (volatility) to match 
    Equation 12 of the paper, without mean-centering like StandardScaler.
    """
    def __init__(self):
        self.scale_ = None
        
    def fit(self, X):
        # Calculate standard deviation along the time axis (available training data)
        self.scale_ = np.std(X, axis=0)
        # Prevent division by zero just in case
        self.scale_ = np.where(self.scale_ == 0, 1e-8, self.scale_)
        return self
        
    def transform(self, X):
        return X / self.scale_
        
    def inverse_transform(self, X):
        return X * self.scale_


# ---------- 2. WLS Split & Scale Helper ----------
def _split_and_scale_wgnn(X_arrays, y, val_frac=0.15):
    """Modified split & scale that uses VolatilityScaler for y targets."""
    n = len(y)
    split = int(n * (1.0 - val_frac))

    scalers = []
    X_train_tensors = []
    X_val_tensors = []

    for X_arr in X_arrays:
        scaler = StandardScaler()
        scaler.fit(X_arr) 
        X_tr, X_va = scaler.transform(X_arr[:split]), scaler.transform(X_arr[split:])
        scalers.append(scaler)
        X_train_tensors.append(torch.tensor(X_tr, dtype=torch.float32))
        X_val_tensors.append(torch.tensor(X_va, dtype=torch.float32))

    y_arr = y if y.ndim == 2 else y.reshape(-1, 1)
    
    # Use Volatility Scaler for targets to implement WLS MSE Loss implicitly
    y_scaler = VolatilityScaler()
    y_scaler.fit(y_arr)
    
    y_tr, y_va = y_scaler.transform(y_arr[:split]), y_scaler.transform(y_arr[split:])
    y_train_t = torch.tensor(y_tr, dtype=torch.float32)
    y_val_t = torch.tensor(y_va, dtype=torch.float32)

    return X_train_tensors, y_train_t, X_val_tensors, y_val_t, scalers, y_scaler


# ---------- 3. WLS Grid Search Helper ----------
def grid_search_wgnn(build_fn, X_arrays, y_array, param_grid, n_out, seed=42, **train_kwargs):
    grid = list(ParameterGrid(param_grid))
    best = (np.inf, None, None, None, None)
    
    for params in grid:
        dm = params.get('Dropout_Macro', params.get('Dropout', 0.0))
        df = params.get('Dropout_Fwd', params.get('Dropout', 0.0))
        lm = params.get('L1L2_Macro', params.get('l1l2', 1e-4))
        lf = params.get('L1L2_Fwd', params.get('l1l2', 1e-4))

        X_tr, y_tr, X_va, y_va, scalers, y_scaler = _split_and_scale_wgnn(X_arrays, y_array)
        
        model = build_fn([arr.shape[1] for arr in X_arrays], n_out, dm, df)
        model.set_sigma(y_scaler.scale_) # Pass volatility vector to the network
        
        val_loss = train_model(model, X_tr, y_tr, X_va, y_va, 
                               l1l2_macro=lm, l1l2_fwd=lf, seed=seed, **train_kwargs)
        
        if val_loss < best[0]:
            best = (val_loss, deepcopy(model), scalers, y_scaler, params)

    best_loss, _, best_scalers, best_y_scaler, best_params = best
    dm = best_params.get('Dropout_Macro', best_params.get('Dropout', 0.0))
    df = best_params.get('Dropout_Fwd', best_params.get('Dropout', 0.0))
    lm = best_params.get('L1L2_Macro', best_params.get('l1l2', 1e-4))
    lf = best_params.get('L1L2_Fwd', best_params.get('l1l2', 1e-4))
    
    X_tr, y_tr, X_va, y_va, _, _ = _split_and_scale_wgnn(X_arrays, y_array)
    model = build_fn([arr.shape[1] for arr in X_arrays], n_out, dm, df)
    model.set_sigma(best_y_scaler.scale_)
    
    train_model(model, X_tr, y_tr, X_va, y_va, l1l2_macro=lm, l1l2_fwd=lf, seed=seed, **train_kwargs)
    
    return model, best_scalers, best_y_scaler, val_loss, best_params


# ---------- 4. New WGNN Architecture ----------
class WeightedGroupEnsembleNet(nn.Module):
    """
    Extends GroupEnsembleNet by incorporating the historical volatility scaling
    natively into the architecture to produce the predicted unscaled return.
    """
    def __init__(self, group_sizes, n_fwd, n_out, archi_group=(1,), archi_fwd=(3,), 
                 dropout_macro=0.1, dropout_fwd=0.0, use_bn=True):
        super().__init__()
        # Internal core architecture maps inputs to scaled predictions
        self.core_net = GroupEnsembleNet(group_sizes, n_fwd, n_out, archi_group, archi_fwd, 
                                         dropout_macro, dropout_fwd, use_bn)
        # Register the volatilities as a buffer so it saves cleanly with model states
        self.register_buffer('sigma', torch.ones(n_out))
        
    def set_sigma(self, sigma_np):
        """Injects \hat{\sigma}_t from the available historical training pool."""
        self.sigma = torch.tensor(sigma_np, dtype=torch.float32).view(-1)
        
    def forward(self, *inputs):
        """Outputs the predicted volatility-scaled return (\widetilde{rx}) used during MSE training."""
        return self.core_net(*inputs)
        
    def predict_unscaled(self, *inputs):
        """Equation 12 logic: Re-scales to raw returns via (\widetilde{rx} * \hat{\sigma})."""
        scaled_pred = self.forward(*inputs)
        return scaled_pred * self.sigma


# ---------- 5. WGNN Model Wrapper ----------
class WeightedGroupEnsembleANN:
    BIANCHI_GRID = {
        'Dropout_Macro': [0.1, 0.3, 0.5],
        'Dropout_Fwd':   [0.0],
        'L1L2_Macro':    [0.01, 0.001],
        'L1L2_Fwd':      [0.0001],
    }

    def __init__(self, archi_group=(1,), archi_fwd=(3,), do_grid_search=True, tune_every=60, **kwargs):
        self.archi_group = archi_group
        self.archi_fwd = archi_fwd
        self.do_grid_search = do_grid_search
        self.tune_every = tune_every
        self.param_grid = kwargs.get('param_grid', self.BIANCHI_GRID)
        self.train_params = kwargs
        self._fit_count = 0
        self._best_params = None
        self._group_names = None
        self._model = None
        self._scalers = None
        self._y_scaler = None

    def _select_features(self, X):
        if self._group_names is None:
            self._group_names = X['fred'].columns.get_level_values(0).unique().tolist()
        return [X['fred'][gn].values for gn in self._group_names] + [X['forward'].values]

    def fit(self, X, y):
        inputs = self._select_features(X)
        y_arr = np.array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self._fit_count += 1
        
        def build_fn(dims, out, drop_m, drop_f):
            return WeightedGroupEnsembleNet(dims[:-1], dims[-1], out, self.archi_group, self.archi_fwd, drop_m, drop_f)

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            self._model, self._scalers, self._y_scaler, _, self._best_params = grid_search_wgnn(
                build_fn, inputs, y_arr, self.param_grid, y_arr.shape[1], **self.train_params)
        else:
            lm, lf = self._best_params.get('L1L2_Macro', 1e-3), self._best_params.get('L1L2_Fwd', 1e-4)
            dm, df = self._best_params.get('Dropout_Macro', 0.1), self._best_params.get('Dropout_Fwd', 0.0)
            X_tr, y_tr, X_va, y_va, self._scalers, self._y_scaler = _split_and_scale_wgnn(inputs, y_arr)
            
            self._model = build_fn([a.shape[1] for a in inputs], y_arr.shape[1], dm, df)
            self._model.set_sigma(self._y_scaler.scale_) # Pass estimated \hat{\sigma}_t
            
            train_model(self._model, X_tr, y_tr, X_va, y_va, l1l2_macro=lm, l1l2_fwd=lf, **self.train_params)

    def predict(self, X):
        raw = self._select_features(X)
        scaled = [torch.tensor(s.transform(r), dtype=torch.float32).to(DEVICE) for s, r in zip(self._scalers, raw)]
        self._model.eval()
        with torch.no_grad():
            # Utilize the architecture's native scaling method to get unscaled bond predictions
            pred_unscaled = self._model.predict_unscaled(*scaled).cpu().numpy()
            return pred_unscaled.squeeze()