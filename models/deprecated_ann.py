import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        if 'weight' not in name or 'output' in name:
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / (1.0 + 0.01 * step))
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
            scheduler.step()

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
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # Fit on all available history to prevent stale scaling in expanding windows
        scaler.fit(X_arr) 
        
        X_tr = scaler.transform(X_arr[:split])
        X_va = scaler.transform(X_arr[split:])
        scalers.append(scaler)
        
        X_train_tensors.append(torch.tensor(X_tr, dtype=torch.float32))
        X_val_tensors.append(torch.tensor(X_va, dtype=torch.float32))

    # Reshape but do not Scale Y to preserve implicit WLS MSE Loss 
    y_arr = y if y.ndim == 2 else y.reshape(-1, 1)
    
    y_tr = y_arr[:split]
    y_va = y_arr[split:]
    
    y_train_t = torch.tensor(y_tr, dtype=torch.float32)
    y_val_t = torch.tensor(y_va, dtype=torch.float32)

    return X_train_tensors, y_train_t, X_val_tensors, y_val_t, scalers, None


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
    
    return model, best_scalers, best_y_scaler, best_loss, best_params


# ---------- compact architectures using builder ----------
class ForwardRateNet(nn.Module):
    def __init__(self, n_in, n_out, archi=(3,), dropout_fwd=0.0, use_bn=True):
        super().__init__()
        # archi_fwd in the name helps _compute_regularization
        
        layers = []
        in_d = n_in
        for h in archi:
            if dropout_fwd:
                layers.append(nn.Dropout(dropout_fwd))
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU(inplace=True))
            in_d = h
            
        if dropout_fwd:
            layers.append(nn.Dropout(dropout_fwd))
            
        self.fwd_net = nn.Sequential(*layers)
        self.bn_merge = nn.BatchNorm1d(in_d) if use_bn else nn.Identity()
        self.output = nn.Linear(in_d, n_out)
        
    def forward(self, x):
        h = self.fwd_net(x)
        return self.output(self.bn_merge(h))

# ---------- public wrappers ----------
class ForwardRateANN:
    def __init__(self, archi=(3,), series='forward', do_grid_search=True, tune_every=60, warm_start=False, **kwargs):
        self.archi = archi
        self.series = series
        self.do_grid_search = do_grid_search
        self.tune_every = tune_every
        self.warm_start = warm_start
        self.param_grid = kwargs.pop('param_grid', {
            'Dropout': [0.0, 0.3], 
            'l1l2': [1e-3, 1e-4]
        })
        self.train_params = kwargs
        self._fit_count = 0
        self._best_params = None
        self._model = None
        self._scalers = None
        self._y_scaler = None
        self._last_val_loss = None

    def fit(self, X, y):
        X_arr = X[self.series].values
        y_arr = np.array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self._fit_count += 1
        
        def build_fn(dims, out, drop_m, drop_f): 
            return ForwardRateNet(dims[0], out, archi=self.archi, dropout_fwd=drop_f)

        do_tune = self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0
        if do_tune and not (self.warm_start and self._model is not None):
            self._model, self._scalers, self._y_scaler, self._last_val_loss, self._best_params = grid_search(
                build_fn, [X_arr], y_arr, self.param_grid, y_arr.shape[1], **self.train_params)

        else:
            if self._best_params is None:
                self._best_params = {}
            lm = self._best_params.get('l1l2', 1e-4)
            df = self._best_params.get('Dropout', 0.0)
            X_tr, y_tr, X_va, y_va, self._scalers, self._y_scaler = _split_and_scale([X_arr], y_arr)
            
            if not self.warm_start or self._model is None:
                self._model = build_fn([X_arr.shape[1]], y_arr.shape[1], 0.0, df)
                
            self._last_val_loss = train_model(self._model, X_tr, y_tr, X_va, y_va, l1l2_fwd=lm, l1l2_macro=lm, **self.train_params)
    
    def predict(self, X):
        X_scaled = self._scalers[0].transform(X[self.series].values)
        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            return pred_scaled.squeeze()

