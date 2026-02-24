import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
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


def _l1_penalty(model, l1):
    if not l1:
        return 0.0
    loss = 0.0
    for name, p in model.named_parameters():
        if 'weight' in name:
            loss = loss + p.abs().sum()
    return l1 * loss


def _make_dataloader(X_list, y_tensor, batch_size, shuffle):
    """
    X_list: list of torch tensors (all same length)
    returns: DataLoader that yields (xs_list, y)
    """
    # concat inputs into single tensor with an input index if simpler —
    # simpler approach: store tuple per sample: DataLoader(tensor_dataset) can take multiple tensors
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


# ---------- compact architectures using builder ----------
class ForwardRateNet(nn.Module):
    def __init__(self, n_in, n_out, archi=(3,), dropout=0.0, use_bn=False):
        super().__init__()
        # archi is now a tuple, e.g., (3,) or (5, 4, 3)
        self.net = build_mlp(n_in, n_out, archi=archi, dropout=dropout, use_batchnorm=use_bn)
        
    def forward(self, x):
        return self.net(x)


class MacroFwdHybridNet(nn.Module):
    def __init__(self, n_macro, n_fwd, n_out, archi_macro=(32, 16, 8), archi_fwd=(3,), dropout=0.0, use_bn=True):
        super().__init__()
        
        # Tower 1: Macro Branch (The bottom part of diagram b)
        # We output to the last element of archi_macro
        self.macro_tower = build_mlp(n_macro, archi_macro[-1], archi=archi_macro[:-1], 
                                     dropout=dropout, use_batchnorm=False)
        
        # Tower 2: Forward Rate Branch (The top part of diagram b)
        # Bianchi often uses a very shallow (3 nodes) branch here
        self.fwd_tower = build_mlp(n_fwd, archi_fwd[-1], archi=archi_fwd[:-1], 
                                   dropout=dropout, use_batchnorm=False)
        
        # Merge layer
        # Concatenates the outputs of both towers
        merge_in_dim = archi_macro[-1] + archi_fwd[-1]
        
        # Final fully connected output layer
        self.output_layer = nn.Linear(merge_in_dim, n_out)
        
        # Optional BN on the merged representation
        self.bn_merge = nn.BatchNorm1d(merge_in_dim) if use_bn else nn.Identity()

    def forward(self, x_macro, x_fwd):
        # Process branches
        h_macro = self.macro_tower(x_macro)
        h_fwd = self.fwd_tower(x_fwd)
        
        # Concatenate on the feature dimension
        merged = torch.cat([h_macro, h_fwd], dim=-1)
        
        # Final pass
        return self.output_layer(self.bn_merge(merged))


class GroupEnsembleNet(nn.Module):
    def __init__(self, group_sizes, n_fwd, n_out, archi_group=(1,), archi_fwd=(3,), dropout=0.0, use_bn=True):
        """
        group_sizes: list of ints, number of features in each FRED group
        n_fwd: number of forward rate features
        """
        super().__init__()
        
        # 1. Create a "Tower" for each Macro Group
        # Bianchi often uses archi_group=(1,) -> reduces each group to a single factor
        self.macro_towers = nn.ModuleList([
            build_mlp(g_size, archi_group[-1], archi=archi_group[:-1], 
                      dropout=dropout, use_batchnorm=False)
            for g_size in group_sizes
        ])
        
        # 2. Create a "Tower" for Forward Rates
        self.fwd_tower = build_mlp(n_fwd, archi_fwd[-1], archi=archi_fwd[:-1], 
                                   dropout=dropout, use_batchnorm=False)
        
        # 3. Final Output Layer
        # Total incoming features = (Number of Macro Groups * latent_dim) + fwd_latent_dim
        merge_in_dim = (len(group_sizes) * archi_group[-1]) + archi_fwd[-1]
        self.output_layer = nn.Linear(merge_in_dim, n_out)
        self.bn_merge = nn.BatchNorm1d(merge_in_dim) if use_bn else nn.Identity()

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

# ---------- generic trainer used by all public wrappers ----------
def train_model(model, X_train_list, y_train, X_val_list, y_val,
                epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                batch_size=32, patience=20, l1=0.0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)

    # create dataloaders
    train_tensors = [t.to(DEVICE) for t in X_train_list]
    val_tensors = [t.to(DEVICE) for t in X_val_list]
    y_train = y_train.to(DEVICE)
    y_val = y_val.to(DEVICE)

    train_dl = _make_dataloader(train_tensors, y_train, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            *batch_x, batch_y = batch
            batch_x = [bx.to(DEVICE) for bx in batch_x]
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(*batch_x)
            loss = criterion(pred, batch_y) + _l1_penalty(model, l1)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_pred = model(*val_tensors)
            val_loss = float(criterion(val_pred, y_val).item())

        if es.step(val_loss, model):
            break

    es.restore(model)
    return es.best_loss


# ---------- unified grid search ----------
def grid_search(build_fn, X_arrays, y_array, param_grid, n_out, seed=42, **train_kwargs):
    grid = list(ParameterGrid(param_grid))
    best = (np.inf, None, None, None)  # (loss, model, scalers, params)
    for params in grid:
        dropout = params['Dropout']
        l1l2 = params['l1l2']

        # split/scale
        X_tr, y_tr, X_va, y_va, scalers = _split_and_scale(X_arrays, y_array)
        model = build_fn([arr.shape[1] for arr in X_arrays], n_out, dropout, params=params)
        val_loss = train_model(model, X_tr, y_tr, X_va, y_va, l1=l1l2, seed=seed, **train_kwargs)
        if val_loss < best[0]:
            best = (val_loss, deepcopy(model), scalers, params)

    # refit best on same split to mirror original behaviour
    best_loss, _, best_scalers, best_params = best
    dropout = best_params['Dropout']
    l1l2 = best_params['l1l2']
    X_tr, y_tr, X_va, y_va, scalers = _split_and_scale(X_arrays, y_array)
    model = build_fn([arr.shape[1] for arr in X_arrays], n_out, dropout, params=best_params)
    val_loss = train_model(model, X_tr, y_tr, X_va, y_va, l1=l1l2, seed=seed, **train_kwargs)
    return model, scalers, val_loss, best_params


def _split_and_scale(X_arrays, y, val_frac=0.15):
    n = len(y)
    split = int(n * (1.0 - val_frac))

    scalers = []
    X_train_tensors = []
    X_val_tensors = []

    for X_arr in X_arrays:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_tr = scaler.fit_transform(X_arr[:split])
        X_va = scaler.transform(X_arr[split:])
        scalers.append(scaler)
        X_train_tensors.append(torch.tensor(X_tr, dtype=torch.float32))
        X_val_tensors.append(torch.tensor(X_va, dtype=torch.float32))

    y_arr = y if y.ndim == 2 else y.reshape(-1, 1)
    y_train_t = torch.tensor(y_arr[:split], dtype=torch.float32)
    y_val_t = torch.tensor(y_arr[split:], dtype=torch.float32)

    return X_train_tensors, y_train_t, X_val_tensors, y_val_t, scalers


class ForwardRateANN:
    DEFAULT_GRID = {
        'Dropout': [0.0, 0.1, 0.3, 0.5],
        'l1l2':    [1e-5, 1e-4, 1e-3],
    }

    def __init__(self, archi=(3,), series='forward',
                 do_grid_search=True, param_grid=None,
                 epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                 batch_size=32, patience=20, tune_every=60, seed=42,
                 dropout=0.0, l1l2=1e-4):

        self.archi = archi # Store tuple here
        self.series = series
        self.do_grid_search = do_grid_search
        self.param_grid = param_grid or self.DEFAULT_GRID
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.tune_every = tune_every
        self.seed = seed
        self.dropout = dropout
        self.l1l2 = l1l2

        self._model = None
        self._scalers = None
        self._fit_count = 0
        self._best_params = None


    def _select_features(self, X):
        if hasattr(X, 'columns') and hasattr(X.columns, 'get_level_values'):
            source = X.columns.get_level_values(0)
            return X.loc[:, source == self.series].values
        return X.values if hasattr(X, 'values') else np.array(X)


    def _build_model(self, n_inputs, n_out, dropout):
        # Pass the fixed architecture tuple to the network
        return ForwardRateNet(n_inputs, n_out,
                              archi=self.archi,
                              dropout=dropout,
                              use_bn=True)

    def fit(self, X, y):
        X_arr = self._select_features(X)
        y_arr = y if isinstance(y, np.ndarray) else np.array(y)
        n_out = y_arr.shape[1] if y_arr.ndim == 2 else 1

        self._fit_count += 1
        train_kwargs = dict(
            epochs=self.epochs, lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size, patience=self.patience
        )

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            print(f"Grid search triggered for ForwardRateANN (Fit #{self._fit_count})...")

            # We fix archi, but grid search optimizes Dropout and L1/L2
            def build_fn(n_inputs_list, n_out_, dropout_val, params=None):
                return self._build_model(n_inputs_list[0], n_out_, dropout_val)

            self._model, self._scalers, _, self._best_params = grid_search(
                build_fn, [X_arr], y_arr, self.param_grid, n_out,
                seed=self.seed, **train_kwargs
            )
        else:
            # Use previously found best hyperparameters or defaults
            dropout = self._best_params['Dropout'] if self._best_params else self.dropout
            l1l2 = self._best_params['l1l2'] if self._best_params else self.l1l2

            X_tr, y_tr, X_va, y_va, self._scalers = _split_and_scale([X_arr], y_arr)
            self._model = self._build_model(X_arr.shape[1], n_out, dropout)

            train_model(self._model, X_tr, y_tr, X_va, y_va,
                        l1=l1l2, seed=self.seed, **train_kwargs)

    def predict(self, X):
        X_arr = self._select_features(X)
        X_scaled = self._scalers[0].transform(X_arr)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t).cpu().numpy()

        return pred.squeeze()
    
    
class HybridANN:
    DEFAULT_GRID = {
        'Dropout': [0.0, 0.1, 0.3, 0.5],
        'l1l2':    [1e-5, 1e-4, 1e-3],
    }

    def __init__(self, archi_macro=(32, 16, 8), archi_fwd=(3,), 
                 do_grid_search=True, param_grid=None,
                 epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                 batch_size=32, patience=20, tune_every=60, seed=42,
                 dropout=0.0, l1l2=1e-4):
        
        self.archi_macro = archi_macro
        self.archi_fwd = archi_fwd
        
        self.do_grid_search = do_grid_search
        self.param_grid = param_grid or self.DEFAULT_GRID
        
        # Training Parameters
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.tune_every = tune_every
        self.seed = seed
        
        # Current best/default hyperparameters
        self.dropout = dropout
        self.l1l2 = l1l2

        self._model = None
        self._scalers = None 
        self._fit_count = 0
        self._best_params = None

    def _select_features(self, X):
        """Extracts the macro and forward components from the MultiIndex DataFrame."""
        macro = X['fred'].values
        fwd = X['forward'].values
        return [macro, fwd]

    def _build_model(self, n_macro, n_fwd, n_out, dropout):
        return MacroFwdHybridNet(n_macro, n_fwd, n_out, 
                                 archi_macro=self.archi_macro,
                                 archi_fwd=self.archi_fwd,
                                 dropout=dropout,
                                 use_bn=True)

    def fit(self, X, y):
        inputs_list = self._select_features(X) # [macro_arr, fwd_arr]
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        n_out = y_arr.shape[1] if y_arr.ndim == 2 else 1

        self._fit_count += 1
        
        train_kwargs = dict(
            epochs=self.epochs, lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size, patience=self.patience
        )

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            print(f"Grid search triggered for HybridANN (Fit #{self._fit_count})...")
            
            def build_fn(in_dims, out_dim, drop_val, params=None):
                return self._build_model(in_dims[0], in_dims[1], out_dim, drop_val)

            self._model, self._scalers, _, self._best_params = grid_search(
                build_fn, inputs_list, y_arr, self.param_grid, n_out,
                seed=self.seed, **train_kwargs
            )
        else:
            # Use current best params
            dropout = self._best_params['Dropout'] if self._best_params else self.dropout
            l1l2 = self._best_params['l1l2'] if self._best_params else self.l1l2

            X_tr, y_tr, X_va, y_va, self._scalers = _split_and_scale(inputs_list, y_arr)
            
            self._model = self._build_model(inputs_list[0].shape[1], 
                                            inputs_list[1].shape[1], 
                                            n_out, dropout)
            
            train_model(self._model, X_tr, y_tr, X_va, y_va, 
                        l1=l1l2, seed=self.seed, **train_kwargs)

    def predict(self, X):
        macro_raw, fwd_raw = self._select_features(X)
        
        # Scaling
        macro_s = self._scalers[0].transform(macro_raw)
        fwd_s = self._scalers[1].transform(fwd_raw)
        
        t_macro = torch.tensor(macro_s, dtype=torch.float32).to(DEVICE)
        t_fwd = torch.tensor(fwd_s, dtype=torch.float32).to(DEVICE)
        
        self._model.eval()
        with torch.no_grad():
            pred = self._model(t_macro, t_fwd).cpu().numpy()
        return pred.squeeze()
    

class GroupEnsembleANN:
    DEFAULT_GRID = {
        'Dropout': [0.0, 0.1, 0.3, 0.5],
        'l1l2':    [1e-5, 1e-4, 1e-3],
    }

    def __init__(self, archi_group=(1,), archi_fwd=(3,),
                 do_grid_search=True, param_grid=None,
                 epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                 batch_size=32, patience=20, tune_every=60, seed=42):
        
        self.archi_group = archi_group
        self.archi_fwd = archi_fwd
        self.do_grid_search = do_grid_search
        self.param_grid = param_grid or self.DEFAULT_GRID
        self.epochs, self.lr, self.momentum = epochs, lr, momentum
        self.weight_decay, self.batch_size, self.patience = weight_decay, batch_size, patience
        self.tune_every, self.seed = tune_every, seed

        self._model = None
        self._scalers = None
        self._fit_count = 0
        self._best_params = None
        self._group_names = None

    def _select_features(self, X):
        """Returns a list of arrays: [group1, group2, ..., groupK, forward]"""
        # 1. Identify FRED groups using the second level of the MultiIndex
        fred_df = X['fred']
        if self._group_names is None:
            self._group_names = fred_df.columns.get_level_values(0).unique().tolist()
        
        # Extract each macro group
        inputs = [fred_df[gn].values for gn in self._group_names]
        
        # 2. Append the forward rates
        inputs.append(X['forward'].values)
        return inputs

    def _build_model(self, group_sizes_list, n_out, dropout):
        # group_sizes_list contains [size_g1, size_g2, ..., size_fwd]
        macro_sizes = group_sizes_list[:-1]
        fwd_size = group_sizes_list[-1]
        return GroupEnsembleNet(macro_sizes, fwd_size, n_out, 
                                archi_group=self.archi_group, 
                                archi_fwd=self.archi_fwd, 
                                dropout=dropout)

    def fit(self, X, y):
        inputs_list = self._select_features(X)
        y_arr = y.values if hasattr(y, 'values') else np.array(y)
        n_out = y_arr.shape[1] if y_arr.ndim == 2 else 1
        self._fit_count += 1

        train_kwargs = dict(epochs=self.epochs, lr=self.lr, momentum=self.momentum,
                            weight_decay=self.weight_decay, batch_size=self.batch_size, 
                            patience=self.patience)

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            print(f"Grid search triggered for GroupEnsembleANN...")
            def build_fn(dims, out, drop, params=None):
                return self._build_model(dims, out, drop)
            
            self._model, self._scalers, _, self._best_params = grid_search(
                build_fn, inputs_list, y_arr, self.param_grid, n_out, seed=self.seed, **train_kwargs
            )
        else:
            drop = self._best_params['Dropout'] if self._best_params else 0.0
            l1 = self._best_params['l1l2'] if self._best_params else 1e-4
            X_tr, y_tr, X_va, y_va, self._scalers = _split_and_scale(inputs_list, y_arr)
            self._model = self._build_model([arr.shape[1] for arr in inputs_list], n_out, drop)
            train_model(self._model, X_tr, y_tr, X_va, y_va, l1=l1, seed=self.seed, **train_kwargs)

    def predict(self, X):
        inputs_raw = self._select_features(X)
        scaled_tensors = []
        for i, arr in enumerate(inputs_raw):
            scaled = self._scalers[i].transform(arr)
            scaled_tensors.append(torch.tensor(scaled, dtype=torch.float32).to(DEVICE))
        
        self._model.eval()
        with torch.no_grad():
            pred = self._model(*scaled_tensors).cpu().numpy()
        return pred.squeeze()

    def __repr__(self):
        status = "Fitted" if self._model is not None else "Not Fitted"
        return f"--- GroupEnsembleANN ({status}) ---\nGroups: {self._group_names}\n{str(self._model) if self._model else ''}"