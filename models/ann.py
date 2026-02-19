"""
PyTorch reimplementations of Bianchi's neural network architectures.

Three architectures:
  - ForwardRateANN:  Flat forward rates only (no macro, no groups)
  - ExogANN:         Deep macro net merged with yields at final layer
  - EnsembleExogANN: Per-group macro sub-networks merged with yields

All share the same .fit() / .predict() API.
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

DEVICE = 'cpu'


# ═══════════════════════════════════════════════════════════════════════
#  Shared training infrastructure
# ═══════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Mirror Keras EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=20)."""
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
            self.best_state = deepcopy(model.state_dict())
            return False  # don't stop
        self.counter += 1
        return self.counter >= self.patience  # stop if patience exceeded

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _train_loop(model, X_inputs_train, y_train, X_inputs_val, y_val,
                epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                batch_size=32, patience=20, seed=42):
    """
    Generic training loop matching Bianchi's Keras setup:
      - SGD with momentum + Nesterov + weight decay
      - Early stopping on val_loss with checkpoint
      - Shuffle training batches each epoch
    
    Parameters
    ----------
    model : nn.Module
    X_inputs_train : list of Tensors (one per model input)
    y_train : Tensor
    X_inputs_val : list of Tensors
    y_val : Tensor
    
    Returns
    -------
    best_val_loss : float
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)

    n_train = y_train.shape[0]

    for epoch in range(epochs):
        model.train()

        # Shuffle training data (like Keras shuffle=True)
        perm = torch.randperm(n_train)
        X_shuf = [x[perm] for x in X_inputs_train]
        y_shuf = y_train[perm]

        # Mini-batch training
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            if (end - start) < 2 and n_train > 1:
                continue # Skip singleton batch to prevent BatchNorm crash

            X_batch = [x[start:end] for x in X_shuf]
            y_batch = y_shuf[start:end]

            optimizer.zero_grad()
            pred = model(*X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(*X_inputs_val)
            val_loss = criterion(val_pred, y_val).item()

        if es.step(val_loss, model):
            break

    es.restore(model)
    return es.best_loss


def _split_and_scale(X_arrays, y, val_frac=0.15):
    """
    Split into train/val (last val_frac as validation, matching Keras validation_split).
    Scale each X array with MinMaxScaler(-1, 1) fit on training portion only.
    
    Parameters
    ----------
    X_arrays : list of np.ndarray, each shape (T, n_features_i)
    y : np.ndarray, shape (T,) or (T, n_out)
    
    Returns
    -------
    X_train_tensors, y_train_tensor, X_val_tensors, y_val_tensor, scalers
    """
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


def _grid_search(build_fn, X_arrays, y, param_grid, n_out,
                 seed=42, **train_kwargs):
    """
    Grid search over param_grid, selecting by val_loss.
    
    Parameters
    ----------
    build_fn : callable(n_inputs_list, n_out, dropout, l1l2) -> nn.Module
        Factory function that builds a fresh model.
    X_arrays : list of np.ndarray
    y : np.ndarray
    param_grid : dict with keys 'Dropout' and 'l1l2' (lists of values)
    n_out : int
    
    Returns
    -------
    best_model, best_scalers, best_val_loss, best_params
    """
    grid = list(ParameterGrid(param_grid))
    best_val_loss = np.inf
    best_result = None

    n_inputs_list = [arr.shape[1] for arr in X_arrays]

    for params in grid:
        dropout = params['Dropout']
        l1l2 = params['l1l2']

        X_tr, y_tr, X_va, y_va, scalers = _split_and_scale(X_arrays, y)
        model = build_fn(n_inputs_list, n_out, dropout, l1l2)

        val_loss = _train_loop(
            model, X_tr, y_tr, X_va, y_va,
            seed=seed, **train_kwargs
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_result = (deepcopy(model), scalers, val_loss, params)

    # Refit best params on the same split (matching Bianchi's refit of best)
    dropout = best_result[3]['Dropout']
    l1l2 = best_result[3]['l1l2']
    X_tr, y_tr, X_va, y_va, scalers = _split_and_scale(X_arrays, y)
    model = build_fn(n_inputs_list, n_out, dropout, l1l2)
    val_loss = _train_loop(model, X_tr, y_tr, X_va, y_va, seed=seed, **train_kwargs)

    return model, scalers, val_loss, best_result[3]


# ═══════════════════════════════════════════════════════════════════════
#  L1/L2 regularisation helper (Bianchi uses keras regularizers.l1_l2)
# ═══════════════════════════════════════════════════════════════════════

def _l1l2_loss(model, l1l2):
    """
    Compute L1+L2 penalty on all weight (not bias) parameters.
    Keras l1_l2(l) applies l*|w| + l*w² per weight.
    PyTorch SGD weight_decay already handles L2, so we only add L1 here.
    """
    if l1l2 is None or l1l2 == 0:
        return 0.0
    l1_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_loss += param.abs().sum()
    return l1l2 * l1_loss


def _train_loop_with_l1(model, X_inputs_train, y_train, X_inputs_val, y_val,
                         l1l2=0.0, epochs=500, lr=0.015, momentum=0.9,
                         weight_decay=0.01, batch_size=32, patience=20, seed=42):
    """
    Training loop with explicit L1 regularisation added to loss.
    L2 is handled by SGD weight_decay.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=patience)

    n_train = y_train.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        X_shuf = [x[perm] for x in X_inputs_train]
        y_shuf = y_train[perm]

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)

            if (end - start) < 2 and n_train > 1:
                continue # Skip singleton batch to prevent BatchNorm crash

            X_batch = [x[start:end] for x in X_shuf]
            y_batch = y_shuf[start:end]

            optimizer.zero_grad()
            pred = model(*X_batch)
            loss = criterion(pred, y_batch) + _l1l2_loss(model, l1l2)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(*X_inputs_val)
            val_loss = criterion(val_pred, y_val).item()

        if es.step(val_loss, model):
            break

    es.restore(model)
    return es.best_loss


def _grid_search_l1(build_fn, X_arrays, y, param_grid, n_out,
                    seed=42, **train_kwargs):
    """Grid search that passes l1l2 into the training loop (not just the model)."""
    grid = list(ParameterGrid(param_grid))
    best_val_loss = np.inf
    best_result = None

    n_inputs_list = [arr.shape[1] for arr in X_arrays]

    for params in grid:
        dropout = params['Dropout']
        l1l2 = params['l1l2']

        X_tr, y_tr, X_va, y_va, scalers = _split_and_scale(X_arrays, y)
        model = build_fn(n_inputs_list, n_out, dropout)

        val_loss = _train_loop_with_l1(
            model, X_tr, y_tr, X_va, y_va,
            l1l2=l1l2, seed=seed, **train_kwargs
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_result = (deepcopy(model), scalers, val_loss, params)

    # Refit best
    dropout = best_result[3]['Dropout']
    l1l2 = best_result[3]['l1l2']
    X_tr, y_tr, X_va, y_va, scalers = _split_and_scale(X_arrays, y)
    model = build_fn(n_inputs_list, n_out, dropout)
    val_loss = _train_loop_with_l1(
        model, X_tr, y_tr, X_va, y_va,
        l1l2=l1l2, seed=seed, **train_kwargs
    )

    return model, scalers, val_loss, best_result[3]


# ═══════════════════════════════════════════════════════════════════════
#  Network architectures (nn.Module)
# ═══════════════════════════════════════════════════════════════════════

class _ForwardRateNet(nn.Module):
    """Simple feedforward net: input → [Dropout → Dense(h) → ReLU] → output."""
    def __init__(self, n_in, n_out, hidden_size=3, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_in, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, n_out),
        )

    def forward(self, x):
        return self.net(x)


class _ExogNet(nn.Module):
    """
    NNExogGeneric: Deep macro network merged with yields.
    
    Macro: Input → [Dropout → Dense → ReLU] × len(archi) layers
    Yields: Input (raw)
    Merge: Concat(macro_hidden, yields) → Dropout → BatchNorm → Dense(output)
    """
    def __init__(self, n_macro, n_yields, n_out, archi=(32, 16, 8), dropout=0.0):
        super().__init__()

        # Build macro layers
        macro_layers = []
        in_dim = n_macro
        for h in archi:
            macro_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_dim, h),
                nn.ReLU(),
            ])
            in_dim = h
        self.macro_net = nn.Sequential(*macro_layers)

        # Merge layer
        merge_dim = archi[-1] + n_yields
        self.merge_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(merge_dim),
            nn.Linear(merge_dim, n_out),
        )

    def forward(self, x_macro, x_yields):
        h = self.macro_net(x_macro)
        merged = torch.cat([h, x_yields], dim=-1)
        return self.merge_net(merged)


class _EnsembleExogNet(nn.Module):
    """
    NNEnsemExogGeneric: Per-group sub-networks merged with yields.
    
    Each group: Input_g → [Dropout → Dense(archi[0]) → ReLU] × len(archi) layers
    Merge groups: Concat(group_1_out, ..., group_K_out)
    Add yields:   Concat(groups_merged, yields) → Dropout → BatchNorm → Dense(output)
    """
    def __init__(self, group_sizes, n_yields, n_out, archi=(1,), dropout=0.0):
        """
        Parameters
        ----------
        group_sizes : list of int
            Number of features in each macro group.
        n_yields : int
        n_out : int
        archi : tuple of int
            Hidden layer sizes for each sub-network. Default (1,) = single
            hidden layer compressing each group to 1 unit.
        """
        super().__init__()
        self.n_groups = len(group_sizes)

        # Build one sub-network per group
        self.group_nets = nn.ModuleList()
        for g_size in group_sizes:
            layers = []
            in_dim = g_size
            for h in archi:
                layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(in_dim, h),
                    nn.ReLU(),
                ])
                in_dim = h
            self.group_nets.append(nn.Sequential(*layers))

        # Merge: all group outputs + yields
        merge_dim = archi[-1] * len(group_sizes) + n_yields
        self.merge_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(merge_dim),
            nn.Linear(merge_dim, n_out),
        )

    def forward(self, *inputs):
        """
        inputs: (group_1, group_2, ..., group_K, yields)
        Each is a (batch, n_features) tensor.
        """
        group_inputs = inputs[:-1]
        x_yields = inputs[-1]

        group_outputs = []
        for net, x_g in zip(self.group_nets, group_inputs):
            group_outputs.append(net(x_g))

        merged_groups = torch.cat(group_outputs, dim=-1)
        merged_all = torch.cat([merged_groups, x_yields], dim=-1)
        return self.merge_net(merged_all)


# ═══════════════════════════════════════════════════════════════════════
#  Public model classes with .fit() / .predict() API
# ═══════════════════════════════════════════════════════════════════════

class ForwardRateANN:
    """
    Simple one-hidden-layer network on forward rates only.
    Equivalent to a simplified NNExogGeneric with no macro variables.
    """

    DEFAULT_GRID = {
        'Dropout': [0.0, 0.1, 0.3, 0.5],
        'l1l2':    [1e-5, 1e-4, 1e-3],
    }

    def __init__(self, hidden_size=3, series='forward',
                 do_grid_search=True, param_grid=None,
                 epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                 batch_size=32, patience=20, tune_every=48, seed=42,
                 # Fixed hyperparams when not doing grid search:
                 dropout=0.0, l1l2=1e-4):
        self.hidden_size = hidden_size
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
        """Extract the relevant series from X based on MultiIndex."""
        if hasattr(X, 'columns') and hasattr(X.columns, 'get_level_values'):
            source = X.columns.get_level_values(0)
            mask = source == self.series
            return X.loc[:, mask].values
        return X.values if hasattr(X, 'values') else np.array(X)

    def fit(self, X, y):
        X_arr = self._select_features(X)
        y_arr = y if isinstance(y, np.ndarray) else np.array(y)
        n_out = y_arr.shape[1] if y_arr.ndim == 2 else 1

        self._fit_count += 1
        train_kwargs = dict(
            epochs=self.epochs, lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            patience=self.patience,
        )

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            # Re-tune hyperparameters
            def build_fn(n_inputs_list, n_out_, dropout):
                return _ForwardRateNet(n_inputs_list[0], n_out_,
                                       self.hidden_size, dropout)

            self._model, self._scalers, _, self._best_params = _grid_search_l1(
                build_fn, [X_arr], y_arr, self.param_grid, n_out,
                seed=self.seed, **train_kwargs
            )
        else:
            # Use current best params (or defaults)
            dropout = self._best_params['Dropout'] if self._best_params else self.dropout
            l1l2 = self._best_params['l1l2'] if self._best_params else self.l1l2

            X_tr, y_tr, X_va, y_va, self._scalers = _split_and_scale([X_arr], y_arr)
            self._model = _ForwardRateNet(X_arr.shape[1], n_out,
                                          self.hidden_size, dropout)
            _train_loop_with_l1(
                self._model, X_tr, y_tr, X_va, y_va,
                l1l2=l1l2, seed=self.seed, **train_kwargs
            )

    def predict(self, X):
        X_arr = self._select_features(X)
        X_scaled = self._scalers[0].transform(X_arr)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(X_t).numpy()
        return pred.squeeze()


class ExogANN:
    """
    NNExogGeneric equivalent: deep macro network merged with yields.
    
    Bianchi's NN3LayerExog uses archi=[32, 16, 8].
    """

    DEFAULT_GRID = {
        'Dropout': [0.0, 0.1, 0.3, 0.5],
        'l1l2':    [1e-5, 1e-4, 1e-3],
    }

    def __init__(self, archi=(32, 16, 8), macro_series='fred',
                 yield_series='yields', do_grid_search=True, param_grid=None,
                 epochs=500, lr=0.01, momentum=0.9, weight_decay=0.01,
                 batch_size=32, patience=20, tune_every=48, seed=42,
                 dropout=0.0, l1l2=1e-4):
        self.archi = archi
        self.macro_series = macro_series
        self.yield_series = yield_series
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
        """Return (macro_array, yields_array)."""
        source = X.columns.get_level_values(0)
        X_macro = X.loc[:, source == self.macro_series].values
        X_yields = X.loc[:, source == self.yield_series].values
        return X_macro, X_yields

    def fit(self, X, y):
        X_macro, X_yields = self._select_features(X)
        y_arr = y if isinstance(y, np.ndarray) else np.array(y)
        n_out = y_arr.shape[1] if y_arr.ndim == 2 else 1

        self._fit_count += 1
        train_kwargs = dict(
            epochs=self.epochs, lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            patience=self.patience,
        )

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            def build_fn(n_inputs_list, n_out_, dropout):
                return _ExogNet(n_inputs_list[0], n_inputs_list[1], n_out_,
                                archi=self.archi, dropout=dropout)

            self._model, self._scalers, _, self._best_params = _grid_search_l1(
                build_fn, [X_macro, X_yields], y_arr, self.param_grid, n_out,
                seed=self.seed, **train_kwargs
            )
        else:
            dropout = self._best_params['Dropout'] if self._best_params else self.dropout
            l1l2 = self._best_params['l1l2'] if self._best_params else self.l1l2

            X_tr, y_tr, X_va, y_va, self._scalers = _split_and_scale(
                [X_macro, X_yields], y_arr
            )
            self._model = _ExogNet(X_macro.shape[1], X_yields.shape[1], n_out,
                                   archi=self.archi, dropout=dropout)
            _train_loop_with_l1(
                self._model, X_tr, y_tr, X_va, y_va,
                l1l2=l1l2, seed=self.seed, **train_kwargs
            )

    def predict(self, X):
        X_macro, X_yields = self._select_features(X)
        X_macro_s = self._scalers[0].transform(X_macro)
        X_yields_s = self._scalers[1].transform(X_yields)

        self._model.eval()
        with torch.no_grad():
            pred = self._model(
                torch.tensor(X_macro_s, dtype=torch.float32),
                torch.tensor(X_yields_s, dtype=torch.float32),
            ).numpy()
        return pred.squeeze()


class EnsembleExogANN:
    """
    NNEnsemExogGeneric equivalent: per-group macro sub-networks + yields.
    
    Bianchi's NN1LayerEnsemExog uses archi=[1] — each group compressed to 
    a single scalar before merging with yields.
    """

    DEFAULT_GRID = {
        'Dropout': [0.0, 0.1, 0.3, 0.5],
        'l1l2':    [1e-5, 1e-4, 1e-3],
    }

    def __init__(self, archi=(1,), macro_series='fred',
                 yield_series='yields', groups=None,
                 do_grid_search=True, param_grid=None,
                 epochs=500, lr=0.015, momentum=0.9, weight_decay=0.01,
                 batch_size=32, patience=20, tune_every=48, seed=42,
                 dropout=0.0, l1l2=1e-4):
        """
        Parameters
        ----------
        groups : np.ndarray
            Integer group labels per macro column (from groups_as_array).
            Required for splitting macro features into sub-networks.
        """
        self.archi = archi
        self.macro_series = macro_series
        self.yield_series = yield_series
        self.groups = groups
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
        self._group_slices = None  # computed on first fit

    def _select_features(self, X):
        """
        Return (list_of_group_arrays, yields_array).
        Each group array has shape (T, n_features_in_group).
        """
        source = X.columns.get_level_values(0)
        X_macro = X.loc[:, source == self.macro_series].values
        X_yields = X.loc[:, source == self.yield_series].values

        # Determine group structure from the macro columns
        if self.groups is None:
            # Infer from MultiIndex 'group' level on macro columns only
            macro_cols = X.loc[:, source == self.macro_series]
            group_labels = macro_cols.columns.get_level_values('group')
            unique_groups = list(dict.fromkeys(group_labels))
            self._group_slices = []
            for g in unique_groups:
                indices = [i for i, gl in enumerate(group_labels) if gl == g]
                self._group_slices.append(indices)
        elif self._group_slices is None:
            # Build from integer group array (filter to macro columns only)
            macro_mask = source == self.macro_series
            macro_groups = self.groups[macro_mask]
            unique_groups = np.unique(macro_groups)
            self._group_slices = []
            for g in unique_groups:
                self._group_slices.append(np.where(macro_groups == g)[0].tolist())

        grouped = [X_macro[:, idx] for idx in self._group_slices]
        return grouped, X_yields

    def fit(self, X, y):
        grouped, X_yields = self._select_features(X)
        y_arr = y if isinstance(y, np.ndarray) else np.array(y)
        n_out = y_arr.shape[1] if y_arr.ndim == 2 else 1

        # X_arrays = [group_0, group_1, ..., group_K, yields]
        X_arrays = grouped + [X_yields]
        group_sizes = [g.shape[1] for g in grouped]
        n_yields = X_yields.shape[1]

        self._fit_count += 1
        train_kwargs = dict(
            epochs=self.epochs, lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            patience=self.patience,
        )

        if self.do_grid_search and (self._fit_count - 1) % self.tune_every == 0:
            def build_fn(n_inputs_list, n_out_, dropout):
                # n_inputs_list = [g0_size, g1_size, ..., gK_size, yields_size]
                g_sizes = n_inputs_list[:-1]
                n_y = n_inputs_list[-1]
                return _EnsembleExogNet(g_sizes, n_y, n_out_,
                                        archi=self.archi, dropout=dropout)

            self._model, self._scalers, _, self._best_params = _grid_search_l1(
                build_fn, X_arrays, y_arr, self.param_grid, n_out,
                seed=self.seed, **train_kwargs
            )
        else:
            dropout = self._best_params['Dropout'] if self._best_params else self.dropout
            l1l2 = self._best_params['l1l2'] if self._best_params else self.l1l2

            X_tr, y_tr, X_va, y_va, self._scalers = _split_and_scale(
                X_arrays, y_arr
            )
            self._model = _EnsembleExogNet(
                group_sizes, n_yields, n_out,
                archi=self.archi, dropout=dropout
            )
            _train_loop_with_l1(
                self._model, X_tr, y_tr, X_va, y_va,
                l1l2=l1l2, seed=self.seed, **train_kwargs
            )

    def predict(self, X):
        grouped, X_yields = self._select_features(X)
        X_arrays = grouped + [X_yields]

        tensors = []
        for arr, scaler in zip(X_arrays, self._scalers):
            tensors.append(torch.tensor(scaler.transform(arr), dtype=torch.float32))

        self._model.eval()
        with torch.no_grad():
            pred = self._model(*tensors).numpy()
        return pred.squeeze()

    def get_group_activations(self, X):
        """
        Return the scalar output of each group sub-network.
        Useful for interpreting which groups drive predictions.
        
        Returns
        -------
        activations : np.ndarray, shape (n_samples, n_groups)
        """
        grouped, X_yields = self._select_features(X)

        scaled = []
        for arr, scaler in zip(grouped, self._scalers[:-1]):
            scaled.append(torch.tensor(scaler.transform(arr), dtype=torch.float32))

        self._model.eval()
        with torch.no_grad():
            activations = []
            for net, x_g in zip(self._model.group_nets, scaled):
                activations.append(net(x_g).numpy())

        return np.concatenate(activations, axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  Convenience aliases matching Bianchi's naming
# ═══════════════════════════════════════════════════════════════════════

def BianchiANN(hidden_size=3, series='forward', **kwargs):
    """Alias for ForwardRateANN with Bianchi's default settings."""
    return ForwardRateANN(hidden_size=hidden_size, series=series, **kwargs)


def NN3LayerExog(**kwargs):
    """Alias: 3-layer macro net (32→16→8) + yields. Matches Bianchi's NN3LayerExog."""
    return ExogANN(archi=(32, 16, 8), **kwargs)


def NN1LayerEnsemExog(**kwargs):
    """Alias: per-group 1-layer (→1 unit) ensemble + yields. Matches Bianchi's NN1LayerEnsemExog."""
    return EnsembleExogANN(archi=(1,), **kwargs)

# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from copy import deepcopy
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import ParameterGrid

# DEVICE = 'cpu'

# class BianchiANN:
#     """
#     One-hidden-layer feedforward network matching Bianchi's design choices,
#     but using forward rates as a flat input (no group ensembling).
#     """

#     def __init__(self, hidden_size=3, series='forward', epochs=500, lr=0.015,
#                  momentum=0.9, weight_decay=0.01, batch_size=32, patience=20,
#                  tune_every=48, seed=42):
#         self.hidden_size = hidden_size
#         self.series = series
#         self.epochs = epochs
#         self.lr = lr
#         self.momentum = momentum
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#         self.patience = patience
#         self.tune_every = tune_every
#         self.seed = seed

#         self.param_grid = list(ParameterGrid({
#             'dropout': [0.1, 0.3, 0.5],
#             'l1l2': [0.01, 0.001],
#         }))

#         self.scaler_ = None
#         self.model_ = None
#         self._fit_count = 0
#         self._best_params = {'dropout': 0.3, 'l1l2': 0.001}

#     def _build_net(self, n_in, n_out, dropout, l1l2):
#         model = nn.Sequential(
#             nn.Linear(n_in, self.hidden_size),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.BatchNorm1d(self.hidden_size),
#             nn.Linear(self.hidden_size, n_out),
#         ).to(DEVICE)
#         for m in model.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 nn.init.zeros_(m.bias)
#         return model

#     def _train(self, model, X_train, y_train, X_val, y_val,
#                l1l2, epochs=None):
#         if epochs is None:
#             epochs = self.epochs

#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=self.lr,
#             momentum=self.momentum,
#             weight_decay=self.weight_decay,
#             nesterov=True,
#         )
#         criterion = nn.MSELoss()

#         best_val = np.inf
#         best_state = deepcopy(model.state_dict())
#         wait = 0
#         n = X_train.shape[0]

#         for epoch in range(epochs):
#             model.train()
#             perm = torch.randperm(n, device=DEVICE)
#             for i in range(0, n, self.batch_size):
#                 idx = perm[i:i + self.batch_size]
#                 if len(idx) < 2:  # BatchNorm needs at least 2 samples
#                     continue
#                 xb, yb = X_train[idx], y_train[idx]

#                 optimizer.zero_grad()
#                 loss = criterion(model(xb), yb)

#                 if l1l2 > 0:
#                     l1 = sum(p.abs().sum() for p in model.parameters())
#                     loss = loss + l1l2 * l1

#                 loss.backward()
#                 optimizer.step()

#             model.eval()
#             with torch.no_grad():
#                 val_loss = criterion(model(X_val), y_val).item()
#             if val_loss < best_val - 1e-6:
#                 best_val = val_loss
#                 best_state = deepcopy(model.state_dict())
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= self.patience:
#                     break

#         model.load_state_dict(best_state)
#         return best_val

#     def _select_features(self, X):
#         if hasattr(X, 'columns') and isinstance(X.columns, pd.MultiIndex):
#             return X[self.series].values.astype(np.float32)
#         return np.array(X, dtype=np.float32)

#     def fit(self, X, y):
#         torch.manual_seed(self.seed)
#         np.random.seed(self.seed)

#         X_np = self._select_features(X)
#         y_np = np.array(y, dtype=np.float32)
#         if y_np.ndim == 1:
#             y_np = y_np.reshape(-1, 1)

#         self.scaler_ = MinMaxScaler(feature_range=(-1, 1))
#         X_np = self.scaler_.fit_transform(X_np)

#         n = X_np.shape[0]
#         n_train = int(n * 0.85)

#         X_train = torch.tensor(X_np[:n_train], device=DEVICE)
#         y_train = torch.tensor(y_np[:n_train], device=DEVICE)
#         X_val = torch.tensor(X_np[n_train:], device=DEVICE)
#         y_val = torch.tensor(y_np[n_train:], device=DEVICE)

#         n_in = X_np.shape[1]
#         n_out = y_np.shape[1]

#         self._fit_count += 1
#         if self._fit_count % self.tune_every == 1 or self._fit_count == 1:
#             best_score = np.inf
#             for params in self.param_grid:
#                 torch.manual_seed(self.seed)
#                 net = self._build_net(n_in, n_out, params['dropout'], params['l1l2'])
#                 val_loss = self._train(net, X_train, y_train, X_val, y_val,
#                                        l1l2=params['l1l2'])
#                 if val_loss < best_score:
#                     best_score = val_loss
#                     self._best_params = params

#         torch.manual_seed(self.seed)
#         self.model_ = self._build_net(n_in, n_out,
#                                        self._best_params['dropout'],
#                                        self._best_params['l1l2'])
#         self._train(self.model_, X_train, y_train, X_val, y_val,
#                     l1l2=self._best_params['l1l2'])

#     def predict(self, X):
#         X_np = self._select_features(X)
#         if X_np.ndim == 1:
#             X_np = X_np.reshape(1, -1)
#         X_np = self.scaler_.transform(X_np)
#         self.model_.eval()
#         with torch.no_grad():
#             pred = self.model_(torch.tensor(X_np, device=DEVICE)).cpu().numpy()
#         if pred.shape[1] == 1:
#             return pred.flatten()
#         return pred