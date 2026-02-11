import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import ParameterGrid


# ---------------------------------------------------------------------------
# 1. Network architectures (nn.Module subclasses)
# ---------------------------------------------------------------------------

class VanillaNet(nn.Module):
    """
    Simple feedforward: Input → [Hidden → ReLU → Dropout] x L → Output
    
    Parameters
    ----------
    input_size : int
    output_size : int
    hidden_sizes : list[int]   e.g. [32, 16, 8] for 3 hidden layers
    dropout : float
    l1 : float                 L1 penalty weight (applied manually in training loop)
    l2 : float                 L2 penalty weight (applied via weight_decay or manually)
    """
    def __init__(self, input_size, output_size, hidden_sizes,
                 dropout=0.0, l1=0.0, l2=0.0):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def forward_dict(self, X_dict):
        return self.forward(X_dict['main'])

    def regularization_loss(self):
        """Return combined L1 + L2 penalty on all weight matrices."""
        l1_loss = torch.tensor(0.0)
        l2_loss = torch.tensor(0.0)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_loss = l1_loss + param.abs().sum()
                l2_loss = l2_loss + (param ** 2).sum()
        return self.l1 * l1_loss + self.l2 * l2_loss


class EnsembleGroupNet(nn.Module):
    """
    Grouped ensemble: each macro group gets its own sub-network,
    outputs are concatenated, merged with exogenous (yield) input,
    passed through BatchNorm → Output.

    This mirrors NNEnsemExogGeneric from Bianchi et al.

    Parameters
    ----------
    group_input_sizes : list[int]   number of features per macro group
    exog_size : int                 number of yield/forward-rate features
    output_size : int
    group_hidden_sizes : list[int]  hidden layer sizes within each group sub-net
                                    e.g. [1] for NN1LayerEnsemExog
    dropout : float
    l1 : float
    l2 : float
    """
    def __init__(self, group_input_sizes, exog_size, output_size,
                 group_hidden_sizes, dropout=0.0, l1=0.0, l2=0.0):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.n_groups = len(group_input_sizes)

        # Build one sub-network per macro group
        self.group_nets = nn.ModuleList()
        for g_size in group_input_sizes:
            layers = []
            prev = g_size
            layers.append(nn.Dropout(dropout))
            for h in group_hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            self.group_nets.append(nn.Sequential(*layers))

        # Merge path: concat(group_outputs, exog) → Dropout → BatchNorm → Linear
        merge_input_size = len(group_input_sizes) * group_hidden_sizes[-1] + exog_size
        self.merge_dropout = nn.Dropout(dropout)
        self.merge_bn = nn.BatchNorm1d(merge_input_size)
        self.output_layer = nn.Linear(merge_input_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, group_inputs, exog_input):
        """
        Parameters
        ----------
        group_inputs : list[Tensor]   one tensor per macro group
        exog_input : Tensor           yield/forward variables
        """
        group_outs = [net(g) for net, g in zip(self.group_nets, group_inputs)]
        merged = torch.cat(group_outs + [exog_input], dim=1)
        merged = self.merge_dropout(merged)
        merged = self.merge_bn(merged)
        return self.output_layer(merged)

    def forward_dict(self, X_dict):
        group_keys = sorted(k for k in X_dict if k.startswith('group_'))
        group_inputs = [X_dict[k] for k in group_keys]
        return self.forward(group_inputs, X_dict['exog'])

    def regularization_loss(self):
        l1_loss = torch.tensor(0.0)
        l2_loss = torch.tensor(0.0)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_loss = l1_loss + param.abs().sum()
                l2_loss = l2_loss + (param ** 2).sum()
        return self.l1 * l1_loss + self.l2 * l2_loss


class ExogMergeNet(nn.Module):
    """
    Vanilla net for macro variables, merged with exogenous yields at the
    final hidden layer. Mirrors NNExogGeneric.

    Parameters
    ----------
    macro_size : int
    exog_size : int
    output_size : int
    hidden_sizes : list[int]    e.g. [32, 16, 8]
    dropout : float
    l1 : float
    l2 : float
    """
    def __init__(self, macro_size, exog_size, output_size,
                 hidden_sizes, dropout=0.0, l1=0.0, l2=0.0):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

        # Macro sub-net: Input → [Dropout → Dense → ReLU] × L
        macro_layers = []
        prev = macro_size
        for h in hidden_sizes:
            macro_layers.append(nn.Dropout(dropout))
            macro_layers.append(nn.Linear(prev, h))
            macro_layers.append(nn.ReLU())
            prev = h
        self.macro_net = nn.Sequential(*macro_layers)

        # Merge: concat(macro_hidden, exog) → Dropout → BatchNorm → Linear
        merge_input_size = hidden_sizes[-1] + exog_size
        self.merge_dropout = nn.Dropout(dropout)
        self.merge_bn = nn.BatchNorm1d(merge_input_size)
        self.output_layer = nn.Linear(merge_input_size, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, macro_input, exog_input):
        macro_hidden = self.macro_net(macro_input)
        merged = torch.cat([macro_hidden, exog_input], dim=1)
        merged = self.merge_dropout(merged)
        merged = self.merge_bn(merged)
        return self.output_layer(merged)

    def forward_dict(self, X_dict):
        return self.forward(X_dict['macro'], X_dict['exog'])

    def regularization_loss(self):
        l1_loss = torch.tensor(0.0)
        l2_loss = torch.tensor(0.0)
        for name, param in self.named_parameters():
            if 'weight' in name:
                l1_loss = l1_loss + param.abs().sum()
                l2_loss = l2_loss + (param ** 2).sum()
        return self.l1 * l1_loss + self.l2 * l2_loss


# ---------------------------------------------------------------------------
# 2. Generic training engine
# ---------------------------------------------------------------------------

class NNTrainer:
    """
    Handles the training loop, early stopping, validation split,
    scaling, and single-step or batch prediction. Architecture-agnostic.

    Parameters
    ----------
    epochs : int
    lr : float
    momentum : float
    weight_decay : float         SGD weight decay (separate from manual L1/L2)
    batch_size : int
    patience : int               early stopping patience
    val_fraction : float         fraction of training data for validation
    scaler_type : str            'minmax' or 'standard'
    optimizer_type : str         'sgd' or 'adam'
    seed : int
    """
    def __init__(self, epochs=500, lr=0.015, momentum=0.9, lr_decay=0.01,
                 batch_size=32, patience=20, val_fraction=0.15,
                 scaler_type='minmax', optimizer_type='sgd', seed=42):
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.scaler_type = scaler_type
        self.optimizer_type = optimizer_type
        self.seed = seed
        # Populated during fit
        self.scalers_ = {}
        self._best_val_loss = np.inf

    def _build_optimizer(self, model):
        """Construct optimizer from stored config."""
        if self.optimizer_type == 'sgd':
            return torch.optim.SGD(
                model.parameters(), lr=self.lr,
                momentum=self.momentum, nesterov=True,
            )
        else:
            return torch.optim.Adam(model.parameters(), lr=self.lr)

    def _make_scaler(self):
        """Return a fresh scaler instance based on scaler_type."""
        if self.scaler_type == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))
        else:
            return StandardScaler()

    def _split_validation(self, X_dict, y):
        """Split last val_fraction off. Return (X_train, y_train, X_val, y_val)."""
        T = y.shape[0]
        n_val = int(T * self.val_fraction)
        n_train = T - n_val

        X_train = {k: v[:n_train] for k, v in X_dict.items()}
        X_val = {k: v[n_train:] for k, v in X_dict.items()}
        return X_train, y[:n_train], X_val, y[n_train:]

    def fit(self, model, X_dict, y):
        """
        Train the model with early stopping.

        Parameters
        ----------
        model : nn.Module
        X_dict : dict[str, np.ndarray]
            Keys depend on architecture:
            - VanillaNet: {'main': array}
            - ExogMergeNet: {'macro': array, 'exog': array}
            - EnsembleGroupNet: {'group_0': array, ..., 'group_N': array, 'exog': array}
        y : np.ndarray  (T, n_outputs)

        Stores fitted scalers and best model state_dict internally.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- 1. Scale each input array with its own scaler ---
        self.scalers_ = {}
        X_scaled = {}
        for key, arr in X_dict.items():
            scaler = self._make_scaler()
            X_scaled[key] = scaler.fit_transform(arr).astype(np.float32)
            self.scalers_[key] = scaler

        # --- 2. Train / val split ---
        use_val = (self.val_fraction > 0 and self.patience is not None)
        if use_val:
            X_train, y_train, X_val, y_val = self._split_validation(X_scaled, y)
        else:
            X_train, y_train = X_scaled, y
            X_val, y_val = None, None

        # Convert to tensors
        X_train_t = {k: torch.tensor(v) for k, v in X_train.items()}
        y_train_t = torch.tensor(y_train.astype(np.float32))
        if use_val:
            X_val_t = {k: torch.tensor(v) for k, v in X_val.items()}
            y_val_t = torch.tensor(y_val.astype(np.float32))

        n_train = y_train_t.shape[0]
        criterion = nn.MSELoss()
        optimizer = self._build_optimizer(model)

        # --- 3. Training loop ---
        best_state = deepcopy(model.state_dict())
        self._best_val_loss = np.inf
        wait = 0
        global_step = 0

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n_train)

            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                batch_X = {k: v[idx] for k, v in X_train_t.items()}
                batch_y = y_train_t[idx]

                optimizer.zero_grad()
                pred = model.forward_dict(batch_X)
                loss = criterion(pred, batch_y) + model.regularization_loss()
                loss.backward()
                optimizer.step()

                # Keras-style LR decay: lr_t = lr / (1 + decay * step)
                global_step += 1
                if self.lr_decay > 0:
                    new_lr = self.lr / (1 + self.lr_decay * global_step)
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr

            # --- 4. Validation & early stopping ---
            if use_val:
                model.eval()
                with torch.no_grad():
                    val_pred = model.forward_dict(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()

                if val_loss < self._best_val_loss - 1e-6:
                    self._best_val_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        # --- 5. Restore best weights ---
        if use_val:
            model.load_state_dict(best_state)

    def predict(self, model, X_dict):
        """
        Generate predictions using stored scalers.

        Parameters
        ----------
        model : nn.Module
        X_dict : dict[str, np.ndarray]

        Returns
        -------
        np.ndarray  (T, n_outputs)
        """
        X_scaled = {}
        for key, arr in X_dict.items():
            X_scaled[key] = self.scalers_[key].transform(arr).astype(np.float32)

        X_t = {k: torch.tensor(v) for k, v in X_scaled.items()}

        model.eval()
        with torch.no_grad():
            pred = model.forward_dict(X_t)
        return pred.numpy()

    @property
    def best_val_loss(self):
        return self._best_val_loss


# ---------------------------------------------------------------------------
# 3. Grid search wrapper
# ---------------------------------------------------------------------------

def grid_search(model_factory, trainer_factory, X_dict, y, param_grid):
    """
    Exhaustive search over param_grid. Selects the configuration with the
    lowest validation loss, then refits on the full training data.

    Parameters
    ----------
    model_factory : callable(params) -> nn.Module
        Function that constructs a fresh model given a param dict.
        Example:
            lambda p: VanillaNet(input_size, output_size, [3],
                                 dropout=p['dropout'], l1=p['l1l2'], l2=p['l1l2'])
    trainer_factory : callable(params) -> NNTrainer
        Function that constructs a fresh trainer given a param dict.
        Example:
            lambda p: NNTrainer(lr=p.get('lr', 0.015), patience=20)
    X_dict : dict[str, np.ndarray]
        Input arrays keyed by name (e.g. {'macro': ..., 'exog': ...}).
    y : np.ndarray
        Target array, shape (T,) or (T, n_outputs).
    param_grid : dict
        Hyperparameter search space.
        E.g. {'dropout': [0.1, 0.3, 0.5], 'l1l2': [1e-5, 1e-4, 1e-3]}

    Returns
    -------
    best_model : nn.Module
        The model refitted on the full data using the best hyperparameters.
    best_trainer : NNTrainer
        The trainer used for the final refit.
    best_params : dict
        The hyperparameter combination that achieved the lowest val loss.
    best_val_loss : float
        The validation loss of the best configuration.
    """
    grid = list(ParameterGrid(param_grid))

    if len(grid) == 0:
        raise ValueError("param_grid produced an empty grid.")

    best_val_loss = np.inf
    best_params = None

    print(f"Grid search: {len(grid)} configurations")

    for i, params in enumerate(grid):
        # Build fresh model and trainer for this configuration
        model = model_factory(params)
        trainer = trainer_factory(params)

        # Train with validation split (trainer.fit uses val_fraction internally)
        try:
            trainer.fit(model, X_dict, y)
            val_loss = trainer.best_val_loss
        except Exception as e:
            print(f"  Config {i+1}/{len(grid)} {params} failed: {e}")
            continue

        print(f"  Config {i+1}/{len(grid)} {params} -> val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

    if best_params is None:
        raise RuntimeError("All grid search configurations failed.")

    print(f"Best config: {best_params} (val_loss={best_val_loss:.6f})")
    print("Refitting on full data...")

    # Refit: build a fresh model + trainer with best params,
    # but set val_fraction=0 so all data is used for training.
    best_model = model_factory(best_params)
    refit_kwargs = {k: v for k, v in best_params.items()
                    if k in ('lr', 'momentum', 'weight_decay', 'batch_size',
                             'epochs', 'scaler_type', 'optimizer_type', 'seed')}
    best_trainer = trainer_factory(best_params)
    best_trainer.val_fraction = 0.0   # no validation split on refit
    best_trainer.patience = None      # no early stopping on refit

    best_trainer.fit(best_model, X_dict, y)

    return best_model, best_trainer, best_params, best_val_loss

# ---------------------------------------------------------------------------
# 4. High-level model classes (match existing API for window_utils)
# ---------------------------------------------------------------------------

class BaseNNModel:
    """
    Base class ensuring all NN models expose .fit(X, y) and .predict(X)
    compatible with window_utils.expanding_window.

    Subclasses must implement:
        _extract_inputs(X) -> dict[str, np.ndarray]
        _build_model(input_shapes, output_size) -> nn.Module
    """
    def __init__(self, trainer_kwargs=None, do_grid_search=False,
                 param_grid=None):
        self.trainer_kwargs = trainer_kwargs or {}
        self.do_grid_search = do_grid_search
        self.param_grid = param_grid
        self.model_ = None
        self.trainer_ = None

    def _extract_inputs(self, X):
        """
        Convert the multi-index DataFrame X into a dict of numpy arrays
        for the trainer. Subclasses override this.
        """
        raise NotImplementedError

    def _build_model(self, input_shapes, output_size, params=None):
        """
        Construct a fresh nn.Module. Subclasses override this.

        Parameters
        ----------
        input_shapes : dict[str, int]
        output_size : int
        params : dict
            Hyperparameters from grid search (dropout, l1l2, etc.)
        """
        raise NotImplementedError

    def fit(self, X, y):
        X_dict = self._extract_inputs(X)
        output_size = y.shape[1] if y.ndim == 2 else 1
        y_arr = y.astype(np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        input_shapes = {k: v.shape[1] for k, v in X_dict.items()}

        if self.do_grid_search and self.param_grid:
            def model_factory(p):
                return self._build_model(input_shapes, output_size, p)

            def trainer_factory(p):
                # Merge base trainer kwargs with grid-searched params
                kw = {**self.trainer_kwargs}
                # Override trainer-level params if present in grid
                for key in ('lr', 'momentum', 'batch_size', 'epochs'):
                    if key in p:
                        kw[key] = p[key]
                return NNTrainer(**kw)

            self.model_, self.trainer_, self.best_params_, _ = grid_search(
                model_factory, trainer_factory, X_dict, y_arr, self.param_grid
            )
        else:
            self.model_ = self._build_model(input_shapes, output_size, {})
            self.trainer_ = NNTrainer(**self.trainer_kwargs)
            self.trainer_.fit(self.model_, X_dict, y_arr)

    def predict(self, X):
        X_dict = self._extract_inputs(X)
        pred = self.trainer_.predict(self.model_, X_dict)
        if pred.shape[1] == 1:
            return pred.flatten()
        return pred


class ForwardRateANN(BaseNNModel):
    """
    Simple forward-rate-only network. Mirrors NN1LayerForward in Bianchi et al.
    """
    def __init__(self, hidden_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes or [3]

    def _extract_inputs(self, X):
        return {'main': X['forward'].values.astype(np.float32)}

    def _build_model(self, input_shapes, output_size, params=None):
        p = params or {}
        return VanillaNet(
            input_shapes['main'], output_size, self.hidden_sizes,
            dropout=p.get('dropout', 0.0),
            l1=p.get('l1l2', 0.0), l2=p.get('l1l2', 0.0),
        )


class ExogMergeANN(BaseNNModel):
    """
    Macro variables through hidden layers, merged with yields at end.
    Mirrors NN3LayerExog.
    """
    def __init__(self, hidden_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes or [32, 16, 8]

    def _extract_inputs(self, X):
        return {
            'macro': X['fred'].values.astype(np.float32),
            'exog': X['forward'].values.astype(np.float32),
        }

    def _build_model(self, input_shapes, output_size, params=None):
        p = params or {}
        return ExogMergeNet(
            input_shapes['macro'], input_shapes['exog'], output_size,
            self.hidden_sizes,
            dropout=p.get('dropout', 0.0),
            l1=p.get('l1l2', 0.0), l2=p.get('l1l2', 0.0),
        )


class EnsembleGroupANN(BaseNNModel):
    """
    Grouped macro ensemble + yields. Mirrors NN1LayerEnsemExog.

    Parameters
    ----------
    group_assignments : np.ndarray
        Integer array of length n_macro_features mapping each feature to a group.
    group_hidden_sizes : list[int]
        Hidden sizes within each group sub-net. Default [1] per Bianchi.
    """
    def __init__(self, group_assignments=None, group_hidden_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.group_assignments = group_assignments
        self.group_hidden_sizes = group_hidden_sizes or [1]

    def _extract_inputs(self, X):
        macro = X['fred'].values.astype(np.float32)
        exog = X['forward'].values.astype(np.float32)

        X_dict = {}
        for g in np.unique(self.group_assignments):
            mask = self.group_assignments == g
            X_dict[f'group_{g}'] = macro[:, mask]
        X_dict['exog'] = exog
        return X_dict

    def _build_model(self, input_shapes, output_size, params=None):
        p = params or {}
        group_sizes = [input_shapes[k] for k in sorted(input_shapes)
                       if k.startswith('group_')]
        return EnsembleGroupNet(
            group_sizes, input_shapes['exog'], output_size,
            self.group_hidden_sizes,
            dropout=p.get('dropout', 0.0),
            l1=p.get('l1l2', 0.0), l2=p.get('l1l2', 0.0),
        )