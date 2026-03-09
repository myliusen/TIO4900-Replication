import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from torch.utils.data import TensorDataset, DataLoader
from .ann import train_model

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

class SimpleRNNNet(nn.Module):
    def __init__(self, n_features, hidden_size=8, n_layers=1, n_out=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, n_out)

    def forward(self, x_seq):
        # x_seq: [B, L, F]
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]          # [B, H]
        return self.head(last)        # [B, n_out]


class SimpleRNNANN:
    def __init__(self, series="forward", lookback=12, hidden_size=8, n_layers=1, **kwargs):
        self.series = series
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.train_params = kwargs

        self._model = None
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

    def _make_sequences(self, X2d, y2d):
        Xs, ys = [], []
        L = self.lookback
        for i in range(L - 1, len(X2d)):
            Xs.append(X2d[i - L + 1:i + 1])  # [L, F]
            ys.append(y2d[i])                # predict y at end of window
        return np.asarray(Xs), np.asarray(ys)

    def fit(self, X, y):
        X_arr = X[self.series].values
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        self._x_scaler.fit(X_arr)
        self._y_scaler.fit(y_arr)

        X_scaled = self._x_scaler.transform(X_arr)
        y_scaled = self._y_scaler.transform(y_arr)

        X_seq, y_seq = self._make_sequences(X_scaled, y_scaled)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        # simple train/val split
        n = len(X_t)
        split = max(1, int(0.85 * n))
        X_tr, y_tr = X_t[:split], y_t[:split]
        X_va, y_va = X_t[split:], y_t[split:]

        self._model = SimpleRNNNet(
            n_features=X_arr.shape[1],
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_out=y_arr.shape[1],
            dropout=self.train_params.get("dropout", 0.0)
        )

        train_model(
            self._model,
            [X_tr], y_tr,
            [X_va if len(X_va) else X_tr], y_va if len(y_va) else y_tr,
            epochs=self.train_params.get("epochs", 100),
            lr=self.train_params.get("lr", 0.01),
            momentum=self.train_params.get("momentum", 0.9),
            batch_size=self.train_params.get("batch_size", 32),
            patience=self.train_params.get("patience", 10),
            l1l2_macro=self.train_params.get("l1l2", 1e-4),
            l1l2_fwd=self.train_params.get("l1l2", 1e-4),
            seed=self.train_params.get("seed", 42),
        )

    def predict_at(self, X, t_index: int):
        # Uses full history up to t_index to build sequence
        i0 = t_index - self.lookback + 1
        if i0 < 0:
            raise ValueError("Not enough history for RNN lookback.")
        X_win = X[self.series].iloc[i0:t_index + 1].values
        X_win = self._x_scaler.transform(X_win)
        x_t = torch.tensor(X_win[None, :, :], dtype=torch.float32)  # [1, L, F]

        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(x_t).cpu().numpy()
        return self._y_scaler.inverse_transform(pred_scaled).squeeze()

    def predict(self, X):
        # fallback for compatibility (single-row predict unsupported for sequence models)
        raise NotImplementedError("Use predict_at(X_full, t_index) with expanding_window.")


class HybridRNNNet(nn.Module):
    def __init__(
        self,
        n_macro,
        n_fwd,
        n_out,
        hidden_macro=16,
        hidden_fwd=8,
        n_layers=1,
        dropout=0.0,
        use_bn=True,
    ):
        super().__init__()
        self.macro_rnn = nn.RNN(
            input_size=n_macro,
            hidden_size=hidden_macro,
            num_layers=n_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fwd_rnn = nn.RNN(
            input_size=n_fwd,
            hidden_size=hidden_fwd,
            num_layers=n_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        merge_dim = hidden_macro + hidden_fwd
        self.bn_merge = nn.BatchNorm1d(merge_dim) if use_bn else nn.Identity()
        self.output_layer = nn.Linear(merge_dim, n_out)

    def forward(self, x_macro_seq, x_fwd_seq):
        # x_*_seq: [B, L, F]
        om, _ = self.macro_rnn(x_macro_seq)
        of, _ = self.fwd_rnn(x_fwd_seq)
        hm = om[:, -1, :]  # [B, Hm]
        hf = of[:, -1, :]  # [B, Hf]
        merged = torch.cat([hm, hf], dim=-1)
        return self.output_layer(self.bn_merge(merged))


class HybridRNNANN:
    def __init__(
        self,
        lookback=12,
        hidden_macro=16,
        hidden_fwd=8,
        n_layers=1,
        use_bn=True,
        **kwargs,
    ):
        self.lookback = int(lookback)
        self.hidden_macro = int(hidden_macro)
        self.hidden_fwd = int(hidden_fwd)
        self.n_layers = int(n_layers)
        self.use_bn = bool(use_bn)
        self.train_params = kwargs

        self._model = None
        self._x_scalers = None  # [macro_scaler, fwd_scaler]
        self._y_scaler = StandardScaler()

    def _select_features(self, X):
        return [X["fred"].values, X["forward"].values]

    def _make_sequences(self, X_macro, X_fwd, y):
        L = self.lookback
        Xm, Xf, ys = [], [], []
        for t in range(L - 1, len(y)):
            Xm.append(X_macro[t - L + 1:t + 1])  # [L, Fm]
            Xf.append(X_fwd[t - L + 1:t + 1])    # [L, Ff]
            ys.append(y[t])
        return np.asarray(Xm), np.asarray(Xf), np.asarray(ys)

    def fit(self, X, y):
        X_macro, X_fwd = self._select_features(X)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        s_macro = StandardScaler().fit(X_macro)
        s_fwd = StandardScaler().fit(X_fwd)
        self._y_scaler.fit(y_arr)
        self._x_scalers = [s_macro, s_fwd]

        Xm = s_macro.transform(X_macro)
        Xf = s_fwd.transform(X_fwd)
        ys = self._y_scaler.transform(y_arr)

        Xm_seq, Xf_seq, y_seq = self._make_sequences(Xm, Xf, ys)

        Xm_t = torch.tensor(Xm_seq, dtype=torch.float32)
        Xf_t = torch.tensor(Xf_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)

        n = len(y_t)
        split = max(1, int(0.85 * n))
        Xm_tr, Xf_tr, y_tr = Xm_t[:split], Xf_t[:split], y_t[:split]
        Xm_va, Xf_va, y_va = Xm_t[split:], Xf_t[split:], y_t[split:]

        self._model = HybridRNNNet(
            n_macro=X_macro.shape[1],
            n_fwd=X_fwd.shape[1],
            n_out=y_arr.shape[1],
            hidden_macro=self.hidden_macro,
            hidden_fwd=self.hidden_fwd,
            n_layers=self.n_layers,
            dropout=self.train_params.get("dropout", 0.0),
            use_bn=self.use_bn,
        )

        train_model(
            self._model,
            [Xm_tr, Xf_tr], y_tr,
            [Xm_va if len(Xm_va) else Xm_tr, Xf_va if len(Xf_va) else Xf_tr],
            y_va if len(y_va) else y_tr,
            epochs=self.train_params.get("epochs", 100),
            lr=self.train_params.get("lr", 0.01),
            momentum=self.train_params.get("momentum", 0.9),
            batch_size=self.train_params.get("batch_size", 32),
            patience=self.train_params.get("patience", 10),
            l1l2_macro=self.train_params.get("l1l2_macro", 1e-4),
            l1l2_fwd=self.train_params.get("l1l2_fwd", 1e-4),
            seed=self.train_params.get("seed", 42),
        )

    def predict_at(self, X, t_index: int):
        i0 = t_index - self.lookback + 1
        if i0 < 0:
            raise ValueError("Not enough history for HybridRNN lookback.")

        X_macro, X_fwd = self._select_features(X)
        wm = self._x_scalers[0].transform(X_macro[i0:t_index + 1])
        wf = self._x_scalers[1].transform(X_fwd[i0:t_index + 1])

        xm = torch.tensor(wm[None, :, :], dtype=torch.float32)  # [1,L,Fm]
        xf = torch.tensor(wf[None, :, :], dtype=torch.float32)  # [1,L,Ff]

        self._model.eval()
        with torch.no_grad():
            pred_scaled = self._model(xm, xf).cpu().numpy()
        return self._y_scaler.inverse_transform(pred_scaled).squeeze()

    def predict(self, X):
        raise NotImplementedError("Use predict_at(X_full, t_index) with expanding_window.")


class GroupEnsembleRNNNet(nn.Module):
    def __init__(self, group_dims, fwd_dim, n_out, h_macro=8, h_fwd=8, n_layers=1, dropout=0.0):
        super().__init__()
        self.macro_rnns = nn.ModuleList([
            nn.RNN(
                input_size=d, hidden_size=h_macro, num_layers=n_layers,
                nonlinearity="tanh", batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0
            ) for d in group_dims
        ])
        self.fwd_rnn = nn.RNN(
            input_size=fwd_dim, hidden_size=h_fwd, num_layers=n_layers,
            nonlinearity="tanh", batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        merge_dim = len(group_dims) * h_macro + h_fwd
        self.output_layer = nn.Linear(merge_dim, n_out)

    def forward(self, *inputs):
        macro_inputs = inputs[:-1]  # each: [B,L,Fg]
        fwd_input = inputs[-1]      # [B,L,Ff]

        hs = []
        for rnn, x in zip(self.macro_rnns, macro_inputs):
            o, _ = rnn(x)
            hs.append(o[:, -1, :])  # [B,h_macro]

        o_fwd, _ = self.fwd_rnn(fwd_input)
        hs.append(o_fwd[:, -1, :])  # [B,h_fwd]

        merged = torch.cat(hs, dim=-1)
        return self.output_layer(merged)


class GroupEnsembleRNNANN:
    def __init__(self, lookback=12, h_macro=8, h_fwd=8, n_layers=1, **kwargs):
        self.lookback = int(lookback)
        self.h_macro = int(h_macro)
        self.h_fwd = int(h_fwd)
        self.n_layers = int(n_layers)
        self.train_params = kwargs

        self._group_names = None
        self._model = None
        self._scalers = None
        self._y_scaler = StandardScaler()

    def _select_blocks(self, X):
        if self._group_names is None:
            self._group_names = X["fred"].columns.get_level_values(0).unique().tolist()
        macro_blocks = [X["fred"][g].values for g in self._group_names]
        fwd_block = X["forward"].values
        return macro_blocks + [fwd_block]

    def _make_sequences(self, X_blocks_scaled, y_scaled):
        L = self.lookback
        n = len(y_scaled)
        X_seq_blocks = [[] for _ in X_blocks_scaled]
        y_seq = []
        for t in range(L - 1, n):
            for i, xb in enumerate(X_blocks_scaled):
                X_seq_blocks[i].append(xb[t - L + 1:t + 1])  # [L,F]
            y_seq.append(y_scaled[t])
        X_seq_blocks = [np.asarray(v) for v in X_seq_blocks]  # each [N,L,F]
        y_seq = np.asarray(y_seq)                              # [N,out]
        return X_seq_blocks, y_seq

    def fit(self, X, y):
        blocks = self._select_blocks(X)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        self._scalers = []
        blocks_scaled = []
        for b in blocks:
            s = StandardScaler().fit(b)
            self._scalers.append(s)
            blocks_scaled.append(s.transform(b))

        self._y_scaler.fit(y_arr)
        y_scaled = self._y_scaler.transform(y_arr)

        X_seq_blocks, y_seq = self._make_sequences(blocks_scaled, y_scaled)

        n = len(y_seq)
        split = max(1, int(0.85 * n))
        X_tr = [torch.tensor(v[:split], dtype=torch.float32) for v in X_seq_blocks]
        X_va = [torch.tensor(v[split:], dtype=torch.float32) for v in X_seq_blocks]
        y_tr = torch.tensor(y_seq[:split], dtype=torch.float32)
        y_va = torch.tensor(y_seq[split:], dtype=torch.float32)

        dims = [b.shape[1] for b in blocks]
        self._model = GroupEnsembleRNNNet(
            group_dims=dims[:-1], fwd_dim=dims[-1], n_out=y_arr.shape[1],
            h_macro=self.h_macro, h_fwd=self.h_fwd, n_layers=self.n_layers,
            dropout=self.train_params.get("dropout", 0.0),
        )

        train_model(
            self._model,
            X_tr, y_tr,
            [v if len(v) else x for v, x in zip(X_va, X_tr)],
            y_va if len(y_va) else y_tr,
            epochs=self.train_params.get("epochs", 100),
            lr=self.train_params.get("lr", 0.01),
            momentum=self.train_params.get("momentum", 0.9),
            batch_size=self.train_params.get("batch_size", 32),
            patience=self.train_params.get("patience", 10),
            l1l2_macro=self.train_params.get("l1l2_macro", 1e-4),
            l1l2_fwd=self.train_params.get("l1l2_fwd", 1e-4),
            seed=self.train_params.get("seed", 42),
        )

    def predict_at(self, X, t_index):
        i0 = t_index - self.lookback + 1
        if i0 < 0:
            raise ValueError("Not enough history for lookback.")
        raw_blocks = self._select_blocks(X)

        seq_tensors = []
        for s, b in zip(self._scalers, raw_blocks):
            win = b[i0:t_index + 1]                 # [L,F]
            win_s = s.transform(win)
            seq_tensors.append(torch.tensor(win_s[None, :, :], dtype=torch.float32))

        self._model.eval()
        with torch.no_grad():
            pred_s = self._model(*seq_tensors).cpu().numpy()
        return self._y_scaler.inverse_transform(pred_s).squeeze()
