"""Estimator wrappers (fit/predict) around the PyTorch networks."""

from __future__ import annotations

import time
from typing import Literal
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit

from pdm_cmaps.modeling.networks import CNNRegressorNet, LSTMRegressorNet, RNNRegressorNet, TransformerRegressorNet
from pdm_cmaps.modeling.sequences import build_causal_engine_sequences, build_engine_sequences


class PyTorchLSTMModel:
    """Scikit-learn-style wrapper that retains fitted scalers and the trained network."""

    def __init__(
        self,
        seq_length: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.002,
        batch_size: int = 128,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device(device)
        self.net = None
        self.scaler = None
        self.target_scaler = None
        self.feature_cols: list[str] = []
        self.target_col = ""
        self.fit_time = 0.0

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> "PyTorchLSTMModel":
        """Train on engine-local windows; early stopping monitors training loss."""
        torch.manual_seed(self.random_state)
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        started = time.perf_counter()
        x, y, self.scaler = build_engine_sequences(
            frame, self.feature_cols, target_col, self.seq_length, fit_scaler=True,
        )
        self.target_scaler = StandardScaler().fit(y.reshape(-1, 1))
        scaled_y = self.target_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(scaled_y)), batch_size=self.batch_size, shuffle=True)
        self.net = LSTMRegressorNet(len(self.feature_cols), self.hidden_dim, self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best_loss, best_state, waiting = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.net.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.net(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch_y)
            epoch_loss = total / len(loader.dataset)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
                waiting = 0
            else:
                waiting += 1
                if waiting >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_time = time.perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned target and sequence-endpoint predictions."""
        if self.net is None or self.scaler is None or self.target_scaler is None:
            raise RuntimeError("Fit the LSTM before calling predict")
        x, y, _ = build_engine_sequences(
            frame, self.feature_cols, self.target_col, self.seq_length,
            scaler=self.scaler, fit_scaler=False,
        )
        self.net.eval()
        with torch.no_grad():
            scaled = self.net(torch.from_numpy(x).to(self.device)).cpu().numpy()
        prediction = self.target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()
        return y, prediction


def get_safe_device() -> torch.device:
    """Return CUDA device only if hardware compute capability is supported by PyTorch, else CPU."""
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                _ = torch.zeros(1, device="cuda")
                return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")

class BasePyTorchSequenceModel:
    """Base class providing sequence batching, training loop, early stopping, and device management."""

    def __init__(
        self,
        seq_length: int = 15,
        learning_rate: float = 0.002,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        if device is None:
            self.device = get_safe_device()
        else:
            self.device = torch.device(device)

        self.net: nn.Module | None = None
        self.scaler: StandardScaler | None = None
        self.target_scaler: StandardScaler | None = None
        self.feature_cols: list[str] = []
        self.fit_time: float = 0.0

    def _build_network(self, input_dim: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_network")

    def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str = "RUL") -> "BasePyTorchSequenceModel":
        """Fit model with engine-preserving causal sequences, deterministic generator, and grouped early stopping."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.feature_cols = list(feature_cols)

        started = time.perf_counter()

        # Engine-grouped internal validation partition (15%) for uncorrupted early stopping
        n_engines = frame["unit_id"].nunique()
        has_val = n_engines >= 4
        if has_val:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
            tr_idx, val_idx = next(gss.split(frame, groups=frame["unit_id"]))
            tr_frame = frame.iloc[tr_idx].copy().reset_index(drop=True)
            val_frame = frame.iloc[val_idx].copy().reset_index(drop=True)
        else:
            tr_frame = frame
            val_frame = None

        x_tr, y_tr, self.scaler = build_causal_engine_sequences(
            tr_frame, self.feature_cols, target_col, self.seq_length, fit_scaler=True,
        )
        self.target_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
        scaled_y_tr = self.target_scaler.transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()

        gen = torch.Generator().manual_seed(self.random_state)
        dataset_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(scaled_y_tr))
        loader_tr = DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True, generator=gen)

        loader_val = None
        if val_frame is not None:
            x_val, y_val, _ = build_causal_engine_sequences(
                val_frame, self.feature_cols, target_col, self.seq_length, scaler=self.scaler, fit_scaler=False,
            )
            scaled_y_val = self.target_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32).ravel()
            loader_val = DataLoader(
                TensorDataset(torch.from_numpy(x_val), torch.from_numpy(scaled_y_val)),
                batch_size=self.batch_size * 2,
                shuffle=False,
            )

        self.net = self._build_network(len(self.feature_cols)).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        waiting = 0

        for _ in range(self.max_epochs):
            self.net.train()
            total_tr_loss = 0.0
            for batch_x, batch_y in loader_tr:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=2.0)
                optimizer.step()
                total_tr_loss += float(loss.item()) * len(batch_y)

            # Evaluate early stopping on internal engine validation split
            if loader_val is not None:
                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in loader_val:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        val_loss += float(loss_fn(self.net(batch_x), batch_y).item()) * len(batch_y)
                epoch_loss = val_loss / len(loader_val.dataset)
            else:
                epoch_loss = total_tr_loss / len(loader_tr.dataset)

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                waiting = 0
            else:
                waiting += 1
                if waiting >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.fit_time = time.perf_counter() - started
        return self

    def predict_values(self, frame: pd.DataFrame) -> np.ndarray:
        """Alias for predict returning raw array."""
        return self.predict(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict aligned target values for every row in the DataFrame."""
        if self.net is None or self.scaler is None or self.target_scaler is None:
            raise RuntimeError("Model must be fitted before predict")

        x, _, _ = build_causal_engine_sequences(
            frame, self.feature_cols, None, self.seq_length, scaler=self.scaler, fit_scaler=False,
        )
        self.net.eval()
        preds = []
        loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=self.batch_size * 2, shuffle=False)
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                scaled_pred = self.net(batch_x).cpu().numpy()
                preds.append(scaled_pred)

        concatenated = np.concatenate(preds)
        unscaled = self.target_scaler.inverse_transform(concatenated.reshape(-1, 1)).ravel()
        return unscaled

class PyTorchRNNModel(BasePyTorchSequenceModel):
    """Modular Recurrent Neural Network (LSTM or GRU) Regressor."""

    def __init__(
        self,
        seq_length: int = 15,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: Literal["lstm", "gru"] = "lstm",
        learning_rate: float = 0.002,
        batch_size: int = 256,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.cell_type = cell_type

    def _build_network(self, input_dim: int) -> nn.Module:
        return RNNRegressorNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            cell_type=self.cell_type,
        )

class PyTorchCNNModel(BasePyTorchSequenceModel):
    """1D Temporal Convolutional Neural Network Regressor."""

    def __init__(
        self,
        seq_length: int = 15,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.002,
        batch_size: int = 256,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return CNNRegressorNet(
            input_dim=input_dim,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

class PyTorchTransformerModel(BasePyTorchSequenceModel):
    """Time Series Transformer Regressor with Multi-Head Attention."""

    def __init__(
        self,
        seq_length: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 25,
        patience: int = 4,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(
            seq_length=seq_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
            device=device,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def _build_network(self, input_dim: int) -> nn.Module:
        return TransformerRegressorNet(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
