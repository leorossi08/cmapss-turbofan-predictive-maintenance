"""PyTorch network architectures (nn.Module)."""

from __future__ import annotations

import math
from typing import Literal
import torch
from torch import nn



class LSTMRegressorNet(nn.Module):
    """LSTM encoder followed by a scalar regression head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(output[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class RNNRegressorNet(nn.Module):
    """Recurrent Neural Network (LSTM or GRU) with regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: Literal["lstm", "gru"] = "lstm",
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        # Take last time step
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)

class CNNRegressorNet(nn.Module):
    """1D Temporal Convolutional Neural Network with Adaptive Pooling & MLP head."""

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(num_filters * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch Conv1D expects (Batch, Channels, SeqLen)
        x_trans = x.transpose(1, 2)
        feats = self.conv_block(x_trans)
        pooled = self.pool(feats).squeeze(-1)
        return self.head(pooled).squeeze(-1)

class TransformerRegressorNet(nn.Module):
    """Time Series Transformer Encoder with Multi-Head Self-Attention and regression head."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, SeqLen, Features)
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        out = self.transformer_encoder(encoded)
        # Global average pooling over sequence length
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)
