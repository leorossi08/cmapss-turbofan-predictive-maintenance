"""Supplementary models (SVM + PyTorch) and a uniform fit/predict helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from pdm_cmaps.modeling.torch_models import PyTorchCNNModel, PyTorchRNNModel, PyTorchTransformerModel


def build_extra_models(seed: int) -> dict[str, tuple[str, object]]:
    """Supplementary models appended to the benchmark (SVM + PyTorch sequence models), fresh and unfitted."""
    return {
        "Support Vector Machine (SVM)": ("sklearn", make_pipeline(StandardScaler(), LinearSVR(C=1.0, random_state=seed, max_iter=5000))),
        "PyTorch CNN": ("deep", PyTorchCNNModel(seq_length=15, random_state=seed)),
        "PyTorch RNN (LSTM)": ("deep", PyTorchRNNModel(seq_length=15, random_state=seed)),
        "PyTorch Transformer": ("deep", PyTorchTransformerModel(seq_length=15, random_state=seed)),
    }

def fit_predict(m_type: str, model, train_frame: pd.DataFrame, features: list[str], eval_frame: pd.DataFrame) -> np.ndarray:
    """Fit ``model`` on ``train_frame`` and return one prediction per row of ``eval_frame`` (row order preserved)."""
    if m_type == "sklearn":
        model.fit(train_frame[features].to_numpy(dtype=float), train_frame["RUL"].to_numpy(dtype=float))
        return np.asarray(model.predict(eval_frame[features].to_numpy(dtype=float)), dtype=float)
    model.fit(train_frame, features, "RUL")
    return np.asarray(model.predict_values(eval_frame), dtype=float)
