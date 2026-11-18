"""Sliding-window sequence builders for the neural models."""

from __future__ import annotations

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd




def build_engine_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 15,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale rows and form left-padded causal windows without crossing engines."""
    if seq_length < 1:
        raise ValueError("seq_length must be positive")
    ordered = frame.sort_values(["unit_id", "cycle"]).copy()
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    values = scaler.fit_transform(ordered[feature_cols]) if fit_scaler else scaler.transform(ordered[feature_cols])
    scaled = pd.DataFrame(values, columns=feature_cols, index=ordered.index)
    scaled["unit_id"] = ordered["unit_id"].to_numpy()
    scaled["cycle"] = ordered["cycle"].to_numpy()
    sequences, labels = [], []
    target_values = ordered[target_col].astype(float)
    for _, group in scaled.groupby("unit_id", sort=False):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        targets = target_values.loc[group.index].to_numpy(dtype=np.float32)
        for t in range(len(features)):
            start = t - seq_length + 1
            if start < 0:
                pad = np.repeat(features[:1], -start, axis=0)
                seq = np.vstack([pad, features[: t + 1]])
            else:
                seq = features[start : t + 1]
            sequences.append(seq)
            labels.append(targets[t])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32), scaler


def build_causal_engine_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
    seq_length: int = 15,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """Build causal sliding windows of length `seq_length` without crossing engine boundaries.
    
    Uses causal left-padding with the first available cycle for early cycles (t < seq_length),
    ensuring that EVERY row in the input DataFrame receives an aligned prediction.
    """
    if seq_length < 1:
        raise ValueError("seq_length must be positive")
    ordered = frame.sort_values(["unit_id", "cycle"]).copy()
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    
    feature_vals = scaler.fit_transform(ordered[feature_cols]) if fit_scaler else scaler.transform(ordered[feature_cols])
    scaled = pd.DataFrame(feature_vals, columns=feature_cols, index=ordered.index)
    scaled["unit_id"] = ordered["unit_id"].to_numpy()
    scaled["cycle"] = ordered["cycle"].to_numpy()

    has_target = target_col is not None and target_col in ordered.columns
    targets_all = ordered[target_col].astype(float) if has_target else None

    sequences = []
    labels = [] if has_target else None

    for _, group in scaled.groupby("unit_id", sort=False):
        features = group[feature_cols].to_numpy(dtype=np.float32)
        n_rows = len(features)
        grp_targets = targets_all.loc[group.index].to_numpy(dtype=np.float32) if has_target else None

        for t in range(n_rows):
            start = t - seq_length + 1
            if start < 0:
                pad_len = -start
                pad = np.repeat(features[:1], pad_len, axis=0)
                sub = features[: t + 1]
                seq = np.vstack([pad, sub])
            else:
                seq = features[start : t + 1]
            sequences.append(seq)
            if has_target:
                labels.append(grp_targets[t])

    seq_arr = np.asarray(sequences, dtype=np.float32)
    lbl_arr = np.asarray(labels, dtype=np.float32) if has_target else None
    return seq_arr, lbl_arr, scaler
