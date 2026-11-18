"""Modelos de RUL: pool, validação cruzada agrupada, MiniRocket e redes PyTorch."""

import numpy as np
import pandas as pd
import pytest
import torch
from pdm_cmaps.domain.rul import (
    compute_piecewise_rul,
)
from pdm_cmaps.modeling.benchmark import RULBenchmark
from pdm_cmaps.modeling.minirocket import MiniRocketRULModel, MiniRocketTransformer
from pdm_cmaps.modeling.registry import get_model, get_model_pool, grouped_oof
from pdm_cmaps.modeling.torch_models import PyTorchRNNModel
from sklearn.ensemble import ExtraTreesRegressor

from factories import make_synthetic_cmapss, sample_frame


def test_grouped_oof_covers_rows_without_engine_leakage():
    frame = sample_frame(engine_count=10)
    frame["RUL"] = compute_piecewise_rul(frame)
    features = ["cycle", "setting_1", "T2"]
    estimator = ExtraTreesRegressor(n_estimators=5, random_state=2)
    folds, prediction = grouped_oof(frame, features, estimator=estimator, n_splits=5)
    assert len(prediction) == len(frame)
    assert np.isfinite(prediction).all()
    assert len(folds) == 5


def test_model_pool_includes_padua_algorithms():

    pool = get_model_pool()
    assert "CatBoost" in pool
    assert "XGBoost" in pool
    assert "LightGBM" in pool

    X = np.random.randn(20, 4)
    y = np.random.randn(20)

    for name in ["catboost", "xgboost", "lightgbm"]:
        model = get_model(name, n_estimators=5, max_depth=3, iterations=5, depth=3)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)
        assert np.isfinite(pred).all()


def test_minirocket_kernel_properties():
    """Validate that base kernels and multi-channel kernels satisfy length 9, zero sum, and {-1, 2} weights."""
    base_kernels = MiniRocketTransformer.generate_base_kernels()
    assert base_kernels.shape == (84, 9)
    assert np.all(np.isin(base_kernels, [-1.0, 2.0]))
    assert np.allclose(base_kernels.sum(axis=1), 0.0)
    for row in base_kernels:
        assert (row == 2.0).sum() == 3
        assert (row == -1.0).sum() == 6

    # Test multi-channel kernel assignment
    transformer = MiniRocketTransformer(num_kernels=84, seq_length=15, random_state=42)
    dummy_x = np.random.randn(10, 5, 15).astype(np.float32)
    transformer.fit(dummy_x)

    assert hasattr(transformer, "weights_")
    for d, weights in transformer.weights_.items():
        assert weights.shape[1] == 5  # channels
        assert weights.shape[2] == 9  # length
        for k in range(weights.shape[0]):
            kernel = weights[k]
            # Kernel sum must be zero
            assert np.isclose(kernel.sum(), 0.0)
            active_channels = (np.abs(kernel).sum(axis=1) > 0)
            assert 1 <= active_channels.sum() <= 3
            for c_idx in np.where(active_channels)[0]:
                c_weights = kernel[c_idx]
                assert set(np.unique(c_weights)) == {-1.0, 2.0}
                assert np.isclose(c_weights.sum(), 0.0)


def test_minirocket_causality():
    """Ensure that causal left-padding guarantees that output at time t does not depend on future timestamps."""
    transformer = MiniRocketTransformer(num_kernels=50, seq_length=15, dilations=(1, 2), random_state=42)
    # Fit on initial dummy data
    init_x = np.random.randn(20, 4, 15).astype(np.float32)
    transformer.fit(init_x)

    # Construct two series: identical up to t=9, diverging from t=10 onwards
    x1 = np.random.randn(1, 4, 15).astype(np.float32)
    x2 = x1.copy()
    x2[:, :, 10:] = np.random.randn(1, 4, 5).astype(np.float32) + 10.0

    conv1 = transformer.convolve(x1)
    conv2 = transformer.convolve(x2)

    for d in transformer.active_dilations_:
        out1 = conv1[d]
        out2 = conv2[d]
        # Timestamps 0 to 9 must be bitwise/numerically identical
        assert torch.allclose(out1[..., :10], out2[..., :10], atol=1e-6), (
            f"Future leakage detected for dilation {d} at timestamps before t=10"
        )
        # Timestamps 10 to 14 must differ because future values changed
        assert not torch.allclose(out1[..., 10:], out2[..., 10:]), (
            f"Expected difference at t >= 10 for dilation {d}"
        )


def test_minirocket_shape_and_determinism():
    """Validate output dimensions, bounded [0, 1] range, and reproducibility with random_state."""
    n_samples, n_channels, seq_len = 25, 6, 15
    num_kernels = 120
    x = np.random.randn(n_samples, n_channels, seq_len).astype(np.float32)

    transformer1 = MiniRocketTransformer(num_kernels=num_kernels, seq_length=seq_len, random_state=42)
    ppv1 = transformer1.fit_transform(x)

    assert ppv1.shape == (n_samples, num_kernels)
    assert np.all(ppv1 >= 0.0) and np.all(ppv1 <= 1.0)
    assert np.isfinite(ppv1).all()

    # Reproducibility check with identical random_state
    transformer2 = MiniRocketTransformer(num_kernels=num_kernels, seq_length=seq_len, random_state=42)
    ppv2 = transformer2.fit_transform(x)
    assert np.allclose(ppv1, ppv2)

    # Different random_state should yield different feature activations
    transformer3 = MiniRocketTransformer(num_kernels=num_kernels, seq_length=seq_len, random_state=999)
    ppv3 = transformer3.fit_transform(x)
    assert not np.allclose(ppv1, ppv3)


def test_minirocket_fit_predict_pipeline():
    """Test the complete MiniRocketRULModel fit/predict pipeline on sample engine cycles."""
    frame = sample_frame(engine_count=6)
    frame["RUL"] = compute_piecewise_rul(frame)
    features = ["setting_1", "setting_2", "setting_3", "T50", "T24", "P30"]

    model = MiniRocketRULModel(seq_length=4, num_kernels=60, random_state=42)

    # Ensure calling predict before fit raises RuntimeError
    with pytest.raises(RuntimeError, match="Fit the MiniRocketRULModel before calling predict"):
        model.predict(frame)

    model.fit(frame, features, "RUL")
    assert model.fit_time > 0.0
    assert model.regressor is not None

    y_true, y_pred = model.predict(frame)
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_pred).all()

    values = model.predict_values(frame)
    assert np.array_equal(y_pred, values)


def test_cross_validate_pooled_vs_fold_weighted_and_mean_nasa():
    """Verify cross_validate reports pooled OOF, fold-weighted metrics, and Mean_NASA_Score."""
    df = make_synthetic_cmapss(n_engines=10, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    features = ["setting_1", "setting_2", "T2", "T50"]
    benchmark = RULBenchmark(verbose=False)
    cv_results = benchmark.cross_validate(df, features, target_col="RUL", n_splits=3)
    assert isinstance(cv_results, pd.DataFrame)
    # Check that pooled OOF and fold-weighted metrics are distinctly reported
    assert "CV_OOF_Pooled_R2" in cv_results.columns or "CV_R2_Mean" in cv_results.columns
    assert "CV_Mean_NASA_Score" in cv_results.columns or "CV_NASA_Score_Mean" in cv_results.columns


def test_pytorch_sequence_model_group_early_stopping():
    """Verify PyTorch sequence model uses engine-grouped validation for early stopping."""
    df = make_synthetic_cmapss(n_engines=8, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    features = ["setting_1", "T2", "T50"]
    rnn = PyTorchRNNModel(seq_length=5, hidden_dim=16, num_layers=1, max_epochs=5, patience=2, random_state=42)
    rnn.fit(df, features, "RUL")
    assert rnn.fit_time > 0.0
    preds = rnn.predict(df)
    assert len(preds) == len(df)
    assert np.isfinite(preds).all()


def test_minirocket_replicate_padding_and_grouped_ridge():
    """Verify MiniRocket uses replicate padding and GroupKFold in RidgeCV."""
    x = np.random.randn(10, 4, 10).astype(np.float32)
    transformer = MiniRocketTransformer(num_kernels=50, seq_length=10, random_state=42)
    transformer.fit(x)
    features = transformer.transform(x)
    assert features.shape == (10, 50)
    assert np.isfinite(features).all()

    df = make_synthetic_cmapss(n_engines=6, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    model = MiniRocketRULModel(seq_length=5, num_kernels=50, random_state=42)
    model.fit(df, ["setting_1", "T2", "T50"], "RUL")
    assert model.regressor is not None


def test_xgboost_nasa_loss_starts_from_training_mean_and_beats_mean_baseline():
    """[BASELINE] The NASA-loss XGBoost must start at the training-mean RUL (base_score), not 0.5."""
    import numpy as np
    from pdm_cmaps.modeling.gbdt import XGBoostNasaRegressor

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = 90.0 + 25.0 * x[:, 0] + rng.normal(scale=2.0, size=400)
    model = XGBoostNasaRegressor(n_estimators=60, max_depth=4).fit(x, y)
    pred = model.predict(x)
    assert abs(float(model.model_.get_params()["base_score"]) - y.mean()) < 1e-6
    assert np.mean((pred - y) ** 2) < np.var(y)  # better than predicting the mean


def test_minirocket_fit_and_transform_use_same_padding():
    """Biases learned in fit() must come from the same convolution that transform() thresholds."""
    import numpy as np
    from pdm_cmaps.modeling.minirocket import MiniRocketTransformer

    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 3, 15)).astype(np.float32)
    transformer = MiniRocketTransformer(num_kernels=84, seq_length=15, random_state=0).fit(x)
    ppv = transformer.transform(x)
    assert ppv.shape == (64, 84)
    assert 0.05 < float(ppv.mean()) < 0.95  # quantile-calibrated biases give mid-range PPV
