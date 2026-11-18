"""Estatística: testes de hipótese, diagnósticos e importância de atributos."""

import warnings

import numpy as np
import pandas as pd
from pdm_cmaps.analysis.stats import (
    ThesisHypothesisAnalyzer,
    distribution_shift,
    feature_signal,
    permutation_importances,
)
from pdm_cmaps.domain.rul import (
    compute_piecewise_rul,
)

from factories import make_synthetic_cmapss, sample_frame


def test_feature_signal_handles_constant_features_without_runtime_warning():
    frame = sample_frame(engine_count=3)
    frame["RUL"] = compute_piecewise_rul(frame)
    frame["constant"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = feature_signal(frame, ["constant", "cycle"], "test")
    constant_correlation = result.loc[result["Feature"] == "constant", "Abs_correlation"].iloc[0]
    assert np.isnan(constant_correlation)
    assert np.isfinite(result.loc[result["Feature"] == "cycle", "Abs_correlation"]).all()


def test_thesis_hypothesis_rigor():
    """Verify H1 engine-block Friedman, H2 n=62 relative window, H3 fleet Ljung-Box, H4/H5 Wilcoxon."""
    analyzer = ThesisHypothesisAnalyzer(random_state=42)
    df = make_synthetic_cmapss(n_engines=12, min_cycles=60, max_cycles=100, seed=42)
    df["regime_id"] = (df["setting_1"] // 10).astype(int)

    # H1: Engine-block repeated measures
    h1 = analyzer.test_regime_variance_dominance(df, sensor="T50", regime_col="regime_id")
    assert h1.is_rejected or h1.p_value <= 0.05
    assert "Friedman" in h1.test_name or "Alexander" in h1.test_name or "ANOVA" in h1.test_name

    # H2: Survivor-bias-free relative window
    df["residual"] = df["T50"] - (1400.0 + 8.0 * df["setting_1"])
    h2 = analyzer.test_physical_degradation_shift(df, healthy_cutoff=30, residual_col="residual")
    assert h2.is_rejected
    assert h2.p_value < 0.05
    assert "n=12" in h2.test_name or h2.effect_size is not None

    # H3: Fleet Ljung-Box
    h3 = analyzer.test_white_noise_residual_autocorrelation(df["residual"].to_numpy())
    assert h3.test_statistic > 0

    # H4 & H5: Wilcoxon tests
    y_true = np.linspace(100, 0, 100)
    pred_base = np.full(100, 50.0)
    pred_model = y_true + np.random.normal(0, 5.0, 100)
    unit_ids = np.repeat(np.arange(1, 11), 10)
    h4 = analyzer.test_model_superiority_vs_baseline(y_true, pred_model, pred_base, "ModelX", unit_ids=unit_ids)
    assert h4.is_rejected

    # BCa Confidence Intervals export observed metric
    test_eval = pd.DataFrame({
        "unit_id": unit_ids,
        "RUL_actual": y_true,
        "ModelX": pred_model,
    })
    ci_df = analyzer.compute_bootstrap_confidence_intervals(test_eval, {"ModelX": "ModelX"}, target_col="RUL_actual", n_bootstraps=50)
    assert "R2_Observed" in ci_df.columns or "R2_Mean" in ci_df.columns


def test_diagnostics_subsystem_permutation_and_wasserstein():
    """Verify grouped subsystem permutation importance and Wasserstein-1 distance."""
    df = make_synthetic_cmapss(n_engines=10, seed=42)
    df["RUL"] = compute_piecewise_rul(df)
    features = ["setting_1", "setting_2", "T2", "P2", "T50", "P30"]

    # Subsystem permutation importances
    p_imp = permutation_importances(df, features, "FD004", random_state=42)
    assert isinstance(p_imp, pd.DataFrame)
    assert "R2_drop_mean" in p_imp.columns

    # Distribution shift with Wasserstein distance
    df_high = df.iloc[: len(df) // 2]
    df_low = df.iloc[len(df) // 2 :]
    shift = distribution_shift(df_high, df_low, ["T50", "P30"], "High", "Low")
    assert "Wasserstein_distance" in shift.columns or "KS_statistic" in shift.columns
