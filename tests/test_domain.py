"""Regras puras do domínio: alvo de RUL, métricas e esquema."""

import numpy as np
from pdm_cmaps.domain.rul import (
    clip_predictions,
    compute_piecewise_rul,
    evaluate_rul_predictions,
    nasa_scoring_function,
)
from pdm_cmaps.domain.schema import DEFAULT_RUL_CAP, SENSOR_COLUMNS

from factories import sample_frame


def test_capped_rul_reaches_zero_and_caps_at_limit():
    frame = sample_frame(engine_count=2)
    rul = compute_piecewise_rul(frame, rul_max=3)
    assert rul.max() == 3.0
    assert rul.min() == 0.0


def test_late_rul_error_has_larger_nasa_penalty():
    truth = np.array([50.0])
    early, _ = nasa_scoring_function(truth, np.array([45.0]))
    late, _ = nasa_scoring_function(truth, np.array([55.0]))
    assert late > early


def test_nasa_scoring_function_overflow_guard():
    """Verify nasa_scoring_function does not trigger RuntimeWarning: overflow on extreme differences."""
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([600.0, -600.0])  # Extreme deviations that would overflow exp(60)
    score, mean_score = nasa_scoring_function(y_true, y_pred)
    assert np.isfinite(score)
    assert np.isfinite(mean_score)
    assert score > 0.0


def test_clip_predictions_and_stratified_evaluation():
    """Verify prediction clipping in [0, 125] and stratified critical-zone scoring."""
    y_true = np.array([125.0, 80.0, 30.0, 10.0, 0.0])
    y_pred = np.array([150.0, 90.0, 35.0, -5.0, -20.0])
    clipped = clip_predictions(y_pred, lower=0.0, upper=DEFAULT_RUL_CAP)
    assert clipped.max() <= DEFAULT_RUL_CAP
    assert clipped.min() >= 0.0

    # Evaluation with clipping
    metrics_unclipped = evaluate_rul_predictions(y_true, y_pred, clip=False)
    metrics_clipped = evaluate_rul_predictions(y_true, y_pred, clip=True)
    assert metrics_clipped["RMSE"] <= metrics_unclipped["RMSE"] + 1e-6

    # Critical zone evaluation (RUL < 50)
    from pdm_cmaps.domain.rul import evaluate_stratified_rul
    strat = evaluate_stratified_rul(y_true, y_pred, critical_threshold=50.0)
    assert "overall" in strat
    assert "critical_zone" in strat
    assert strat["critical_zone"]["N_samples"] == 3


def test_todo6_physical_directionality_schema():
    """TODO-6: Positive drift, negative drift and exogenous sensors form disjoint partitions with valid units."""
    from pdm_cmaps.domain.schema import (
        EXOGENOUS_CONSTANT_SENSORS,
        NEGATIVE_DRIFT_SENSORS,
        POSITIVE_DRIFT_SENSORS,
        SENSOR_UNITS,
    )
    s_pos = set(POSITIVE_DRIFT_SENSORS)
    s_neg = set(NEGATIVE_DRIFT_SENSORS)
    s_exo = set(EXOGENOUS_CONSTANT_SENSORS)

    assert len(s_pos.intersection(s_neg)) == 0
    assert len(s_pos.intersection(s_exo)) == 0
    assert len(s_neg.intersection(s_exo)) == 0

    # Every sensor column has a declared physical unit
    for sensor in SENSOR_COLUMNS:
        assert sensor in SENSOR_UNITS
        assert isinstance(SENSOR_UNITS[sensor], str) and len(SENSOR_UNITS[sensor]) > 0
