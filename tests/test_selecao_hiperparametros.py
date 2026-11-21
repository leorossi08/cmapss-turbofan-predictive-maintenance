"""Busca de hiperparâmetros (regra do erro-padrão, janela, validação por motor) e as métricas novas de RUL, sensores virtuais e alertas."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from pdm_cmaps.detection.virtual_sensor_metrics import error_metrics
from pdm_cmaps.domain.rul import compute_piecewise_rul, extended_rul_metrics
from pdm_cmaps.monitoring.alerts import CriticalityPolicy, evaluate_strategies
from factories import make_synthetic_cmapss

SENSORS = ["T24", "T30", "T50", "P30", "Nf", "Nc"]


def test_extended_rul_metrics_hand_computed():
    m = extended_rul_metrics([5, 12, 30, 60], [8, 10, 40, 50])  # d = +3, -2, +10, -10
    assert m["Bias"] == pytest.approx(0.25)
    assert m["Median_AE"] == pytest.approx(6.5)
    assert m["P90_AE"] == pytest.approx(10.0)
    assert m["Hit_within_10"] == 1.0 and m["Hit_within_20"] == 1.0
    assert (m["Precision_c15"], m["Recall_c15"], m["F1_c15"]) == (1.0, 1.0, 1.0)
    assert m["Precision_c10"] == pytest.approx(0.5) and m["Recall_c10"] == 1.0 and m["F1_c10"] == pytest.approx(2 / 3)
    assert m["PR_AUC_c15"] == pytest.approx(1.0)  # os dois motores críticos têm as duas menores RUL previstas


def test_extended_rul_metrics_class_without_members_is_nan():
    m = extended_rul_metrics([100, 120, 130], [90, 110, 140])
    assert np.isnan(m["PR_AUC_c10"]) and np.isnan(m["Precision_c10"]) and np.isnan(m["Recall_c10"])


def test_virtual_sensor_error_metrics():
    g = pd.DataFrame({"actual": [1.0, 2.0, 3.0, 4.0], "prediction": [1.0, 2.0, 3.0, 6.0], "regime_id": [0, 0, 1, 1]})
    mae, rmse, nrmse = error_metrics(g)
    assert mae == pytest.approx(0.5) and rmse == pytest.approx(1.0)
    assert nrmse == pytest.approx(2.0)  # desvio em torno da média de cada regime = 0,5


def test_alert_precision_recall_f1_per_strategy():
    t = pd.DataFrame({"unit_id": 1, "cycle": range(1, 7), "RUL_true": [50, 40, 30, 20, 10, 1],
                      "level_true": [0, 0, 1, 2, 3, 3], "level_point": [0, 1, 1, 2, 3, 0]})
    for other in ("level_interval", "level_interval+anomaly", "level_interval+anomaly_mv"):
        t[other] = t["level_point"]
    row = evaluate_strategies(t, CriticalityPolicy()).set_index("Strategy").loc["point"]
    assert row["Alert_precision"] == 1.0
    assert row["Alert_recall"] == pytest.approx(2 / 3)
    assert row["Alert_F1"] == pytest.approx(0.8)


def test_one_standard_error_rule_picks_the_smallest_window_within_one_se():
    optuna = pytest.importorskip("optuna")  # noqa: F841
    from pdm_cmaps.experiments.selecao_hiperparametros import escolher_um_se

    t = pd.DataFrame({"rmse_mean": [14.0, 14.3, 14.4, 16.0], "rmse_se": [0.5, 0.4, 0.4, 0.3], "window": [40, 10, 20, 5], "trial": [0, 1, 2, 3]})
    assert int(escolher_um_se(t)["window"]) == 10  # limite = 14,0 + 0,5; entre 14,0 (40), 14,3 (10) e 14,4 (20) fica a menor janela
    only_best = t.assign(rmse_se=[0.0, 0.0, 0.0, 0.0])
    assert int(escolher_um_se(only_best)["window"]) == 40  # sem erro-padrão, só o melhor


def test_search_space_sets_the_window_by_model_family():
    optuna = pytest.importorskip("optuna")
    from pdm_cmaps.experiments.selecao_hiperparametros import WINDOWS, sugerir

    base = {"representation": "regime", "alpha": 1.0}
    ridge_plain = sugerir(optuna.trial.FixedTrial({**base, "temporal": False}), "Ridge")
    assert ridge_plain["window"] == 0 and ridge_plain["temporal"] is False and ridge_plain["representation"] == "regime"
    ridge_win = sugerir(optuna.trial.FixedTrial({**base, "temporal": True, "window": 30}), "Ridge")
    assert ridge_win["window"] == 30 and 30 in WINDOWS
    lstm = sugerir(optuna.trial.FixedTrial({"representation": "global", "window": 20, "learning_rate": 0.002, "batch_size": 256, "dropout": 0.1,
                                            "hidden_dim": 32, "num_layers": 1}), "PyTorch RNN (LSTM)")
    assert lstm["window"] == 20 and lstm["temporal"] is False and lstm["hyper"]["hidden_dim"] == 32


def test_validation_is_grouped_by_engine_and_runs_with_regime_and_temporal_features():
    pytest.importorskip("optuna")
    from pdm_cmaps.experiments.selecao_hiperparametros import validar

    df = make_synthetic_cmapss(n_engines=9, min_cycles=40, max_cycles=60, seed=5)
    df["RUL"] = compute_piecewise_rul(df)
    params = {"representation": "regime", "temporal": True, "window": 10, "hyper": {"alpha": 10.0}}
    scores = validar(df, SENSORS, 6, "Ridge", params, seed=1, n_splits=3)
    assert len(scores) == 3 and all(np.isfinite(scores)) and all(s > 0 for s in scores)
    again = validar(df, SENSORS, 6, "Ridge", params, seed=1, n_splits=3)
    assert scores == again  # determinístico


def test_model_builders_construct_every_tabular_family():
    pytest.importorskip("optuna")
    from pdm_cmaps.experiments.selecao_hiperparametros import montar_modelo

    for name, hyper in {"XGBoost": {"n_estimators": 10, "max_depth": 3}, "LightGBM": {"n_estimators": 10, "max_depth": 3}, "CatBoost": {"iterations": 10, "depth": 3},
                        "Random Forest": {"n_estimators": 10, "max_depth": 4}, "XGBoost (NASA Loss)": {"n_estimators": 10, "max_depth": 3, "subsample": 0.8}}.items():
        kind, est = montar_modelo(name, {"hyper": hyper, "window": 0}, seed=1)
        assert kind == "sklearn" and hasattr(est, "fit")


def test_the_search_never_reads_the_official_test_file():
    pytest.importorskip("optuna")
    from pdm_cmaps.experiments import selecao_hiperparametros as sh

    assert "load_test" not in inspect.getsource(sh)
