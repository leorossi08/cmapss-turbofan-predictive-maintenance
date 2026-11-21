"""Detectores de comparação do estudo de injeção (EWMA clássica com sinal e Isolation Forest): fórmulas e propriedades."""
from __future__ import annotations

import numpy as np

from pdm_cmaps.experiments.fault_detection_study import EXTRA_DETECTORS, iforest_ratio, signed_ewma_ratio


def test_signed_ewma_is_zero_without_residual_and_symmetric_in_sign():
    assert np.allclose(signed_ewma_ratio(np.zeros(30)), 0.0)
    z = np.random.default_rng(0).normal(size=200)
    assert np.allclose(signed_ewma_ratio(z), signed_ewma_ratio(-z))


def test_signed_ewma_matches_the_closed_form_limit_for_a_constant_shift():
    lam, L, c = 0.10, 3.0, 1.5
    r = signed_ewma_ratio(np.full(80, c), lam, L)
    t = np.arange(1, 81)
    w = c * (1 - (1 - lam) ** t)                      # EWMA de uma constante partindo de zero
    limit = L * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    assert np.allclose(r, w / limit)
    assert r[-1] > 1.0 > r[0]                         # um deslocamento de 1,5 sigma acaba fora de controle, mas não no primeiro ciclo


def test_signed_ewma_rarely_alarms_on_white_noise():
    r = signed_ewma_ratio(np.random.default_rng(1).normal(size=5000))
    assert (r >= 1.0).mean() < 0.02


def test_iforest_ratio_is_zero_at_the_center_and_large_for_an_outlier():
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(2)
    cal = rng.normal(size=(2000, 3))
    model = IsolationForest(n_estimators=100, random_state=0).fit(cal)
    r = iforest_ratio(model, cal, np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 8.0]]))
    assert r[0] < 0.2 and r[1] > 1.0
    assert (iforest_ratio(model, cal, cal) >= 1.0).mean() < 0.01   # quantil 1 - 0,0027 da própria calibração


def test_extra_detectors_stay_out_of_the_main_protocol():
    from pdm_cmaps.experiments.fault_detection_study import DETECTORS

    assert set(EXTRA_DETECTORS).isdisjoint(DETECTORS) and len(EXTRA_DETECTORS) == 2
