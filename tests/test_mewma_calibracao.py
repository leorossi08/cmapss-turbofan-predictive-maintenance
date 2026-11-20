"""Calibração sintética do limite do T² da MEWMA (detection/multivariate.py, calibration="synthetic")."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor

P, W, ENG, CYC = 4, 50, 40, 50


def _setup(seed=0, offset_sd=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, ENG + 1):
        off = rng.normal(0, offset_sd, P)  # nível próprio do motor, que a previsão não conhece
        for c in range(1, CYC + 1):
            rows.append([u, c, *(off + rng.normal(0, 1, P))])
    df = pd.DataFrame(rows, columns=["unit_id", "cycle", *"abcd"]).set_index(["unit_id", "cycle"])
    regime = pd.Series(0, index=df.index)
    mv = MultivariateResidualMonitor(sensors=tuple("abcd"), healthy_cycle_limit=W, engine_offset=True, calibration="synthetic",
                                     synthetic_reps=10, synthetic_horizon=120, random_state=1)
    # o que o fit faria: z centrado em cada motor DENTRO da janela, depois mu e Sigma
    centered = df - df.groupby(level="unit_id").transform("mean")
    mv.mu_ = centered.to_numpy().mean(axis=0)
    mv.sigma_inv_ = np.linalg.inv(np.cov(centered.to_numpy(), rowvar=False))
    return mv, df, regime, centered


def test_default_calibration_is_the_previous_protocol():
    assert MultivariateResidualMonitor().calibration == "empirical"
    with pytest.raises(ValueError):
        MultivariateResidualMonitor(calibration="outra")


def test_synthetic_limit_is_higher_than_the_in_window_empirical_limit():
    """No protocolo anterior o T² da calibração é medido onde a média do z é zero por construção, então o limite sai baixo demais; a calibração
    sintética mede o T² depois da janela, com o erro de estimar o deslocamento, e dá um limite maior."""
    mv, df, regime, centered = _setup()
    t2_in_window = mv._t2(centered.to_numpy(), centered.index.get_level_values("unit_id").to_numpy())
    synth = mv._synthetic_post_window_t2(df, regime)
    assert synth.mean() > t2_in_window.mean() * 1.02
    assert np.quantile(synth, 0.9973) > np.quantile(t2_in_window, 0.9973)


def test_synthetic_calibration_shape_and_determinism():
    mv, df, regime, _ = _setup()
    a = mv._synthetic_post_window_t2(df, regime)
    b = mv._synthetic_post_window_t2(df, regime)
    assert a.shape == (ENG * 10 * 120,)          # motores x réplicas x ciclos depois da janela
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_synthetic_mean_t2_matches_the_theory_for_offset_estimation_error():
    """Com ruído i.i.d. e o deslocamento estimado de W ciclos, o erro de estimação (variância 1/W) soma-se à variância estacionária da EWMA
    (λ/(2−λ)): E[T²] ≈ p · (1 + (1/W) / (λ/(2−λ))). Para p=4, λ=0,1 e W=50 são 5,5; a média em janela, centrada nela mesma, fica ABAIXO de p."""
    mv, df, regime, centered = _setup(seed=2)
    lam = mv.lam
    expected = P * (1 + (1 / W) / (lam / (2 - lam)))
    synth = mv._synthetic_post_window_t2(df, regime)
    assert abs(synth.mean() / expected - 1) < 0.15, (synth.mean(), expected)
    t2_in = mv._t2(centered.to_numpy(), centered.index.get_level_values("unit_id").to_numpy())
    assert t2_in.mean() < P, "a média em janela deveria ficar abaixo de p (centrada nela mesma)"


def test_synthetic_t2_has_a_transient_right_after_the_window():
    """Logo depois da janela a EWMA ainda carrega a memória da janela (z centrado por construção), então o T² médio cresce até o estacionário.
    Por isso a validação em dados reais nos primeiros ciclos pós-janela exige o limite sintético RESTRITO a esses ciclos."""
    mv, df, regime, _ = _setup()
    m = mv._synthetic_post_window_t2(df, regime, by_cycle=True)
    assert m.shape == (ENG * 10, 120)
    assert np.array_equal(m.ravel(), mv._synthetic_post_window_t2(df, regime))
    assert m[:, :10].mean() < m[:, -50:].mean() * 0.95
    assert np.quantile(m[:, :25].ravel(), 0.99) < np.quantile(m.ravel(), 0.99)
