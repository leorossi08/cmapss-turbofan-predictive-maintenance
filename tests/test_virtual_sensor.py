"""Sensor virtual, resíduo e deslocamento por motor (engine offset)."""

import numpy as np
import pandas as pd
from pdm_cmaps.detection.virtual_sensor import VirtualSensor

from factories import make_synthetic_cmapss


def _offset_trace(n_engines=6, cycles=120, seed=0, constant=False):
    """Leitura = deslocamento do motor + ruído pequeno + degradação depois do ciclo 60; o modelo prevê a média da frota."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 5.0, n_engines)
    rows = []
    for u in range(n_engines):
        for c in range(1, cycles + 1):
            actual = 100.0 if constant else 100.0 + offsets[u] + rng.normal(0, 0.3) + max(0, c - 60) * 0.5
            rows.append({"sensor": "S", "model": "M", "unit_id": u, "cycle": c, "actual": actual, "prediction": 100.0})
    return pd.DataFrame(rows)


def _mewma_offset_world(n_engines=10, seed=3):
    df = make_synthetic_cmapss(n_engines=n_engines, seed=seed)
    rng = np.random.default_rng(seed)
    for col in ("T50", "T30", "P30"):
        df[col] = df[col] + df["unit_id"].map({u: rng.normal(0, 4.0) for u in df["unit_id"].unique()})
    df["regime_id"] = 0
    return df[df["unit_id"] <= 7], df[df["unit_id"] > 7]


def test_canonical_exogenous_virtual_sensor_and_fallback():
    """Verify VirtualSensor uses canonical exogenous features and handles sparse regimes via fallback."""
    from pdm_cmaps.detection.virtual_sensor import (
        CANONICAL_EXOGENOUS_FEATURES,
        VirtualSensor,
    )

    assert "setting_1" in CANONICAL_EXOGENOUS_FEATURES
    assert "T2" in CANONICAL_EXOGENOUS_FEATURES
    assert "T50" not in CANONICAL_EXOGENOUS_FEATURES  # Target is not exogenous

    df = make_synthetic_cmapss(n_engines=6, seed=42)
    df["regime_id"] = df["cycle"] % 6
    # Fit virtual sensor with canonical features
    vs = VirtualSensor(target_sensor="T50", healthy_cycle_limit=30, random_state=42)
    # Check that it fits without error even if a regime has limited samples
    vs.fit(df)
    assert len(vs.models) > 0
    preds = vs.predict(df)
    assert "prediction" in preds.columns
    assert "residual" in preds.columns
    assert np.isfinite(preds["prediction"]).all()


def test_todo4_virtual_sensor_strictly_exogenous():
    """TODO-4 & F4: Virtual sensor feature_cols in exogenous mode contains only canonical exogenous features."""
    from pdm_cmaps.detection.virtual_sensor import CANONICAL_EXOGENOUS_FEATURES
    train_df = make_synthetic_cmapss(n_engines=4, seed=42)
    train_df["regime_id"] = 0
    vs = VirtualSensor(target_sensor="T50", mode="exogenous")
    vs.fit(train_df)

    # Verify no degraded internal sensors are among feature_cols
    forbidden = {"T24", "T30", "T50", "P30", "Ps30", "BPR", "W31", "W32"}
    assert not set(vs.feature_cols).intersection(forbidden)
    for col in vs.feature_cols:
        assert col in CANONICAL_EXOGENOUS_FEATURES


def test_virtual_sensor_r2_separates_engine_offset_from_degradation():
    from pdm_cmaps.detection.virtual_sensor_metrics import virtual_sensor_r2

    out = virtual_sensor_r2(_offset_trace(), healthy_limit=50).iloc[0]
    assert out["r2_late_model"] < 0.2                 # o modelo é a média da frota: não explica o deslocamento dos motores
    assert out["r2_late_motor_base"] > 0.9            # a média do próprio motor (ciclos 1 a 25) explica quase tudo
    assert out["r2_late_model_offset"] > 0.9          # modelo + deslocamento do motor também
    assert out["pct_between_engine"] > 90.0
    assert out["r2_holdout"] < out["r2_healthy"]      # a degradação afasta a leitura da média saudável


def test_virtual_sensor_r2_constant_sensor_is_nan_and_columns_are_idempotent(tmp_path):
    from pdm_cmaps.detection.virtual_sensor_metrics import virtual_sensor_r2
    from pdm_cmaps.experiments.virtual_sensor_models import R2_COLUMNS, add_r2_columns

    out = virtual_sensor_r2(_offset_trace(constant=True), healthy_limit=50).iloc[0]
    assert np.isnan(out["r2_holdout"]) and np.isnan(out["pct_between_engine"])

    tr = _offset_trace()
    tr.to_csv(tmp_path / "trace.csv.gz", index=False)
    pd.DataFrame([{"sensor": "S", "model": "M", "rmse_holdout": 1.0}]).to_csv(tmp_path / "model_scores.csv", index=False)
    first = add_r2_columns(tmp_path, 50)
    second = add_r2_columns(tmp_path, 50)               # rodar de novo não duplica colunas
    assert list(first.columns) == list(second.columns) == ["sensor", "model", "rmse_holdout"] + R2_COLUMNS


def test_virtual_sensor_engine_offset_removes_engine_to_engine_shift():
    """Com engine_offset=True o z de cada motor sai centrado em zero na janela saudável; sem ele, o deslocamento do motor sobra."""
    from pdm_cmaps.detection.virtual_sensor import VirtualSensor

    df = make_synthetic_cmapss(n_engines=10, seed=3)
    rng = np.random.default_rng(3)
    df["T50"] = df["T50"] + df["unit_id"].map({u: rng.normal(0, 8.0) for u in df["unit_id"].unique()})  # deslocamento por motor
    df["regime_id"] = 0
    train, test = df[df["unit_id"] <= 7], df[df["unit_id"] > 7]
    H = 30

    def window_means(offset):
        vs = VirtualSensor(target_sensor="T50", healthy_cycle_limit=H, random_state=0, engine_offset=offset, min_consecutive_breaches=2)
        scored, _, _ = vs.evaluate(train, test)
        return scored[scored["cycle"] <= H].groupby("unit_id")["z_score"].mean().abs().mean(), vs

    plain, _ = window_means(False)
    centred, vs = window_means(True)
    assert centred < 0.05 < plain            # z médio por motor na janela: ~0 com o deslocamento removido
    assert vs.engine_z_scale_ > 0
    default_vs = VirtualSensor(target_sensor="T50", healthy_cycle_limit=H, random_state=0)
    assert default_vs.engine_offset is True   # padrão adotado; engine_offset=False reproduz o comportamento anterior
    assert VirtualSensor(target_sensor="T50", engine_offset=False).engine_offset is False


def test_virtual_sensor_engine_offset_is_per_regime_and_keeps_single_regime_behaviour():
    """Em multirregime o deslocamento é por (motor, regime): um regime de σ grande não pode distorcer a previsão do outro."""
    from pdm_cmaps.detection.virtual_sensor import VirtualSensor

    vs = VirtualSensor(target_sensor="T50", healthy_cycle_limit=20, engine_offset=True)
    z = pd.DataFrame({"unit_id": [1] * 8, "cycle": [1, 2, 3, 4, 5, 6, 7, 8], "regime_id": [0, 0, 0, 0, 1, 1, 1, 9],
                      "z_score": [2.0, 2.0, 2.0, 2.0, 100.0, 100.0, 100.0, 5.0]})
    off = vs._engine_regime_offsets(z)
    assert off.loc[(1, 0)] == 2.0 and off.loc[(1, 1)] == 100.0     # cada regime com a sua média
    assert off.loc[(1, 9)] == 0.0                                     # menos de 3 linhas: sem correção
    looked = vs._lookup_offsets(z, off)
    assert looked.tolist() == [2.0] * 4 + [100.0] * 3 + [0.0]


def test_mewma_engine_offset_removes_engine_level():
    """Com engine_offset, o z de cada motor tem média 0 nos primeiros H ciclos (por regime); desde 2026-11-18 o padrão é ligado, como no T50."""
    import numpy as np
    import pandas as pd
    from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor

    assert MultivariateResidualMonitor().engine_offset is True   # padrão do protocolo principal desde 2026-11-18
    assert MultivariateResidualMonitor(engine_offset=False).engine_offset is False
    mon = MultivariateResidualMonitor(healthy_cycle_limit=10, engine_offset=True)
    rng = np.random.default_rng(0)
    rows = []
    for unit, level in ((1, 5.0), (2, -3.0)):
        for cycle in range(1, 31):
            rows.append({"unit_id": unit, "cycle": cycle, "regime_id": 0, "z": level + rng.normal(0, 0.1)})
    df = pd.DataFrame(rows)
    out = mon._remove_offset(df["z"].to_numpy(), df)
    for unit in (1, 2):
        m = (df["unit_id"] == unit) & (df["cycle"] <= 10)
        assert abs(out[m.to_numpy()].mean()) < 1e-9
    off = MultivariateResidualMonitor(healthy_cycle_limit=10, engine_offset=False)
    assert (off._remove_offset(df["z"].to_numpy(), df) == df["z"].to_numpy()).all()


def test_model_scores_apply_the_engine_offset_like_score():
    """`model_scores` (fault_model_study e fault_hard_study) precisa dar o mesmo T² que `MultivariateResidualMonitor.score` com o deslocamento ligado."""
    from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
    from pdm_cmaps.experiments.fault_model_study import MEWMA, model_scores

    train, test = _mewma_offset_world()
    H = 30
    for offset in (False, True):
        mv = MultivariateResidualMonitor(sensors=("T50", "T30", "P30"), healthy_cycle_limit=H, random_state=0, engine_offset=offset).fit(train)
        scored = mv.score(test).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        scores, _ = model_scores(mv, test, set(), H)
        ratio = scores[MEWMA].sort_values(["blind_id", "cycle"]).reset_index(drop=True)["ratio"].to_numpy()
        assert np.allclose(ratio, (scored["t2"] / scored["ucl"]).to_numpy())
        # gráfico por sensor: o limite da EWMA foi calibrado na escala do z que o alimenta (média de |z| ~ a calibrada)
        z = mv.chart_z(mv.virtual_["T50"], mv.virtual_["T50"].predict(test))
        calib_mean = mv.virtual_["T50"].ewma_detector.abs_z_mean
        assert 0.5 * calib_mean < np.abs(z[test["cycle"].to_numpy() <= H]).mean() < 2.0 * calib_mean


def test_mewma_out_of_window_offset_calibrates_and_measures_after_the_offset_window():
    """Com offset_window < H e calibrate_out_of_window, μ, Σ e o limite saem só dos ciclos (W, H]; o falso alarme da janela também; o padrão não muda."""
    from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor

    train, test = _mewma_offset_world()
    H, W = 30, 15
    common = dict(sensors=("T50", "T30", "P30"), healthy_cycle_limit=H, random_state=0, engine_offset=True)
    inside = MultivariateResidualMonitor(**common, offset_window=W).fit(train)
    outside = MultivariateResidualMonitor(**common, offset_window=W, calibrate_out_of_window=True).fit(train)
    assert MultivariateResidualMonitor().calibrate_out_of_window is False
    assert inside.ucl_ != outside.ucl_
    assert len(outside.calibration_t2_) < len(inside.calibration_t2_)
    # o limite fora da janela vem do quantil dos T² calibrados só nos ciclos (W, H]
    assert np.isclose(outside.ucl_, np.quantile(outside.calibration_t2_["t2"], 1 - outside.alpha))
    scored = outside.score(test)
    _, per, _ = outside.alarms(scored)
    manual = scored.assign(win=(scored["cycle"] > W) & (scored["cycle"] <= H)).query("win").groupby("unit_id")["breach"].mean()
    assert np.allclose(per.set_index("unit_id")["healthy_far"].sort_index().to_numpy(), manual.sort_index().to_numpy())
    assert 0.0 <= outside.offset_uncovered_pct(test) <= 100.0


def test_virtual_sensor_out_of_window_calibration_uses_only_cycles_after_the_offset_window():
    from pdm_cmaps.detection.virtual_sensor import VirtualSensor

    train, test = _mewma_offset_world()
    H, W = 30, 15
    vs = VirtualSensor(target_sensor="T50", healthy_cycle_limit=H, random_state=0, engine_offset=True, offset_window=W, calibrate_out_of_window=True)
    scored, per, _ = vs.evaluate(train, test)
    cp = vs.calibration_predictions_
    off = vs._engine_regime_offsets(cp)
    z_after = (cp["z_score"] - vs._lookup_offsets(cp, off))[cp["cycle"] > W]
    assert np.isclose(vs.engine_z_scale_, z_after.std())
    base = VirtualSensor(target_sensor="T50", healthy_cycle_limit=H, random_state=0, engine_offset=True, offset_window=W)
    base.evaluate(train, test)
    assert not np.isclose(base.engine_z_scale_, vs.engine_z_scale_)
    assert (VirtualSensor(target_sensor="T50", engine_offset=True).calibrate_out_of_window is False)
