"""Estudos executáveis: injeção de falhas, detecção, janela saudável, poder e bootstrap."""

import numpy as np
import pandas as pd
import pytest
from pdm_cmaps.domain.schema import COLUMNS, SENSOR_COLUMNS


def _fi_world(seed=0):
    from pdm_cmaps.experiments import fault_injection as fi

    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, 9):
        life = 260
        for c in range(1, life + 1):
            row = {col: 0.0 for col in COLUMNS}
            row.update(unit_id=unit, cycle=c, setting_1=0.0, setting_2=0.0, setting_3=100.0)
            for s in SENSOR_COLUMNS:
                row[s] = 100.0 + rng.normal(0, 1.0) + (0.1 * max(0, c - (life - 100)) if s in ("T50", "Nc") else 0.0)
            rows.append(row)

    class FakeLoader:
        data_dir = None

        def load(self, subset):
            return pd.DataFrame(rows)

    return fi, fi.load_world("FD001", seed, 50, loader=FakeLoader())


def _write_fake_alarm_run(tmp_path, subset="FD001", seed=1, H=25, n_rec=8, detect_all=True):
    """Corrida sintética para o bootstrap do poder: n_rec motores, 4 controles (alarme só em s baixo) e 2 falhas (alarme em todo s <= 20) por motor."""
    import numpy as np
    import pandas as pd
    from pdm_cmaps.experiments.fault_detection_study import DETECTORS, S_GRID
    from pdm_cmaps.experiments.poder_bootstrap import _slug

    rows, bid = [], 0
    for rec in range(1, n_rec + 1):
        for _ in range(4):
            bid += 1
            rows.append({"blind_id": bid, "recipient": rec, "role": "controle", "kind": "", "magnitude": np.nan, "t0": 0, "T": 300})
        for _ in range(2):
            bid += 1
            rows.append({"blind_id": bid, "recipient": rec, "role": "falha", "kind": "degrau", "magnitude": 0.25, "t0": H + 10, "T": H + 90})
    meta = pd.DataFrame(rows)
    arrays = {}
    for d in DETECTORS:
        mat = np.full((len(meta), len(S_GRID)), np.nan)
        ctrl = (meta["role"] == "controle").to_numpy()
        low = S_GRID < 0.5
        mat[np.ix_(ctrl, low)] = H + 50.0                      # controles alarmam só com limiar baixo
        if detect_all:
            mat[np.ix_(~ctrl, S_GRID < 3.0)] = H + 15.0       # falhas detectadas com qualquer limiar abaixo de 3
        slug = _slug(d)
        arrays[f"{slug}__ids"], arrays[f"{slug}__mat"], arrays[f"{slug}__last"] = meta["blind_id"].to_numpy(), mat, np.full(len(meta), H + 100.0)
    path = tmp_path / f"{subset}_seed{seed}_H{H}.npz"
    meta.assign(Subset=subset, Seed=seed, H=H).to_csv(tmp_path / f"{subset}_seed{seed}_H{H}_manifesto.csv", index=False)
    np.savez_compressed(path, **arrays)
    return path


def test_healthy_window_reference_metrics_do_not_depend_on_h():
    """early alarm / FAR are scored against true RUL >= 125, the same cycles for every H."""
    from pdm_cmaps.experiments.healthy_window_study import _reference_metrics, _windows

    cyc = np.arange(1, 301)
    pc = pd.DataFrame({"unit_id": 1, "cycle": cyc, "rul_true": 300 - cyc, "breach": cyc == 120})
    for H in (50, 100):
        m = _reference_metrics(pc, {1: 120}, H, k=2)
        assert m["early_alarm_pct"] == 100.0  # alarm at RUL 180 >= 125
        assert m["far_common_pct"] == pytest.approx(100 / 75)  # cycles 101..175, one breach
    assert _reference_metrics(pc, {1: 200}, 50, 2)["early_alarm_pct"] == 0.0
    train = pd.DataFrame({"unit_id": [1, 1, 2, 2], "cycle": [1, 200, 1, 240]})
    assert _windows(train)["H=25% vida (=55)"] == 55  # 0.25 x median(200, 240)


def test_nc_step_scan_flags_a_core_reset(monkeypatch):
    """A synthetic core-speed reset (high Nc, then back to the healthy level) is found at its cycle."""
    from pdm_cmaps.experiments import rul_prediction_plots as rpp

    rng = np.random.default_rng(0)
    rows = []
    for unit in (1, 2, 3):
        for c in range(1, 201):
            nc = 9000 + rng.normal(0, 1.0) + (8.0 if unit == 1 and 100 <= c < 150 else 0.0)
            rows.append({"unit_id": unit, "cycle": c, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0, "Nc": nc})

    class FakeLoader:
        def load(self, subset):
            return pd.DataFrame(rows)

    scan = rpp.scan_discontinuities("FD001", loader=FakeLoader())
    top = scan.iloc[0]
    assert int(top["unit_id"]) == 1 and int(top["cycle"]) == 150 and bool(top["candidate"])
    assert not scan.loc[scan["unit_id"] != 1, "candidate"].any()


def test_fault_injection_is_causal_blind_and_paired(tmp_path):
    fi, world = _fi_world()
    assert world.signature[fi.INJECT_SENSORS.index("T50")] > 3  # EOL signature recovered in sigma units
    eng = world.test[world.test["unit_id"] == int(world.test["unit_id"].iloc[0])].sort_values("cycle")
    rng = np.random.default_rng(1)
    base = fi.healthy_surrogate(eng, 120, 50, rng)
    assert len(base) == 120 and base["cycle"].tolist() == list(range(1, 121))
    t0 = 60
    for kind in ("degrau", "rampa", "variancia", "vies_sensor"):
        twin = fi.inject(world, base, fi.delta_z(world, base, kind, 1.0, t0, rng))
        pre = base["cycle"] < t0
        assert np.allclose(twin.loc[pre, list(fi.INJECT_SENSORS)], base.loc[pre, list(fi.INJECT_SENSORS)])  # identical before t0
        assert not np.allclose(twin.loc[~pre, list(fi.INJECT_SENSORS)], base.loc[~pre, list(fi.INJECT_SENSORS)])
        assert np.allclose(twin[["setting_1", "setting_2", "setting_3"]], base[["setting_1", "setting_2", "setting_3"]])
    # step of magnitude 1 adds exactly the signature in sigma units
    step = fi.inject(world, base, fi.delta_z(world, base, "degrau", 1.0, t0, rng))
    got = (fi.standardize(world, step) - fi.standardize(world, base))[-1]
    assert np.allclose(got, world.signature)
    # blind files carry the raw schema only
    run_dir = tmp_path / "run"
    fi.generate(world, run_dir, max_engines=2)
    assert list(pd.read_csv(run_dir / "dados_cegos" / "cego_deteccao.csv").columns) == COLUMNS
    assert list(pd.read_csv(run_dir / "dados_cegos" / "cego_rul.csv").columns) == COLUMNS
    # tampering with the sealed key is detected by score()
    key = run_dir / "selado" / "manifesto_deteccao.csv"
    key.write_text(key.read_text() + "\n")
    (run_dir / "saidas").mkdir()
    with pytest.raises(RuntimeError):
        fi.score(world, run_dir, 60.0)


def test_fault_injection_transplant_label_ends_at_last_cycle(tmp_path):
    fi, world = _fi_world()
    fi.generate(world, tmp_path / "r", max_engines=2)
    man = pd.read_csv(tmp_path / "r" / "selado" / "manifesto_rul.csv")
    blind = pd.read_csv(tmp_path / "r" / "dados_cegos" / "cego_rul.csv")
    for row in man[man["role"] == "transplante"].itertuples():
        assert blind.loc[blind["unit_id"] == row.blind_id, "cycle"].max() == row.eol_cycle == row.t0 + fi.TRANSPLANT_N - 1


def test_alarm_cycles_are_causal_monotone_and_start_at_min_cycle():
    from pdm_cmaps.experiments import fault_detection_study as fd

    rng = np.random.default_rng(0)
    cycles = np.arange(1, 201)
    ratio = np.abs(rng.normal(0.5, 0.2, 200))
    ratio[120:] += 1.5
    gate = np.ones(200, bool)
    full = fd.alarm_cycles(ratio, gate, cycles, 51)
    for cut in (110, 140, 180):  # truncation: an alarm inside the prefix is identical, a later one is not seen
        part = fd.alarm_cycles(ratio[:cut], gate[:cut], cycles[:cut], 51)
        inside = full <= cut - fd.K_CONSEC + 1
        assert np.array_equal(part[inside], full[inside])
        assert np.isnan(part[~inside]).all() or (part[~inside] >= full[~inside]).all()
    has = ~np.isnan(full)
    assert np.all(np.diff(full[has]) >= 0) and (np.diff(has.astype(int)) <= 0).all()  # higher s: later or none
    assert np.nanmin(fd.alarm_cycles(ratio, gate, cycles, 130)) >= 130  # monitoring starts at min_cycle
    assert np.isnan(fd.alarm_cycles(ratio[:1], gate[:1], cycles[:1], 1)).all()


def test_control_far_and_arl0_use_exposure_until_first_alarm():
    from pdm_cmaps.experiments import fault_detection_study as fd

    alarms = np.array([[60.0, np.nan], [np.nan, np.nan]])
    far, arl0 = fd.control_far(alarms, np.array([150, 150]), 50)
    assert far[0] == pytest.approx(100 * 1 / (10 + 100)) and arl0[0] == pytest.approx(110)
    assert far[1] == 0 and np.isinf(arl0[1])


def test_detection_study_series_are_blind_and_paired():
    from pdm_cmaps.domain.schema import COLUMNS
    from pdm_cmaps.experiments import fault_detection_study as fd

    fi, world = _fi_world()
    blind, meta = fd.build_series(world, seed=0, n_ctrl=2, n_rep=1)
    assert list(blind.columns) == COLUMNS
    assert {"controle", "falha"} == set(meta["role"]) and meta["blind_id"].is_unique
    ctrl = meta[meta["role"] == "controle"].iloc[0]
    assert blind.loc[blind["unit_id"] == ctrl["blind_id"], "cycle"].max() == fd.HORIZON_CTRL
    sc = fd.raw_t50_scores(world, blind)
    mv = fd.raw_mewma_scores(world, blind, ["T24", "T30", "T50", "P30"])
    assert set(sc["blind_id"]) == set(meta["blind_id"]) == set(mv["blind_id"]) and np.isfinite(sc["ratio"]).all()
    # a large injected step is detected by the raw T50 chart only after t0
    big = meta[(meta["kind"] == "degrau") & (meta["magnitude"] == 0.5)]
    fid = int(big.iloc[0]["blind_id"])
    t0 = int(big.iloc[0]["t0"])
    one = sc[sc["blind_id"] == fid]
    a = fd.alarm_cycles(one["ratio"].to_numpy(), one["gate"].to_numpy(), one["cycle"].to_numpy(), t0)
    assert np.isnan(a).all() or np.nanmin(a) >= t0


def test_hard_fault_family_is_causal_documented_and_paired_delta_is_signed():
    from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
    from pdm_cmaps.experiments import fault_hard_study as fh

    fi, world = _fi_world()
    rng = np.random.default_rng(3)
    eng = world.test[world.test["unit_id"] == int(world.test["unit_id"].iloc[0])].sort_values("cycle")
    base = fh.bootstrap_series(eng, 160, rng)
    specs = [fh.draw_fault(world, rng, 50) for _ in range(60)]
    assert {s["family"] for s in specs} == {"processo", "sensor"} and {s["shape"] for s in specs} >= {"degrau", "rampa", "intermitente"}
    for s in specs:
        dz = fh.fault_dz(s, base, rng)
        assert np.allclose(dz[(base["cycle"] < s["t0"]).to_numpy()], 0)          # nothing before t0
        assert 50 + fh.T0_RANGE[0] <= s["t0"] <= 50 + fh.T0_RANGE[1]
        if s["family"] == "sensor":
            assert (s["weights"] != 0).sum() == 1 and fh.SENS_MAG[0] <= s["mag"] <= fh.SENS_MAG[1]
        else:
            assert fh.PROC_MAG[0] <= s["mag"] <= fh.PROC_MAG[1]
    # paired difference: estimator detects 1 of 2 more faults than the baseline in the same cluster
    keys = {"Subset": "FD001", "Seed": 1, "Chart": "MEWMA", "Mode": "exogenous"}
    series = pd.DataFrame([{**keys, "Estimator": fh.BASELINE, "blind_id": i, "detected": d} for i, d in ((1, False), (2, True))] +
                          [{**keys, "Estimator": "Ridge", "blind_id": i, "detected": d} for i, d in ((1, True), (2, True))])
    man = pd.DataFrame({"Subset": "FD001", "Seed": 1, "blind_id": [1, 2], "recipient": [10, 11], "family": "processo", "mag": 0.1})
    out = fh.paired_delta(series, man, np.random.default_rng(0), n_boot=50)
    assert out.loc[0, "delta_pp"] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        MultivariateResidualMonitor(mode="nonsense")


def test_compare_engine_offset_power_table_pairs_cells(tmp_path):
    from pdm_cmaps.experiments.compare_engine_offset import DETECTORS, power_table

    def write(root, power):
        (root / "runs").mkdir(parents=True)
        pd.DataFrame({"Detector": [DETECTORS[0]] * 2, "kind": ["degrau"] * 2, "magnitude": [0.1, 0.25], "Subset": "FD003", "Seed": 42,
                      "power_pct": power, "far_ctrl_per100": 0.1, "delay_med": 10.0}).to_csv(root / "runs" / "FD003_seed42_H25_pareado.csv", index=False)

    write(tmp_path / "base", [10.0, 40.0])
    write(tmp_path / "offset", [30.0, 35.0])
    out = power_table([tmp_path / "base"], [tmp_path / "offset"]).iloc[0]
    assert out["celulas"] == 2 and out["celulas_melhores"] == 1 and out["celulas_piores"] == 1
    assert out["ganho_medio"] == 7.5 and out["poder_base"] == 25.0 and out["poder_offset"] == 32.5


def test_poder_bootstrap_matches_point_power_and_pairs_detectors(tmp_path):
    import numpy as np
    from pdm_cmaps.experiments.fault_detection_study import DETECTORS
    from pdm_cmaps.experiments.poder_bootstrap import Run, bootstrap, summarize

    runs = [Run(_write_fake_alarm_run(tmp_path, seed=s)) for s in (1, 2)]
    # o s* escolhido deixa os controles fora do orçamento de 0,2 e as falhas dentro: poder 100 % em todos os detectores
    point = runs[0].power(DETECTORS[0], np.ones(len(runs[0].recipients)))
    assert np.allclose(point, 100.0)
    boot = bootstrap(runs, n_boot=60, seed=3)
    power, contr = summarize(runs, boot)
    assert (power["poder_pct"] == 100.0).all() and (power["ic_lo"] == 100.0).all()
    assert (contr["diferenca_pp"] == 0.0).all() and (contr["p_boot"] == 1.0).all()
    # caso sem nenhuma detecção: poder 0
    (tmp_path / "sem").mkdir()
    run0 = Run(_write_fake_alarm_run(tmp_path / "sem", seed=9, detect_all=False))
    assert np.allclose(run0.power(DETECTORS[0], np.ones(len(run0.recipients))), 0.0)


def test_holm_adjusts_monotonically():
    import numpy as np
    from pdm_cmaps.experiments.poder_bootstrap import holm

    p = np.array([0.001, 0.04, 0.03, 0.5])
    adj = holm(p)
    assert np.allclose(adj, [0.004, 0.09, 0.09, 0.5])
    assert (adj >= p).all() and adj.max() <= 1.0
