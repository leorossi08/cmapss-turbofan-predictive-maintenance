"""O protocolo primário congelado (src/pdm_cmaps/protocol.py) bate com os padrões do código e com as tabelas."""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pandas as pd
import pytest

from pdm_cmaps import protocol as P

SRC = Path(__file__).resolve().parent.parent / "src" / "pdm_cmaps"
STUDIES = sorted((SRC / "experiments").glob("*.py")) + [SRC / "monitoring" / "layer.py"]


def _argparse_defaults(path: Path, flag: str):
    """Valores `default=` literais de `add_argument("<flag>", ...)` no arquivo (via ast, sem importar nem rodar nada)."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            if isinstance(node.args[0], ast.Constant) and node.args[0].value == flag:
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            out.append(ast.literal_eval(kw.value))
                        except ValueError:
                            pass
    return out


def test_class_defaults_match_the_protocol():
    from pdm_cmaps.detection.multivariate import MultivariateResidualMonitor
    from pdm_cmaps.detection.virtual_sensor import VirtualSensor
    mv = MultivariateResidualMonitor()
    assert (mv.lam, mv.alpha, mv.k, mv.healthy_cycle_limit) == (P.MEWMA_LAMBDA, P.MEWMA_ALPHA, P.PERSISTENCE_K, P.H_NATIVE)
    assert mv.engine_offset is P.ENGINE_OFFSET and mv.calibration == P.MEWMA_CALIBRATION
    vs = VirtualSensor()
    assert (vs.ewma_alpha, vs.threshold_sigma, vs.healthy_cycle_limit) == (P.EWMA_ALPHA, P.THRESHOLD_SIGMA, P.H_NATIVE)
    assert vs.engine_offset is P.ENGINE_OFFSET


def test_monitoring_layer_uses_the_native_mewma_calibration():
    """A camada de monitoramento usa o limite sintético (emenda de 2026-11-18); o padrão da classe e a injeção seguem empíricos."""
    from pdm_cmaps.monitoring import layer
    assert P.MEWMA_CALIBRATION_NATIVE == "synthetic" and P.MEWMA_CALIBRATION == "empirical"
    assert layer.MEWMA_CALIBRATION_MONITORING == P.MEWMA_CALIBRATION_NATIVE
    assert inspect.signature(layer.compute_monitoring).parameters["mewma_calibration"].default == P.MEWMA_CALIBRATION_NATIVE
    assert inspect.signature(layer.run_one).parameters["mewma_calibration"].default == P.MEWMA_CALIBRATION_NATIVE
    src = (SRC / "monitoring" / "layer.py").read_text(encoding="utf-8")
    assert re.search(r'"--mewma-calibration".{0,120}default=MEWMA_CALIBRATION_MONITORING', src, re.S), "o padrão da CLI deve ser a constante do protocolo"
    assert re.search(r"MultivariateResidualMonitor\([^)]*calibration=mewma_calibration", src, re.S), "a camada deixou de passar a calibração ao MEWMA"
    for f in ("fault_detection_study.py", "fault_hard_study.py", "fault_model_study.py", "healthy_window_study.py", "fault_injection.py"):
        assert "calibration=" not in (SRC / "experiments" / f).read_text(encoding="utf-8"), f"{f}: a injeção deve manter o limite empírico (padrão da classe)"


def test_schema_constants_match_the_protocol():
    from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RUL_CAP
    assert DEFAULT_HEALTHY_CYCLE_LIMIT == P.H_NATIVE and DEFAULT_RUL_CAP == P.RUL_CAP


@pytest.mark.parametrize("path", STUDIES, ids=lambda p: p.name)
def test_every_study_uses_the_protocol_seeds(path):
    for d in _argparse_defaults(path, "--seeds"):
        if path.name in {"exploracao_r2.py", "mewma_calibracao_sintetica.py", "benchmark_oficial_completo.py"}:
            continue  # estudos de apoio declaram as próprias sementes (a exploração usa mais; o benchmark oficial usa 5 no `main` e 3 nas redes extras)
        expected = P.SEEDS_RUL if path.name == "benchmark_repeated.py" else P.SEEDS
        assert tuple(d) == expected, f"{path.name}: --seeds {d} != {expected}"


def test_study_cli_defaults_match_the_protocol():
    hard = SRC / "experiments" / "fault_hard_study.py"
    assert _argparse_defaults(hard, "--H") == [P.H_INJECTION]
    assert _argparse_defaults(hard, "--n-reps") == [P.N_REPS_HARD]
    for f in ("fault_detection_study.py", "fault_hard_study.py", "fault_model_study.py", "healthy_window_study.py"):
        assert _argparse_defaults(SRC / "experiments" / f, "--mewma-offset") in ([True], []), f
    assert _argparse_defaults(SRC / "experiments" / "fault_detection_study.py", "--engine-offset") == [P.ENGINE_OFFSET]


def test_persistence_and_split_constants_in_the_studies():
    from pdm_cmaps.experiments import envelope_ignored_cases, native_offset_fora_janela
    assert native_offset_fora_janela.K == P.PERSISTENCE_K and envelope_ignored_cases.PERSIST == P.PERSISTENCE_K
    assert native_offset_fora_janela.W_SPLIT == P.W_SPLIT and native_offset_fora_janela.H_NATIVE == P.H_NATIVE
    from pdm_cmaps.experiments.virtual_sensor_models import run_subset
    assert inspect.signature(run_subset).parameters["min_consecutive"].default == P.PERSISTENCE_K


def test_every_engine_split_uses_the_protocol_test_ratio():
    ratios = set()
    for path in SRC.rglob("*.py"):
        ratios |= set(re.findall(r"test_ratio\s*=\s*([0-9.]+)", path.read_text(encoding="utf-8")))
    ratios.discard("")
    assert {float(r) for r in ratios} <= {P.TEST_RATIO}, ratios


def test_rul_injection_keeps_the_mewma_without_offset_exception():
    src = (SRC / "experiments" / "fault_injection.py").read_text(encoding="utf-8")
    assert P.ENGINE_OFFSET_MEWMA_IN_RUL_INJECTION is False
    assert re.search(r"MultivariateResidualMonitor\([^)]*engine_offset=False", src, re.S), "a exceção da injeção de RUL deixou de existir no código"
