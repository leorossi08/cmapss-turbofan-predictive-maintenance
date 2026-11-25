"""Invariantes das tabelas de resultados (tcc-writing/tables/ e results/virtual_sensor_models/).

POR QUE EXISTE: os testes unitários usam dados sintéticos e não veem o que uma rodada longa escreve nas tabelas. O bug do deslocamento em
multirregime (R² de −934 num sensor) só apareceu depois de uma reexecução de horas, olhando uma tabela. Estes testes leem os CSVs
já versionados (pequenos, sem rodar nenhum estudo) e falham se uma reexecução gravar algo impossível ou perder linhas.

REGRA: se um invariante falha nos dados atuais, o dado é que deve ser investigado; NÃO afrouxar o limite até passar. Os limites abaixo são
fisicamente motivados (porcentagem em [0, 100], p-valor em [0, 1], R² ≤ 1) ou um piso largo que pega erros de ordem de grandeza
(R² > −5). A contagem de linhas de alguns arquivos está em EXPECTED_ROWS: mudar de propósito (novo conjunto de configurações) exige
atualizar o número aqui, no mesmo commit.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "tcc-writing" / "tables"
VS = ROOT / "results" / "virtual_sensor_models"
TOL = 1e-9
SUBSETS = {"FD001", "FD002", "FD003", "FD004"}
SEEDS = {7, 42, 123}
VS_MODELS = {"Extra Trees", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "HistGradientBoosting", "Ridge"}

# colunas em que +inf é um valor LEGÍTIMO: ARL0 (comprimento médio até o primeiro falso alarme) é infinito quando nenhum controle
# disparou naquele limiar (dado censurado). Só +inf, e só aqui; qualquer outro inf (divisão por zero) continua sendo erro.
INF_ALLOWED = re.compile(r"^arl0($|_)")

# subconjuntos esperados quando NÃO são os quatro, por desenho do estudo: a versão anterior das falhas sorteadas (só T50 com deslocamento) rodou em 3 subconjuntos × 3 sementes; a atual inclui o FD003
# (9 execuções); o FD003 só entrou na versão atual (scripts reexecucao_offset_todos.sh e colab_falhas_fd003.sh, no repositório antigo PdM-CMaps).
EXPECTED_SUBSETS = {
    "falhas_dificeis_por_subconjunto_offset_so_t50.csv": {"FD001", "FD002", "FD004"},
    "diagnostico_fd003_ruido_estimacao.csv": {"FD001", "FD003"},  # diagnóstico só no FD003 e no FD001 (contraste)
    "taxa_fallback_deslocamento.csv": {"FD002", "FD004"},  # o recuo só existe com seis regimes
    "conformal_modelo_ajustado_resumo.csv": {"FD004"},  # foco da Seção 5.7/5.8 no subconjunto mais complexo
    "conformal_modelo_ajustado_custos.csv": {"FD004"},
    "assinaturas_modo_falha_eol.csv": {"FD003", "FD004"},  # subconjuntos com dois modos de falha (HPC e fan)
    "deteccao_nativa_por_modo.csv": {"FD003", "FD004"},
    "injecao_assinatura_por_modo.csv": {"FD003", "FD004"},
    "teste_bloco_injecao_por_semente.csv": {"FD004"},  # teste da série saudável em blocos só no FD004 (Apêndice A.6)
    "teste_bloco_injecao_resumo.csv": {"FD004"},
}

# número de linhas esperado em arquivos cuja grade é fixa (perder linhas = rodada incompleta)
EXPECTED_ROWS = {
    "contrastes_ic_principal.csv": 192,
    "contrastes_ic_so_t50.csv": 192,
    "benchmark_repeated_tests.csv": 40,  # 48 antes de tirar o MiniRocket (8 linhas)
    "thesis_hypothesis_tests.csv": 28,  # 29 antes de tirar o MiniRocket (1 linha)
    "margin_check_point_thresholds_raw.csv": 396,
}


def _csvs() -> list[Path]:
    files = sorted(TABLES.glob("*.csv"))
    if not files:
        pytest.skip("tcc-writing/tables/ sem CSVs")
    return files


def _read(f: Path) -> pd.DataFrame:
    return pd.read_csv(f)


def _numeric(d: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if c in d.columns and pd.api.types.is_numeric_dtype(d[c])]


def _pct_cols(d: pd.DataFrame) -> list[str]:
    """Colunas de porcentagem (…_pct, …_pct_mean, …_pct_std, pct_…), exceto contagens de linhas."""
    out = [c for c in d.columns if re.search(r"(^|_)pct($|_)|_pct_", c)]
    return _numeric(d, out)


def test_tables_have_no_inf_and_no_all_nan_columns():
    bad = []
    for f in _csvs():
        d = _read(f)
        num = d.select_dtypes("number")
        for c in num.columns:
            v = num[c].to_numpy(dtype=float)
            if INF_ALLOWED.match(c):
                if np.isneginf(v).any():
                    bad.append(f"{f.name}:{c} -inf")
            elif np.isinf(v).any():
                bad.append(f"{f.name}:{c} inf")
        for c in d.columns:
            if len(d) > 0 and d[c].isna().all():
                bad.append(f"{f.name}: coluna {c} toda NaN")
    assert not bad, bad


def test_percentage_columns_are_within_0_100():
    bad = []
    for f in _csvs():
        d = _read(f)
        for c in _pct_cols(d):
            lo, hi = d[c].min(), d[c].max()
            if lo < -TOL or hi > 100 + TOL:
                bad.append(f"{f.name}:{c} em [{lo:.3f}, {hi:.3f}]")
    assert not bad, bad


def test_p_values_and_coverage_are_probabilities():
    bad = []
    for f in _csvs():
        d = _read(f)
        cols = [c for c in d.columns if re.match(r"^p(_|$)|^p_value|_p$", c)] + [c for c in d.columns if re.search(r"coverage", c, re.I)]
        for c in _numeric(d, dict.fromkeys(cols)):
            if "std" in c.lower():
                continue
            lo, hi = d[c].min(), d[c].max()
            if lo < -TOL or hi > 1 + TOL:
                bad.append(f"{f.name}:{c} em [{lo:.4g}, {hi:.4g}]")
    assert not bad, bad


def test_r2_is_at_most_one_and_never_absurdly_negative():
    """R² ≤ 1 sempre; o piso de −5 pega erros de ordem de grandeza (o bug do multirregime dava −934), não R² negativo legítimo."""
    bad = []
    for f in _csvs():
        d = _read(f)
        for c in _numeric(d, [c for c in d.columns if re.search(r"r2", c, re.I) and "std" not in c.lower()]):
            lo, hi = d[c].min(), d[c].max()
            if hi > 1 + TOL or lo < -5:
                bad.append(f"{f.name}:{c} em [{lo:.3f}, {hi:.3f}]")
    assert not bad, bad


def test_std_columns_are_non_negative():
    bad = []
    for f in _csvs():
        d = _read(f)
        for c in _numeric(d, [c for c in d.columns if c.endswith("_std") or c == "std"]):
            if d[c].min() < -TOL:
                bad.append(f"{f.name}:{c} min {d[c].min():.4g}")
    assert not bad, bad


def test_adjusted_p_values_are_not_smaller_than_raw():
    """Holm e BH-FDR só podem aumentar o p-valor bruto."""
    pairs = [("p_holm", "p_boot"), ("p_value_BH_FDR", "p_value_raw")]
    seen = 0
    for f in _csvs():
        d = _read(f)
        for adj, raw in pairs:
            if adj in d.columns and raw in d.columns:
                seen += 1
                x = d[[adj, raw]].dropna()
                assert (x[adj] >= x[raw] - 1e-9).all(), f"{f.name}: {adj} < {raw} em {int((x[adj] < x[raw] - 1e-9).sum())} linhas"
    if not seen:
        pytest.skip("nenhuma tabela com p ajustado")


def test_subset_and_seed_sets_are_complete():
    bad = []
    for f in _csvs():
        d = _read(f)
        if "Subset" in d.columns and d["Subset"].astype(str).str.fullmatch(r"FD00[1-4]").all():
            got = set(d["Subset"])
            if f.name.startswith("sensor_virtual_exploracao"):
                continue  # só FD001 e FD003 foram explorados (declarado)
            if got != EXPECTED_SUBSETS.get(f.name, SUBSETS):
                bad.append(f"{f.name}: Subset {sorted(got)}")
        if "seed" in d.columns and set(d["seed"]) != SEEDS:
            bad.append(f"{f.name}: seed {sorted(set(d['seed']))}")
    assert not bad, bad


def test_expected_row_counts_of_fixed_grid_tables():
    bad = []
    for name, n in EXPECTED_ROWS.items():
        f = TABLES / name
        if not f.exists():
            bad.append(f"{name}: ausente")
            continue
        got = len(_read(f))
        if got != n:
            bad.append(f"{name}: {got} linhas, esperado {n}")
    assert not bad, bad


@pytest.mark.parametrize("subset", sorted(SUBSETS))
def test_virtual_sensor_model_scores_are_plausible(subset):
    f = VS / subset / "model_scores.csv"
    if not f.exists():
        pytest.skip(f"{f} ausente")
    d = pd.read_csv(f)
    assert set(d["model"]) == VS_MODELS
    per_sensor = d.groupby("sensor")["model"].nunique()
    assert (per_sensor == len(VS_MODELS)).all(), "sensor sem todos os modelos"
    for c in ("r2_holdout", "r2_healthy", "r2_late_model", "r2_late_within_regime", "r2_holdout_within_regime", "noise_ceiling_late"):
        assert c in d.columns, f"coluna {c} ausente (rodar --recompute-r2)"
        v = d[c].dropna()
        assert (v <= 1 + TOL).all(), f"{c} > 1"
        assert (v > -5).all(), f"{c} com valor absurdo: {v.min():.1f} (o bug do deslocamento em multirregime dava -934)"
    for c in ("detection_rate_pct", "mean_healthy_false_alarm_pct", "premature_alarm_pct"):
        v = d[c].dropna()
        assert ((v >= 0) & (v <= 100 + TOL)).all(), f"{c} fora de [0, 100]"
    assert (d["total_engines"] > 0).all()
