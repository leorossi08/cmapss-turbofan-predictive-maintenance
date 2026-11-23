"""Planos de execução para `rodar.py`: o que rodar, em que ordem, como saber que deu certo e quanto deve demorar.

Um plano é só dados (`Plan` com `Step`s). Para criar um novo, escreva uma função que devolve um `Plan` e registre em PLANOS.

Uso cotidiano:   tudo, rul_oficial, hiperparametros, diagnostico_fd003
Registro histórico (como cada conjunto de números do TCC foi regenerado; não são necessários para usar o projeto):
    engine_offset, varredura_janelas, multirregime, offset_todos, protocolo_principal,
    ic_bootstrap, injecao_h25, injecao_h35_50, mewma_nativo, deteccao_fd004
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from rodar import ROOT, Context, Plan, Step

PY = sys.executable
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# Repositório do site de painéis, irmão deste; os passos "site" só existem nos planos que o atualizam.
DASHBOARD = (ROOT / os.environ.get("DASHBOARD_DIR", "../cmapss-turbofan-predictive-maintenance-dashboard")).resolve()


# ── verificações de sucesso ───────────────────────────────────────────────────
# O comando pode sair com código 0 sem ter gerado o resultado; cada passo confere o que deveria existir.
def existe(*paths: str):
    return lambda ctx: all((ctx.root / p).exists() for p in paths)


def n_arquivos(padrao: str, minimo: int):
    from glob import glob
    return lambda ctx: len(glob(str(ctx.root / padrao))) >= minimo


def pipeline_ok(fd: str):
    """Tabela principal existe e alguma tabela é mais nova que o início do plano (não vale resultado de uma execução antiga)."""
    tables = f"results/maintenance_guide_{fd.lower()}/tables"
    return lambda ctx: (ctx.root / tables / "rul_holdout_predictions.csv").exists() and any(
        p.stat().st_mtime > ctx.started for p in (ctx.root / tables).glob("*.csv"))


def manifesto_t0(fd: str, t0: int = 35):
    """Injeção de falhas: todas as séries cegas devem ter a falha no ciclo t0 (H = 25 -> t0 = 35)."""
    def check(ctx: Context) -> bool:
        m = pd.read_csv(ctx.root / f"results/injecao_de_falhas/{fd}_seed42/selado/manifesto_deteccao.csv")
        return set(m.t0) == {t0}
    return check


def sensor_virtual_ok(fd: str, estrito: bool = False):
    def check(ctx: Context) -> bool:
        d = pd.read_csv(ctx.root / f"results/virtual_sensor_models/{fd}/model_scores.csv")
        ok = "r2_late_motor_base" in d and d["model"].nunique() == 7 and len(d) >= 7 * 14
        return ok and (not estrito or d["r2_holdout"].min() > -5)
    return check


def linhas_minimas(path: str, minimo: int):
    return lambda ctx: sum(1 for _ in open(ctx.root / path)) >= minimo


def passo(name: str, modulo: str, *args: str, est: int = 600, check=lambda ctx: True, clean: tuple[str, ...] = (), after: tuple[str, ...] = ()) -> Step:
    """Passo que roda `python -m modulo args...`."""
    return Step(name, (PY, "-m", modulo, *args), check, est, clean, after)


def passos_do_site(tabela_ok) -> list[Step]:
    """Final comum das reexecuções: sincroniza as tabelas, roda os testes, copia os resultados para o dashboard e regenera o site."""
    return [
        passo("sincronizacao", "pdm_cmaps.cli.update_all", "--so-sincronizar", est=400, check=tabela_ok),
        passo("pytest", "pytest", "-q", est=60),
        Step("dados_site", (PY, str(DASHBOARD / "fonte/atualizar_dados.py")), lambda c: (DASHBOARD / "dados/tables/pareado_far_resumo.csv").exists(), 120),
        Step("site", (PY, str(DASHBOARD / "fonte/build_site.py")), lambda c: (DASHBOARD / "site/sensores_FD003.html").exists(), 200),
    ]


def sensor_virtual(estimativas: dict[str, int], estrito: bool = False) -> list[Step]:
    return [passo(f"vs_{fd}", "pdm_cmaps.experiments.virtual_sensor_models", "--subset", fd, est=est, check=sensor_virtual_ok(fd, estrito))
            for fd, est in estimativas.items()]


def injecoes(estimativas: dict[str, int], *args: str) -> list[Step]:
    return [passo(f"injecao_{fd}", "pdm_cmaps.experiments.fault_injection", "--subset", fd, *args, est=est, check=manifesto_t0(fd))
            for fd, est in estimativas.items()]


# ── uso cotidiano ─────────────────────────────────────────────────────────────
def plano_tudo() -> Plan:
    """pytest, pipeline dos 4 subconjuntos, todos os estudos e a sincronização das tabelas, um passo por vez (equivale a `pdm_cmaps --all`, mas retomável)."""
    from pdm_cmaps.cli.update_all import STUDIES  # a lista de estudos fica em um lugar só

    estimativas = {"monitoring": 1500, "benchmark_repeated": 6000, "healthy_window": 1500, "fault_detection": 600, "fault_model_diag13": 1800,
                   "fault_model_diag24": 2400, "fault_hard": 5400, "fault_injection": 120, "rul_predicoes": 600, "envelope_cases": 120}
    pipeline = [passo(f"pipeline_{fd}", "pdm_cmaps", "--subset", fd, est=est, check=pipeline_ok(fd))
                for fd, est in {"FD001": 1100, "FD002": 1300, "FD003": 700, "FD004": 1300}.items()]
    estudos = [passo(s.name, s.module, *s.args, est=estimativas.get(s.name, 600)) for s in STUDIES]
    return Plan("tudo", "pytest + pipeline dos 4 subconjuntos + todos os estudos + sincronização das tabelas e gráficos (horas)",
                lanes={"A": [passo("pytest", "pytest", "-q", est=60), *pipeline, *estudos]},
                final=[passo("sincronizacao", "pdm_cmaps.cli.update_all", "--so-sincronizar", est=400)])


def plano_rul_oficial() -> Plan:
    """Teste oficial de RUL: 12 modelos x 5 sementes, redes com normalização por regime e janela de 30 ciclos (3 sementes) e boosting com o mesmo tratamento."""
    of = "pdm_cmaps.experiments.benchmark_oficial_completo"
    saida = lambda nome: f"results/benchmark_oficial/{nome}"  # noqa: E731
    pronto = lambda nome: existe(f"{saida(nome)}/previsoes_oficial.csv")  # noqa: E731
    principais = [passo(f"main_{fd}_s{seed}", of, "--subsets", fd, "--part", "main", "--seeds", str(seed), "--output-dir", saida(f"main_{fd}_s{seed}"),
                        est=1500, check=pronto(f"main_{fd}_s{seed}")) for fd in SUBSETS for seed in (42, 7, 123, 1, 2)]
    redes = [passo(f"extras_{fd}_s{seed}", of, "--subsets", fd, "--part", "extras", "--seeds", str(seed), "--output-dir", saida(f"{fd}_s{seed}"),
                   est=1800, check=pronto(f"{fd}_s{seed}")) for fd in SUBSETS for seed in (42, 7, 123)]
    boosting = [passo(f"boosting_{fd}", of, "--subsets", fd, "--part", "boosting", "--output-dir", saida(f"boost_{fd}"),
                      est=400, check=pronto(f"boost_{fd}")) for fd in SUBSETS]
    return Plan("rul_oficial", "teste oficial de RUL + redes com regime e janela 30 + boosting com o mesmo tratamento, agregado (horas, 4 raias)",
                lanes={"A": [], "B": [], "C": [], "D": []}, pool=[*redes, *principais, *boosting],
                final=[passo("agregar", of, "--aggregate-only", "--output-dir", "results/benchmark_oficial", est=60,
                             check=existe("results/benchmark_oficial/resumo_oficial.csv"))])


def plano_diagnostico_fd003() -> Plan:
    """Por que o Extra Trees fica abaixo da média do regime no FD003? Mistura com a média, ET regularizado e janela saudável de 50 (FD003 e FD001)."""
    hard = "pdm_cmaps.experiments.fault_hard_study"
    base = ("Média do regime (sem ML)",)
    e25 = ("Extra Trees (padrão)", "Mistura ET 25%", "Mistura ET 50%", "Extra Trees regularizado", *base)
    e50 = ("Extra Trees (padrão)", *base)
    por_subconjunto = lambda h: Step(f"por_subconjunto_{h}", (PY, "scripts/hard_by_subset.py", "--results", f"results/diagnostico_fd003/{h}",  # noqa: E731
                                                              "--out", f"results/diagnostico_fd003/por_subconjunto_{h}.csv"), existe(f"results/diagnostico_fd003/por_subconjunto_{h}.csv"), 60)
    return Plan("diagnostico_fd003", "diagnóstico do ganho negativo do Extra Trees no FD003 (~10 min)",
                lanes={"A": [passo("h25", hard, "--subsets", "FD003", "FD001", "--H", "25", "--jobs", "6", "--estimators", *e25, "--output-dir", "results/diagnostico_fd003/H25",
                                   est=300, check=n_arquivos("results/diagnostico_fd003/H25/runs/*_series.csv", 6)),
                       passo("h50", hard, "--subsets", "FD003", "FD001", "--H", "50", "--jobs", "6", "--estimators", *e50, "--output-dir", "results/diagnostico_fd003/H50",
                             est=300, check=n_arquivos("results/diagnostico_fd003/H50/runs/*_series.csv", 6)),
                       por_subconjunto("H25"), por_subconjunto("H50")]})



# ── registro histórico das reexecuções ────────────────────────────────────────
def plano_engine_offset() -> Plan:
    """Reexecução completa dos estudos afetados por VirtualSensor(engine_offset=True)."""
    limpar_runs = ("results/injecao_deteccao/runs",)  # senão os runs antigos são reaproveitados (cache por arquivo)
    return Plan(
        "engine_offset", "reexecução completa depois de o deslocamento por motor virar padrão do sensor virtual (~1,5 h, 3 raias)",
        lanes={
            "A": [*[passo(f"pipeline_{fd}", "pdm_cmaps", "--subset", fd, est=est, check=pipeline_ok(fd)) for fd, est in {"FD001": 1100, "FD002": 1300, "FD003": 700, "FD004": 1300}.items()],
                  passo("rul_predicoes", "pdm_cmaps.experiments.rul_prediction_plots", "--jobs", "2", est=600, check=existe("results/rul_predicoes")),
                  passo("envelope_cases", "pdm_cmaps.experiments.envelope_ignored_cases", "--results", "results", est=120)],
            "B": [passo("monitoring", "pdm_cmaps.monitoring.layer", "--jobs", "2", est=1500, check=existe("results/monitoring_repeated/tables")),
                  passo("healthy_window", "pdm_cmaps.experiments.healthy_window_study", "--jobs", "2", est=1500, check=existe("results/healthy_window_study/tables/healthy_window_summary.csv")),
                  passo("fault_detection", "pdm_cmaps.experiments.fault_detection_study", "--jobs", "2", est=600, check=n_arquivos("results/injecao_deteccao/runs/*_pareado.csv", 12), clean=limpar_runs),
                  *injecoes({"FD001": 30, "FD002": 100, "FD003": 30, "FD004": 100}, "--healthy-cycle-limit", "25")],
            "C": [],  # sem passos próprios: pega tarefas do pool
        },
        pool=sensor_virtual({"FD002": 2200, "FD004": 2200, "FD001": 1300, "FD003": 1320}),
        final=passos_do_site(existe("tcc-writing/tables/benchmark_repeated_summary.csv")))


def plano_varredura_janelas() -> Plan:
    return Plan(
        "varredura_janelas", "detecção a falso alarme igualado com as janelas H = 25, 35 e 50 + sincronização e site (~20 min)",
        lanes={"A": [passo("fault_detection", "pdm_cmaps.experiments.fault_detection_study", "--windows", "25", "35", "50", "--jobs", "2",
                           est=1100, check=n_arquivos("results/injecao_deteccao/runs/*_pareado.csv", 36))]},
        final=[x for x in passos_do_site(existe("tcc-writing/tables/pareado_far_resumo.csv")) if x.name != "pytest"])


def plano_multirregime() -> Plan:
    """Refaz FD002 e FD004 com o deslocamento POR REGIME (FD001 e FD003 têm regime único e não mudam)."""
    return Plan(
        "multirregime", "refaz FD002 e FD004 com o deslocamento do motor por regime (~1,5 h, 4 raias)",
        lanes={
            "A": [passo("pipeline_FD002", "pdm_cmaps", "--subset", "FD002", est=1800, check=pipeline_ok("FD002")),
                  passo("rul_predicoes", "pdm_cmaps.experiments.rul_prediction_plots", "--jobs", "2", est=600, check=existe("results/rul_predicoes"), after=("pipeline_FD004",)),
                  passo("envelope_cases", "pdm_cmaps.experiments.envelope_ignored_cases", "--results", "results", est=30)],
            "B": [passo("pipeline_FD004", "pdm_cmaps", "--subset", "FD004", est=1300, check=pipeline_ok("FD004")),
                  passo("fault_detection", "pdm_cmaps.experiments.fault_detection_study", "--subsets", "FD002", "FD004", "--windows", "25", "35", "50", "--jobs", "2",
                        est=700, check=n_arquivos("results/injecao_deteccao/runs/*_pareado.csv", 36), clean=("results/injecao_deteccao/runs/FD002_*", "results/injecao_deteccao/runs/FD004_*"))],
            "C": [passo("monitoring", "pdm_cmaps.monitoring.layer", "--subsets", "FD002", "FD004", "--jobs", "2", est=700,
                        check=n_arquivos("results/monitoring_repeated/runs/FD00[24]_seed*", 6), clean=("results/monitoring_repeated/runs/FD002_seed*", "results/monitoring_repeated/runs/FD004_seed*")),
                  passo("healthy_window", "pdm_cmaps.experiments.healthy_window_study", "--subsets", "FD002", "FD004", "--jobs", "2", est=900,
                        check=n_arquivos("results/healthy_window_study/runs/FD00[24]_seed*.csv", 6), clean=("results/healthy_window_study/runs/FD002_seed*.csv", "results/healthy_window_study/runs/FD004_seed*.csv")),
                  *injecoes({"FD002": 100, "FD004": 150}, "--healthy-cycle-limit", "25")],
            "D": [],
        },
        pool=sensor_virtual({"FD002": 1100, "FD004": 1100}, estrito=True),
        final=passos_do_site(lambda c: len(open(c.root / "tcc-writing/tables/pareado_far_resumo.csv").readlines()) >= 961
                             and len(open(c.root / "tcc-writing/tables/magnitude_minima_detectavel.csv").readlines()) >= 271))


def plano_offset_todos() -> Plan:
    """Estudos longos com o deslocamento em todos os detectores (hoje o padrão) e o falso alarme nativo com o deslocamento fora da janela."""
    return Plan(
        "offset_todos", "falhas sorteadas, decomposição por modelo, janela saudável e estudo nativo com o deslocamento (~1 h, 3 raias)",
        lanes={
            "A": [passo("fm_diag13", "pdm_cmaps.experiments.fault_model_study", "--H", "25", "--subsets", "FD001", "FD003", "--jobs", "2", "--output-dir", "results/injecao_modelos_diag",
                        est=1800, check=n_arquivos("results/injecao_modelos_diag/runs/*_pareado.csv", 6)),
                  passo("fm_diag24", "pdm_cmaps.experiments.fault_model_study", "--H", "25", "--subsets", "FD002", "FD004", "--jobs", "2", "--output-dir", "results/injecao_modelos_diag24",
                        est=2400, check=n_arquivos("results/injecao_modelos_diag24/runs/*_pareado.csv", 6))],
            "B": [passo("hard", "pdm_cmaps.experiments.fault_hard_study", "--H", "25", "--jobs", "2", est=5400, check=n_arquivos("results/falhas_dificeis/runs/*_series.csv", 9))],
            "C": [passo("healthy_window", "pdm_cmaps.experiments.healthy_window_study", "--jobs", "2", est=1500, check=n_arquivos("results/healthy_window_study/runs/*.csv", 12)),
                  passo("nativo_fora_janela", "pdm_cmaps.experiments.native_offset_fora_janela", "--jobs", "2", est=1200, check=n_arquivos("results/native_offset_fora_janela/runs/*.csv", 12))],
        })


def plano_protocolo_principal() -> Plan:
    return Plan(
        "protocolo_principal", "camada de monitoramento (H = 50, conformal, custo) e injeção de RUL com a MEWMA corrigida do deslocamento (~25 min)",
        lanes={"A": [passo("monitoramento", "pdm_cmaps.monitoring.layer", "--jobs", "2", est=1500, check=n_arquivos("results/monitoring_repeated/runs/*/uncertainty.csv", 12))],
               "B": injecoes({"FD001": 60, "FD002": 120, "FD003": 60, "FD004": 120}, "--healthy-cycle-limit", "25")})


def plano_ic_bootstrap() -> Plan:
    """Coleta as matrizes de alarme da injeção (H = 25, 3 sementes, 4 subconjuntos) e analisa o poder por bootstrap de motores."""
    coleta = [passo(f"coleta_{cfg}", "pdm_cmaps.experiments.poder_bootstrap", "collect", "--config", cfg, "--jobs", "2", est=360,
                    check=n_arquivos(f"results/injecao_deteccao_ic/{cfg}/alarmes/*.npz", 12)) for cfg in ("principal", "so_t50")]
    analise = [passo(f"analise_{cfg}", "pdm_cmaps.experiments.poder_bootstrap", "analyse", "--config", cfg, est=60) for cfg in ("principal", "so_t50")]
    return Plan("ic_bootstrap", "matrizes de alarme da injeção + IC por bootstrap do poder dos detectores (~12 min)", lanes={"A": [*coleta, *analise]})


def plano_injecao_h25() -> Plan:
    """Injeção a falso alarme igualado (H = 25, 4 subconjuntos, 3 sementes). O protocolo anterior (só T50) sai com --no-mewma-offset."""
    return Plan("injecao_h25", "estudo de injeção a falso alarme igualado, H = 25 (~10 min)",
                lanes={"A": [passo("fault_detection", "pdm_cmaps.experiments.fault_detection_study", "--windows", "25", "--jobs", "3", est=600,
                                   check=n_arquivos("results/injecao_deteccao/runs/*_pareado.csv", 12))]})


def plano_injecao_h35_50() -> Plan:
    return Plan("injecao_h35_50", "varredura da janela saudável H = 35 e 50 na injeção (o H = 25 já existe; ~30 min)",
                lanes={"A": [passo("fault_detection", "pdm_cmaps.experiments.fault_detection_study", "--windows", "35", "50", "--jobs", "2", est=1800,
                                   check=n_arquivos("results/injecao_deteccao/runs/*_pareado.csv", 36))]})


def plano_mewma_nativo() -> Plan:
    return Plan("mewma_nativo", "MEWMA nos motores nativos (H = 50) com e sem o deslocamento do motor (~17 min)",
                lanes={"A": [passo("native_mewma_offset", "pdm_cmaps.experiments.native_mewma_offset", est=1000,
                                   check=lambda c: (c.root / "results/native_mewma_offset/resumo.csv").stat().st_size > 0)]})


def plano_deteccao_fd004() -> Plan:
    return Plan("deteccao_fd004", "estudo de injeção cega só para o FD004 (H = 25, 3 sementes), usado nas figuras 5.12 e 5.13 (~10 a 20 min)",
                lanes={"A": [passo("fault_detection_FD004", "pdm_cmaps.experiments.fault_detection_study", "--subsets", "FD004", "--windows", "25", "--jobs", "3", est=900,
                                   check=n_arquivos("results/injecao_deteccao/runs/*", 9))]})


def plano_hiperparametros() -> Plan:
    """Escolha de hiperparâmetros e da janela por validação cruzada (Optuna, só nos motores de treino) e teste oficial com o conjunto escolhido.

    10 modelos x 4 subconjuntos, um estudo por passo (retomável: o banco SQLite guarda as tentativas). Depois, `--part tuned` refaz o ajuste final
    de cada modelo na CPU com os melhores parâmetros e lê o teste oficial uma única vez (3 sementes)."""
    from pdm_cmaps.experiments.selecao_hiperparametros import BUDGET, MODELS, slug
    sel, of, saida = "pdm_cmaps.experiments.selecao_hiperparametros", "pdm_cmaps.experiments.benchmark_oficial_completo", "results/hiperparametros"
    # segundos por tentativa medidos no Colab (8 núcleos); só orientam a previsão de tempo
    por_tentativa = {"Random Forest": 400, "XGBoost": 280, "XGBoost (NASA Loss)": 300, "CatBoost": 330, "LightGBM": 40, "MLP": 160, "Ridge": 30}
    estudos = [passo(f"busca_{fd}_{slug(m)}", sel, "--subsets", fd, "--models", m, "--output-dir", saida,
                     est=int(por_tentativa.get(m, 50) * BUDGET[m]), check=existe(f"{saida}/{fd}/{slug(m)}/melhores.json"))
               for m in sorted(MODELS, key=lambda m: -por_tentativa.get(m, 50) * BUDGET[m]) for fd in SUBSETS]
    tuned = [passo(f"tuned_{fd}_s{seed}", of, "--subsets", fd, "--part", "tuned", "--seeds", str(seed), "--params-dir", saida,
                   "--output-dir", f"results/benchmark_oficial_tuned/{fd}_s{seed}", est=1500,
                   check=existe(f"results/benchmark_oficial_tuned/{fd}_s{seed}/previsoes_oficial.csv")) for fd in SUBSETS for seed in (42, 7, 123)]
    return Plan("hiperparametros", "busca de hiperparâmetros e janela (Optuna, validação cruzada por motor) + teste oficial ajustado (muitas horas; 4 raias)",
                lanes={"A": [], "B": [], "C": [], "D": []}, pool=[*estudos, *tuned],
                final=[passo("agregar", of, "--aggregate-only", "--output-dir", "results/benchmark_oficial_tuned", est=60,
                             check=existe("results/benchmark_oficial_tuned/resumo_oficial.csv"))])


PLANOS = {
    "tudo": plano_tudo,
    "rul_oficial": plano_rul_oficial,
    "hiperparametros": plano_hiperparametros,
    "diagnostico_fd003": plano_diagnostico_fd003,
    "engine_offset": plano_engine_offset,
    "varredura_janelas": plano_varredura_janelas,
    "multirregime": plano_multirregime,
    "offset_todos": plano_offset_todos,
    "protocolo_principal": plano_protocolo_principal,
    "ic_bootstrap": plano_ic_bootstrap,
    "injecao_h25": plano_injecao_h25,
    "injecao_h35_50": plano_injecao_h35_50,
    "mewma_nativo": plano_mewma_nativo,
    "deteccao_fd004": plano_deteccao_fd004,
}
