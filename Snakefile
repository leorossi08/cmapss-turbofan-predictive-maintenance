"""Fluxo reprodutível da busca de hiperparâmetros e do teste oficial ajustado (Snakemake).

    pip install -e ".[workflow,tuning]"           # snakemake + optuna
    snakemake --cores 8 -n                        # só mostra o que seria executado (dry-run)
    snakemake --cores 8                           # tudo: busca (44 estudos) -> teste oficial ajustado (12 tarefas) -> agregado
    snakemake --cores 8 --config usar_hiperparametros_prontos=1
                                                  # pula a busca e usa data/hiperparametros/ (os parâmetros escolhidos no TCC)
    snakemake --cores 2 --config subsets=FD001 trials=3   # teste rápido: um subconjunto, 3 tentativas por modelo

Cada regra declara o arquivo que produz; o Snakemake só refaz o que falta ou ficou desatualizado, então uma rodada interrompida
continua de onde parou (os estudos do Optuna também são retomáveis, pois ficam em SQLite). Cada tarefa usa 2 núcleos
(`threads: 2`), por isso `--cores 8` roda 4 tarefas em paralelo. A busca nunca lê o teste oficial; ele é lido só na regra `tuned`.

Conferido (Colab, FD001, semente 42, com `usar_hiperparametros_prontos=1`): as previsões reproduzem as do TCC; 10 dos 11 modelos saem idênticos
bit a bit e o MiniRocket difere em no máximo 3e-4 ciclo (ruído de ponto flutuante).
"""
import re

SUBSETS = config.get("subsets", "FD001,FD002,FD003,FD004").split(",")
SEEDS = [int(s) for s in str(config.get("seeds", "42,7,123")).split(",")]
TRIALS = config.get("trials")                      # None = orçamento padrão de cada modelo
PRONTOS = bool(int(config.get("usar_hiperparametros_prontos", 0)))
PARAMS = "data/hiperparametros" if PRONTOS else "results/hiperparametros"

# Mesma lista (e mesmo apelido de pasta) de pdm_cmaps.experiments.selecao_hiperparametros
MODELS = ["Ridge", "Random Forest", "XGBoost", "XGBoost (NASA Loss)", "LightGBM", "CatBoost", "MLP", "MiniRocket + RidgeCV",
          "PyTorch RNN (LSTM)", "PyTorch CNN", "PyTorch Transformer"]
slug = lambda name: re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
NOME = {slug(m): m for m in MODELS}
ENV = "OMP_NUM_THREADS=1 PDM_TORCH_THREADS=1"


rule all:
    input: "results/benchmark_oficial_tuned/resumo_oficial.csv"


rule busca:
    """Estudo Optuna de um (subconjunto, modelo): validação cruzada por motor, só nos motores de treino."""
    output: "results/hiperparametros/{fd}/{modelo}/melhores.json"
    wildcard_constraints: fd="FD00[1-4]", modelo="|".join(NOME)
    params: nome=lambda w: NOME[w.modelo], trials=lambda w: f"--trials {TRIALS}" if TRIALS else ""
    threads: 2
    log: "results/logs/snakemake/busca_{fd}_{modelo}.log"
    shell: '{ENV} python -m pdm_cmaps.experiments.selecao_hiperparametros --subsets {wildcards.fd} --models "{params.nome}" '
           '--n-jobs {threads} --output-dir results/hiperparametros {params.trials} > {log} 2>&1'


rule tuned:
    """Teste oficial com os parâmetros escolhidos: refaz o ajuste de cada modelo (na CPU) e prevê o último ciclo dos motores de teste."""
    input: [] if PRONTOS else expand("results/hiperparametros/{{fd}}/{modelo}/melhores.json", modelo=list(NOME))
    output: "results/benchmark_oficial_tuned/{fd}_s{seed}/previsoes_oficial.csv"
    wildcard_constraints: fd="FD00[1-4]", seed=r"\d+"
    params: saida=lambda w: f"results/benchmark_oficial_tuned/{w.fd}_s{w.seed}"
    threads: 2
    log: "results/logs/snakemake/tuned_{fd}_s{seed}.log"
    shell: '{ENV} python -m pdm_cmaps.experiments.benchmark_oficial_completo --subsets {wildcards.fd} --part tuned --seeds {wildcards.seed} '
           '--params-dir {PARAMS} --output-dir {params.saida} > {log} 2>&1'


rule agregar:
    """Junta as previsões por (subconjunto, semente) e calcula RMSE, NASA, F1 do estado crítico etc., com média e desvio entre sementes."""
    input: expand("results/benchmark_oficial_tuned/{fd}_s{seed}/previsoes_oficial.csv", fd=SUBSETS, seed=SEEDS)
    output: "results/benchmark_oficial_tuned/resumo_oficial.csv", "results/benchmark_oficial_tuned/metricas_por_semente.csv"
    log: "results/logs/snakemake/agregar.log"
    shell: "python -m pdm_cmaps.experiments.benchmark_oficial_completo --aggregate-only --output-dir results/benchmark_oficial_tuned > {log} 2>&1"
