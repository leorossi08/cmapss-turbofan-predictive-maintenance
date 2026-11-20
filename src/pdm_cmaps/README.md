# src/pdm_cmaps: mapa dos módulos

Pacote Python em camadas. As dependências apontam para dentro: `domain` ← `data` ← `detection`/`modeling` ← `monitoring`/`analysis` ← `pipeline` ← `experiments`/`cli`; `reporting` só consome as camadas internas. Caminhos de dados e saídas ficam em `config.py` (`ProjectPaths`).

Os docstrings dos módulos estão em inglês; a tabela abaixo os resume. Cada subpacote também tem um docstring em `__init__.py` com a visão geral. **Para entender um módulo, leia o docstring e as assinaturas antes do corpo.** Os maiores (`reporting/plots.py`, `reporting/plotly_hub.py`, `pipeline/pipeline.py`, `analysis/stats.py`) convém ler por trechos.

**Roteiro de leitura sugerido:** `domain/` → `data/` → `detection/regime.py` → `detection/virtual_sensor.py` → `detection/spc.py` → `modeling/registry.py` → `monitoring/` (`uncertainty` → `alerts` → `reports` → `layer`) → `pipeline/pipeline.py` (a receita que junta tudo). `reporting/` pode ser ignorado até a hora de mudar uma figura.

Entradas de linha de comando: `python -m pdm_cmaps` (pipeline), `python -m pdm_cmaps.cli.update_all` (estudos extras e sincronização), `python -m pdm_cmaps.experiments.<estudo>` e `python -m pdm_cmaps.monitoring.layer`. Visão geral e quickstart no [README da raiz](../../README.md).


## Raiz do pacote: Configuração de caminhos e ponto de entrada

| Arquivo | O que faz |
|---|---|
| `__init__.py` | PdM-CMaps: predictive maintenance for NASA C-MAPSS turbofans (RUL, anomaly detection, monitoring layer) |
| `__main__.py` | `python -m pdm_cmaps` runs the command-line pipeline |
| `config.py` | Project paths: the single place that knows where data and outputs live |
| `progress.py` | Progress line per step with elapsed time and a rough ETA (used by the pipeline phases, `--all` and `update_all`) |

## domain: Regras puras, sem E/S (esquema dos dados, alvo e métricas de RUL)

| Arquivo | O que faz |
|---|---|
| `rul.py` | RUL targets and evaluation metrics (RMSE, MAE, R2, NASA score) |
| `schema.py` | Schema of the C-MAPSS files, sensor metadata and project-wide defaults |

## data: Leitura, divisão por motor e atributos causais

| Arquivo | O que faz |
|---|---|
| `features.py` | Causal temporal feature engineering (present and past of each engine only) |
| `loader.py` | Reading of the raw C-MAPSS text files |
| `splits.py` | Engine-level splits and grouped folds (no leakage between engines) |

## detection: Regimes, sensor virtual e detectores (EWMA, envelope, MEWMA)

| Arquivo | O que faz |
|---|---|
| `health_index.py` | Fusion of sensor residuals into a single health index |
| `multivariate.py` | Multivariate residual monitor (MEWMA) over 13 sensors |
| `regime.py` | Health monitoring: operating-regime decoupling, virtual sensors and SPC envelope anomaly detection |
| `spc.py` | Univariate statistical process control: EWMA/3-sigma and envelope detectors |
| `virtual_sensor.py` | Regime-conditioned virtual sensor and its residual; `engine_offset=True` (default) removes each engine's own level per (engine, regime) using only its healthy window; the MEWMA monitor does not use it |
| `virtual_sensor_metrics.py` | R² of the virtual sensor (holdout, healthy window, late window, engine baseline, model + offset) |

## modeling: Modelos de RUL: árvores, redes, MiniRocket, benchmark

| Arquivo | O que faz |
|---|---|
| `benchmark.py` | RULBenchmark: grouped-CV comparison of the RUL model pool |
| `gbdt.py` | XGBoost trained with the asymmetric NASA/PHM loss (objective + sklearn-style wrapper) |
| `minirocket.py` | MiniRocket convolutional feature transformer and RUL model |
| `networks.py` | PyTorch network architectures (nn.Module) |
| `registry.py` | Model factory/pool and grouped out-of-fold predictions |
| `sequences.py` | Sliding-window sequence builders for the neural models |
| `torch_models.py` | Estimator wrappers (fit/predict) around the PyTorch networks |

## monitoring: Intervalos conformais, criticidade, alertas, custo e relatórios

| Arquivo | O que faz |
|---|---|
| `alerts.py` | Criticality levels and alert generation |
| `layer.py` | Monitoring layer: RUL + uncertainty + criticality + alerts + grounded reports, and its figures |
| `reports.py` | Alert facts and deterministic report text for the maintenance engineer |
| `uncertainty.py` | Uncertainty for RUL predictions: split-conformal prediction intervals |

## analysis: Testes estatísticos e bootstrap

| Arquivo | O que faz |
|---|---|
| `stats.py` | Statistical hypothesis testing (Friedman, Wilcoxon, Ljung-Box, BH-FDR) and engine-cluster BCa bootstrap CIs |

## pipeline: Orquestração do pipeline por subconjunto

| Arquivo | O que faz |
|---|---|
| `config.py` | Strongly-typed pipeline configuration |
| `extra_models.py` | Supplementary models (SVM + PyTorch) and a uniform fit/predict helper |
| `pipeline.py` | Pipeline stages: virtual sensor / SPC / RUL benchmark, additional models and statistical inference |

## reporting: Figuras (matplotlib e Plotly)

| Arquivo | O que faz |
|---|---|
| `monitoring_figures.py` | Thesis figures of the monitoring layer (dataviz method; tokens and mark specs in viz_style.py) |
| `plotly_hub.py` | Interactive (Plotly) figures for the TCC, behind one clickable index page |
| `plots.py` | All figures: pipeline plots and thesis diagnostic figures |
| `viz_style.py` | Shared figure style for the thesis figures (tokens + helpers), following the dataviz method |

## experiments: Estudos executáveis (`python -m pdm_cmaps.experiments.<nome>`)

| Arquivo | O que faz |
|---|---|
| `benchmark_repeated.py` | Repeated RUL benchmark (RQ-A) over subsets x seeds, in both evaluation protocols |
| `benchmark_oficial_completo.py` | Official-test RUL with per-engine predictions (RMSE vs capped and uncapped RUL, NASA score), networks with per-regime normalization and a 30-cycle window, and boosting with the same treatment |
| `mewma_calibracao_sintetica.py` | Synthetic calibration of the MEWMA T² limit (option; validation on real healthy cycles) |
| `envelope_ignored_cases.py` | Small deviations that leave the envelope but do NOT become alerts: cases, numbers and figure for the TCC |
| `fault_detection_study.py` | Early-detection study with injected faults: power, delay and false alarms at MATCHED false-alarm rate |
| `fault_hard_study.py` | Hard, numerous, randomly drawn faults: does a learning model beat the regime mean as virtual sensor? |
| `fault_injection.py` | Blind fault-injection benchmark: do the detectors and RUL models react to a fault only WE know about? |
| `fault_model_study.py` | Which learning model should be the virtual sensor? Detection of injected faults per sensor and per estimator |
| `healthy_window_study.py` | Healthy-window sensitivity study: how many initial cycles should calibrate the SPC baseline? |
| `rul_prediction_plots.py` | Predicted vs real RUL for every model and every held-out engine, with maintenance markers |
| `virtual_sensor_models.py` | Virtual sensor for EVERY informative sensor, EVERY model and EVERY held-out engine (feeds the dashboards); `--engine-offset/--no-engine-offset`, `--recompute-r2` |
| `native_offset_fora_janela.py` | Native detection with the engine offset estimated INSIDE vs OUTSIDE the calibration window, with independent controls at the nominal limit |
| `poder_bootstrap.py` | Engine-cluster bootstrap CIs of the injection power, paired detector differences and Holm (`collect` / `analyse`) |
| `compare_engine_offset.py` | Paired power/delay and R² tables with and without the engine offset (`results/sensor_virtual_offset/`) |

## cli: Linha de comando e atualização de tudo

| Arquivo | O que faz |
|---|---|
| `main.py` | Command-line entry point: `python -m pdm_cmaps` |
| `update_all.py` | Reexecuta os estudos além do pipeline por dataset e sincroniza as tabelas da monografia e os gráficos Plotly |

## Classes e funções centrais

- `PdMPipeline` (`pipeline/pipeline.py`), `PipelineConfig` (`pipeline/config.py`): pipeline por subconjunto.
- `CMapssLoader` (`data/loader.py`), `ProjectPaths` (`config.py`).
- `RegimeDecoupler` (`detection/regime.py`), virtual sensor em `detection/virtual_sensor.py`, MEWMA em `detection/multivariate.py`.
- `RULBenchmark` (`modeling/benchmark.py`), pool de modelos em `modeling/registry.py`.
- `StudyRunner`, `TableSynchronizer` (`cli/update_all.py`).

## Cuidados

- Qualquer mudança em divisão, calibração ou atributos deve preservar: divisão por motor, calibração só no treino, atributos causais (há testes para isso em `tests/`).
- Os estudos longos aceitam `--jobs`, `--subsets` e `--output-dir`; os testes padrão não os executam (`-m "not slow"`).
