# PdM-CMaps

Manutenção preditiva em turbofans (**NASA C-MAPSS**): detecta anomalias sem usar o rótulo de falha, prevê a vida útil restante (RUL) e transforma a previsão em alertas de manutenção com incerteza calibrada. É o código do TCC.

> O site de painéis interativos fica em outro repositório: [`cmapss-turbofan-predictive-maintenance-dashboard`](https://github.com/leorossi08/cmapss-turbofan-predictive-maintenance-dashboard). Os resultados que ele apresenta são **copiados** daqui para `dados/` lá (`python fonte/atualizar_dados.py`, no repositório do site); o código gerador e as imagens do site moram só lá.

## Começando

Requer **Python 3.10+**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # `make install` faz o mesmo

make test                             # roda a suíte de testes
python -m pdm_cmaps --subset FD001    # primeira execução (um subconjunto)
```

A primeira execução não é rápida: medimos **~28 min** para FD001 sem MiniRocket (com a máquina também ocupada com outras tarefas), e o progresso aparece fase a fase (ver "Acompanhar uma rodada longa"). Os resultados vão para `results/maintenance_guide_fd001/` (`figures/` em PNG e `tables/` em CSV). O diretório `results/` é gerado e regenerável; não é versionado.

> **Atenção:** sem `--no-clean`, a pasta de saída é apagada no início da execução.

### Tudo em um notebook

[`notebooks/projeto_completo.ipynb`](notebooks/projeto_completo.ipynb) é **autocontido**: traz o código do pacote, os testes, os planos de execução, os dados públicos do C-MAPSS (embutidos) e os hiperparâmetros já escolhidos, com células para reproduzir resultados, tabelas e gráficos. Abra no Google Colab (CPU High-RAM, 8 núcleos) e rode da seção 1 à 4; as execuções longas ficam desligadas por padrão (chaves `EXECUTAR` na seção 5), e `SESSAO` divide o trabalho em várias máquinas. O padrão (testes, o pipeline do FD001 e a taxa de recuo) leva cerca de 10 a 15 minutos. O notebook é um instantâneo do código; ele é gerado no repositório da monografia e não deve ser editado à mão.

### MiniRocket

O MiniRocket (`modeling/minirocket.py`) continua no código, mas **fica desligado por padrão**: não entra no benchmark de RUL, na busca de hiperparâmetros nem no pipeline. `python -m pdm_cmaps --include-minirocket` o liga no pipeline.

## O que quero fazer → comando

Sempre da raiz do repositório, com o ambiente virtual ativo.

| Quero… | Comando | Saída |
|---|---|---|
| Rodar o pipeline em um subconjunto | `python -m pdm_cmaps --subset FD004` | `results/maintenance_guide_fd004/` |
| Rodar o pipeline nos 4 subconjuntos | `python -m pdm_cmaps --subset ALL --only-pipeline` | `results/maintenance_guide_fd00x/` |
| Rodar **tudo** (pipeline + todos os estudos + tabelas + gráficos) | `python scripts/rodar.py rodar tudo` (retomável) ou `python -m pdm_cmaps --all` | `results/` e `tcc-writing/tables/` (várias horas) |
| Só reexecutar os estudos e sincronizar as tabelas | `python -m pdm_cmaps.cli.update_all` (`--so-sincronizar` pula os estudos) | `results/`, `tcc-writing/tables/` |
| Camada de monitoramento (conformal, criticidade, custo) | `python -m pdm_cmaps.monitoring.layer --jobs 3` | `results/monitoring_repeated/` |
| Teste oficial de RUL com redes por regime e boosting | `python scripts/rodar.py rodar rul_oficial` | `results/benchmark_oficial/` (horas) |
| Reproduzir a busca e o teste ajustado com um comando (retomável, paralelo) | `pip install -e ".[workflow,tuning]"`; `snakemake --cores 8` (`-n` só mostra o plano; `--config usar_hiperparametros_prontos=1` pula a busca) | `results/hiperparametros/`, `results/benchmark_oficial_tuned/resumo_oficial.csv` |
| Estudos complementares (conformal com o modelo ajustado, ablação de exogeneidade, baselines multivariados, modos de falha) | `python -m pdm_cmaps.experiments.estudos_complementares --study all --jobs 4` | `results/estudos_complementares/`, `tcc-writing/tables/` |
| Teste da série saudável com bootstrap em blocos (diagnóstico e execução) | `python scripts/teste_bloco_injecao.py diag FD004 42` e `... run bloco FD004 42` | `results/`, `tcc-writing/tables/` |
| Rodar o teste oficial com os hiperparâmetros e janelas JÁ escolhidos (sem refazer a busca) | `python -m pdm_cmaps.experiments.benchmark_oficial_completo --part tuned --params-dir data/hiperparametros --subsets FD004` | `results/benchmark_oficial/` |
| Escolher hiperparâmetros e janela (Optuna) e rodar o teste oficial com eles | `pip install -e ".[tuning]"`; `python scripts/rodar.py rodar hiperparametros` (ou `python -m pdm_cmaps.experiments.selecao_hiperparametros --subsets FD004 --models XGBoost`) | `results/hiperparametros/` (`melhores.json` por subconjunto e modelo), `results/benchmark_oficial_tuned/` (dezenas de horas; retomável) |
| Benchmark de modelos de RUL (várias sementes) | `python -m pdm_cmaps.experiments.benchmark_repeated` (retoma se interrompido; `--aggregate-only` só reagrega) | `results/benchmark_repeated/` (~1h40) |
| Comparar a MEWMA com a EWMA clássica (com sinal) e um Isolation Forest, nas mesmas séries cegas | `python -m pdm_cmaps.experiments.fault_detection_study --extras --windows 25` | `results/injecao_deteccao_extras/` (poucos minutos com 8 núcleos) |
| Injeção cega de falhas e falso alarme igualado | `python -m pdm_cmaps.experiments.fault_detection_study` | `results/injecao_deteccao/` (rodada longa) |
| Falhas sorteadas (ganho do ML sobre a média do regime) | `python -m pdm_cmaps.experiments.fault_hard_study` | `results/falhas_dificeis/` (rodada longa) |
| Sensibilidade da janela saudável | `python -m pdm_cmaps.experiments.healthy_window_study` | `results/healthy_window_study/` |
| Gráficos interativos (Plotly) | `python -m pdm_cmaps.reporting.plotly_hub --open` (`--only t05 t12` para algumas figuras) | `results/plotly/index.html` |
| Rodar os testes | `make test` (ou `pytest`) | terminal |

Todo estudo aceita `--help`. Os longos aceitam `--jobs`, `--subsets` e `--output-dir`. A lista completa dos estudos e o que cada um responde está em [`src/pdm_cmaps/README.md`](src/pdm_cmaps/README.md).

### Acompanhar uma rodada longa

Estes comandos imprimem uma linha por passo com o que está rodando, o tempo decorrido e uma estimativa do que falta:

| Comando | Passos contados |
|---|---|
| `python -m pdm_cmaps` (um subconjunto) | as 5 fases do pipeline (4 se `--healthy-cycle-limit 100`) |
| `python -m pdm_cmaps --subset ALL` / `--all` | os 4 subconjuntos (e, no `--all`, depois cada estudo) |
| `python -m pdm_cmaps.cli.update_all` | cada estudo |
| `python scripts/rodar.py rodar <plano>` | cada passo do plano, com previsão a partir do tempo estimado de cada um |

```text
[14:04:38] [3/13] 03_monitoring ... iniciando | decorrido 1h12min | restante ~2h05min (média dos passos concluídos)
```

A estimativa é a média dos passos já concluídos vezes os que faltam; como os passos têm custos bem diferentes (as fases 3 e 4 pesam mais que as outras), é um guia aproximado, e só aparece depois do 1º passo. As linhas também ficam em arquivo (`tail -f` de outro terminal): `results/logs/_progresso.txt` nas rodadas do `pdm_cmaps`/`update_all`, e `results/logs/<plano>/progresso.tsv` nos planos do `rodar.py` (`python scripts/rodar.py status <plano>` resume o andamento). O log completo de cada estudo está em `results/logs/`.

**Sem estimativa:** os estudos rodados **avulsos** (`python -m pdm_cmaps.experiments.<nome>`, `python -m pdm_cmaps.monitoring.layer`) imprimem só o próprio log. Para acompanhá-los, rode-os por um plano do `scripts/rodar.py` (ver `python scripts/rodar.py lista`).

Opções do pipeline (`python -m pdm_cmaps --help`): `--subset {FD001..FD004,ALL}` (padrão `ALL`), `--seed` (42), `--healthy-cycle-limit` (50), `--unit-id`, `--no-minirocket`, `--include-lstm`, `--no-clean`, `--output-dir`, `--only-pipeline`.

## Como o código está organizado

```text
src/pdm_cmaps/            # pacote em camadas; cada pasta tem um docstring em __init__.py
  domain/                 #  1. regras puras, sem E/S: esquema dos dados, alvo de RUL, métricas
  data/                   #  2. leitura, divisão por motor (sem vazamento), atributos causais
  detection/              #  3. regimes, sensor virtual, cartas SPC (EWMA, envelope), MEWMA
  modeling/               #  4. modelos de RUL: árvores, redes, benchmark (o MiniRocket continua no código, desligado por padrão)
  monitoring/             #  5. intervalos conformais, criticidade, alertas, custo, relatórios
  analysis/               #  6. testes de hipótese e bootstrap
  pipeline/               #  7. orquestra tudo para um subconjunto (PipelineConfig → PdMPipeline)
  experiments/            #  8. estudos executáveis (`python -m pdm_cmaps.experiments.<nome>`)
  cli/                    #  9. linhas de comando (`pdm_cmaps`, `update_all`)
  reporting/              #     figuras; só desenha o que as outras camadas calcularam
  config.py               #     ProjectPaths: o único lugar que sabe onde ficam dados e saídas
tests/                    # um arquivo por camada (test_data.py, test_detection_spc.py, ...)
scripts/                  # rodar.py (motor) + planos.py (o que rodar): execuções longas, retomáveis, com progresso
data/CMaps/               # arquivos brutos do C-MAPSS (FD001–FD004)
tcc-writing/tables/       # criada ao rodar: CSVs-resumo dos estudos (sincronizados de results/); o nome vem da monografia, que fica em outro repositório
results/                  # gerado ao rodar; não versionado (a cópia dos resultados usados pelo site fica em dados/, no repositório do dashboard)
```

**Regra de dependência:** as setas apontam sempre para dentro, `domain` ← `data` ← `detection`/`modeling` ← `monitoring`/`analysis` ← `pipeline` ← `experiments`/`cli`. `reporting` só consome. Uma camada interna nunca importa uma externa.

**Por onde começar a ler:** `pipeline/pipeline.py` é a "receita" que chama todas as camadas na ordem; depois siga cada chamada até a camada correspondente. Cada módulo abre com um docstring que diz o que faz e por quê; leia o docstring e as assinaturas antes do corpo. O mapa de arquivos está em [`src/pdm_cmaps/README.md`](src/pdm_cmaps/README.md).

## O que o pipeline faz

1. **Regimes:** K-Means nas condições de voo (6 regimes em FD002/FD004, 1 em FD001/FD003), com escala ajustada só nos motores de treino.
2. **Sensor virtual:** linha de base condicionada ao regime para o T50 (`sensor_4`), calibrada na janela saudável; o resíduo é `real − esperado`.
3. **Detecção (SPC):** EWMA sobre o resíduo do T50 e MEWMA sobre 13 sensores, com persistência (`k = 2`) e limites calibrados em resíduos saudáveis *out-of-fold*.
4. **RUL:** compara XGBoost, LightGBM, CatBoost, florestas, SVM, MLP, CNN, LSTM e Transformer (o MiniRocket continua em `modeling/minirocket.py`, desligado por padrão: `--include-minirocket`).
5. **Estatística:** testes pareados, IC por *cluster bootstrap* BCa e correção de Benjamini-Hochberg.
6. **Monitoramento:** intervalos conformais para a RUL, níveis de criticidade, custo da decisão e mensagens de alerta geradas por *template* só a partir de fatos estruturados.

## Cuidados ao usar os resultados

- **Dois protocolos de avaliação — não misture.** `rul_holdout_benchmark.csv`: 25 % dos motores de treino, todos os ciclos; serve para comparar modelos entre si (otimista em valor absoluto). `rul_censored_test_benchmark.csv`: treino em todos os motores e só o **último ciclo** de cada motor de teste contra `RUL_FDxxx.txt` (limitado a 125); serve para comparar com a literatura.
- **Cite números multi-semente**, de `results/monitoring_repeated/` e `results/benchmark_repeated/`, não os de uma execução única. No benchmark, cite o maior valor-p entre as sementes (`p_per_seed_max`), não o p combinado por Fisher (as sementes reutilizam os mesmos motores).
- **Ao mudar divisão, calibração ou atributos**, preserve: divisão por motor, calibração só no treino, atributos causais. Há testes para isso em `tests/`.

## Mais documentação

- [`src/pdm_cmaps/README.md`](src/pdm_cmaps/README.md): mapa de módulos e dos estudos.
- [`tests/README.md`](tests/README.md): o que cada arquivo de teste cobre.
- [`scripts/README.md`](scripts/README.md): scripts de rodada longa e histórico de reexecuções.
- [`data/README.md`](data/README.md): proveniência e colunas do C-MAPSS.
