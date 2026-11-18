# tests/

Um arquivo por camada do pacote, espelhando `src/pdm_cmaps/`. Rodar: `make test` ou `python -m pytest` (os testes marcados `slow` ficam de fora por padrão; configuração em `pyproject.toml`).

| Arquivo | Camada coberta | O que garante |
|---|---|---|
| `test_domain.py` | `domain/` | alvo de RUL limitado, métricas e o NASA score (penaliza mais o erro tardio), esquema dos dados |
| `test_data.py` | `data/` | leitura do C-MAPSS, divisão por motor **sem vazamento**, atributos temporais causais |
| `test_detection_regime.py` | `detection/regime.py` | regimes: ordenação dos centroides, margens, decomposição de variância |
| `test_detection_spc.py` | `detection/spc.py`, `multivariate.py`, `health_index.py` | persistência de alarmes (k ciclos), envelope por regime, MEWMA, índice de saúde |
| `test_virtual_sensor.py` | `detection/virtual_sensor*.py` | sensor virtual estritamente exógeno, R², deslocamento por motor (*engine offset*) |
| `test_modeling.py` | `modeling/` | pool de modelos, validação cruzada agrupada por motor, MiniRocket causal, redes PyTorch |
| `test_analysis.py` | `analysis/` | testes de hipótese, diagnósticos, importância de atributos |
| `test_monitoring.py` | `monitoring/` | cobertura conformal, criticidade, alertas causais, custo, mensagens só com números dos fatos |
| `test_experiments.py` | `experiments/` | injeção cega e pareada de falhas, alarmes causais, bootstrap de poder, janela saudável |
| `test_reporting.py` | `reporting/` | as figuras são geradas sem erro e seguem o estilo |
| `test_scripts.py` | `scripts/rodar.py` | tentativas, retomada, dependências entre raias, pool, cancelamento em falha, `--dry-run` e os planos reais bem formados |
| `test_pipeline_cli.py` | `pipeline/`, `cli/` | configuração padrão, argumentos da linha de comando, instanciação do pipeline, progresso com estimativa de tempo (`progress.py`) no `main()`, no `StudyRunner` e nas fases do pipeline |

`factories.py` guarda os construtores de dados sintéticos usados por mais de um arquivo (`sample_frame`, `make_synthetic_cmapss`, `_toy_timeline`); os usados por um arquivo só ficam no próprio arquivo. `conftest.py` registra os marcadores do pytest.

Nomes `test_todoN_*` vêm de uma lista de tarefas antiga; mantenha os nomes (há referências em notas).
