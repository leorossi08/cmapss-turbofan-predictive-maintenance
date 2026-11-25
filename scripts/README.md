# scripts/

Execuções longas (horas) em Python, sem montes de shell: **um motor genérico** e **os planos como dados**.

```text
rodar.py    o motor: roda os passos de um plano, em sequência ou em raias paralelas, e cuida de tentativas, retomada e progresso
planos.py   os planos: o que rodar, em que ordem, como saber que deu certo e quanto deve demorar
```

```bash
python scripts/rodar.py lista                    # planos disponíveis, com descrição e duração aproximada
python scripts/rodar.py rodar tudo --dry-run     # mostra os comandos sem executar (confira antes de gastar horas)
python scripts/rodar.py rodar tudo               # executa; se for interrompido, rodar de novo retoma de onde parou
python scripts/rodar.py status tudo              # concluídos, em andamento, falhos
nohup python scripts/rodar.py rodar tudo > /dev/null 2>&1 &    # em segundo plano (acompanhe com `status`)
```

Opções de `rodar`: `--tentativas N` (padrão 3), `--refazer` (ignora o que já terminou), `--dry-run`. Pode ser chamado de qualquer pasta. Use o Python do ambiente virtual ativo.

## Como funciona

- **Passo** = comando + verificação do resultado + duração estimada. O código de saída 0 não basta: o passo só conta como feito se o arquivo esperado existe (ex.: "pelo menos 12 `*_pareado.csv`").
- **Retomada:** cada passo concluído deixa um marcador em `results/logs/<plano>/done/`; rodar de novo pula esses passos.
- **Tentativas:** um passo que falha é refeito até `--tentativas` vezes; se desistir, os que dependem dele são cancelados, os independentes seguem e o `final` não roda.
- **Raias:** passos de raias diferentes rodam em paralelo; `after=` faz um passo esperar outro de outra raia; o `pool` é uma fila de tarefas independentes que qualquer raia livre pega.
- **Threads:** cada processo usa 2 threads (`OMP_NUM_THREADS` etc.). Com várias raias, mais que isso deixa a máquina muito mais lenta (medido: mais de 7x).
- **Progresso:** `[12:03:11] [5/17] vs_FD002 ... ok (tentativa 1) 21min | decorrido 1h02min | restante ~48min`. A previsão usa a razão entre o tempo real e o estimado dos passos já concluídos, aplicada aos que faltam; é um guia, não uma promessa. Eventos também em `results/logs/<plano>/progresso.tsv`; o log de cada passo, em `results/logs/<plano>_<passo>.log`.

## Planos

| Plano | O que faz | Duração |
|---|---|---|
| `tudo` | pytest, pipeline dos 4 subconjuntos, todos os estudos (mesma lista do `update_all`) e sincronização das tabelas | horas |
| `rul_oficial` | teste oficial de RUL (12 modelos x 5 sementes), redes com regime e janela de 30 ciclos e boosting com o mesmo tratamento, agregado em `results/benchmark_oficial/` (4 raias) | horas |
| `hiperparametros` | busca de hiperparâmetros e janela (Optuna, validação cruzada por motor, só no treino; 10 modelos x 4 subconjuntos) e teste oficial ajustado (`--part tuned`, 3 sementes), em `results/hiperparametros/` e `results/benchmark_oficial_tuned/` | dezenas de horas |
| `diagnostico_fd003` | por que o Extra Trees fica abaixo da média do regime no FD003: mistura com a média, ET regularizado e janela de 50 | ~10 min |

**Registro histórico:** como cada conjunto de números do TCC foi regenerado. Não são necessários para usar o projeto; ficam como documentação executável.

| Plano | O que faz | Duração |
|---|---|---|
| `engine_offset` | reexecução completa depois de o deslocamento por motor virar padrão do sensor virtual (3 raias) | ~1,5 h |
| `multirregime` | refaz FD002 e FD004 com o deslocamento por regime (4 raias) | ~1,5 h |
| `varredura_janelas` | detecção a falso alarme igualado com H = 25, 35 e 50 + site | ~20 min |
| `offset_todos` | falhas sorteadas, decomposição por modelo, janela saudável e estudo nativo (3 raias) | ~1 h |
| `protocolo_principal` | camada de monitoramento (H = 50, conformal, custo) e injeção de RUL com a MEWMA corrigida | ~25 min |
| `ic_bootstrap` | matrizes de alarme da injeção e IC por bootstrap do poder | ~12 min |
| `injecao_h25`, `injecao_h35_50` | estudo de injeção com a janela H = 25 / varredura H = 35 e 50 | ~10 / ~30 min |
| `mewma_nativo`, `deteccao_fd004` | MEWMA nativa com e sem o deslocamento / injeção cega só no FD004 | ~17 / ~15 min |

Os planos que regeneram o site (`engine_offset`, `multirregime`, `varredura_janelas`) copiam os resultados para o repositório do dashboard (`fonte/atualizar_dados.py`) e rodam o `build_site.py`; esse repositório é procurado em `../cmapss-turbofan-predictive-maintenance-dashboard` (outro local: `DASHBOARD_DIR=/caminho`).

## Criar um plano novo

Em `planos.py`, escreva uma função que devolve um `Plan` e registre-a em `PLANOS`:

```python
def plano_meu() -> Plan:
    return Plan("meu", "descrição curta",
                lanes={"A": [passo("estudo", "pdm_cmaps.experiments.healthy_window_study", "--jobs", "2",
                                   est=1500, check=existe("results/healthy_window_study/tables/healthy_window_summary.csv"))]})
```

`python scripts/rodar.py rodar meu --dry-run` confere o resultado.

## Tabelas da monografia

`montar_tabelas.py` monta `tcc-writing/tables/` a partir de `results/`: copia as tabelas dos estudos e calcula as derivadas (testes por motor do RUL oficial, boosting por regime, largura do conformal na zona crítica, diagnóstico do FD003, hiperparâmetros escolhidos, ajustado × padrão, métricas estendidas, F1 dos alertas e métricas dos sensores virtuais). No fim tira as linhas do MiniRocket. Exige os `results/` da rodada completa (Colab); sem eles para com erro, sem mexer nas tabelas já versionadas. Depois rode `python -m pytest tests/test_invariantes_tabelas.py`.
