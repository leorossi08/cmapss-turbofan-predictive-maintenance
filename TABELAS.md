# Tabelas da monografia → dados → comando

Cada linha é uma tabela do PDF da monografia (numeração do PDF de 24/11/2026). A coluna **CSV** diz de onde saem os números; salvo indicação, o arquivo fica em [`tcc-writing/tables/`](tcc-writing/tables/), que é versionado, então dá para conferir um número do texto sem reexecutar nada. A coluna **Como regenerar** diz qual estudo produz o CSV (ver também a tabela "O que quero fazer → comando" do [README](README.md)). As figuras têm o seu próprio mapa em [`figuras/MANIFESTO.md`](figuras/MANIFESTO.md).

Convenções usadas abaixo:

- `pareado_far_resumo.csv` e derivados: filtrar `H = 25` e `budget = 0.2` (falso alarme igualado a 0,2 por 100 ciclos).
- `rul_oficial_resumo.csv`: a coluna `Config` separa as configurações; `global_w15` é a referência das Tabelas 5.1 e B.2, e as redes "global, 15" da Tabela 5.2/B.4 estão em `ref_w15`.
- Custos (`monitoring_repeated_costs_summary.csv`): `Cost_ratio = 10`, `Lead = 0`; `Strategy` = `point` (previsão pontual), `interval` (limite inferior), `interval+anomaly` (+ T50), `interval+anomaly_mv` (+ MEWMA).
- **(texto)**: a tabela é descritiva (definições, parâmetros, revisão da literatura) e não sai de um CSV.

## Capítulos 2 a 4

| Tabela | Conteúdo | CSV | Como regenerar |
|---|---|---|---|
| 2.1 | Estações do motor e sensores | (texto) | — |
| 2.2 | Os 13 sensores: variação e deslocamento z do início ao fim de vida (FD001) | dados brutos `data/CMaps/train_FD001.txt` | sem script dedicado neste repositório |
| 2.3 | Exemplo numérico da MEWMA | (texto, conta à mão) | — |
| 3.1 | Trabalhos mais próximos | (texto) | — |
| 4.1 | Os quatro subconjuntos do C-MAPSS | dados brutos `data/CMaps/` (contagem de motores e linhas) | — |
| 4.2 | Parâmetros do protocolo principal | (texto) — os valores estão em [`src/pdm_cmaps/protocol.py`](src/pdm_cmaps/protocol.py) | — |

## Capítulo 5

| Tabela | Conteúdo | CSV | Como regenerar |
|---|---|---|---|
| 5.1 | Teste oficial no FD004 (RMSE, RMSE sem truncar, NASA) | `rul_oficial_resumo.csv` (linhas `global_w15`; LSTM por regime em `regime_w30`) + `rul_oficial_boosting_regime.csv` (XGBoost por regime, `regime_t30`) | `python scripts/rodar.py rodar rul_oficial` → `results/benchmark_oficial/resumo_oficial.csv` |
| 5.2 | Redes em quatro configurações (FD004) | `rul_oficial_resumo.csv` (redes: `ref_w15`, `global_w30`, `regime_w15`, `regime_w30`) + `rul_oficial_boosting_regime.csv` (XGBoost) | idem 5.1 |
| 5.3 | Teste oficial ajustado (FD004) | `rul_ajustado_vs_padrao.csv` (Ajustado, Referência, Anterior) + `hiperparametros_escolhidos.csv` (Escolha, Validação) | `snakemake --cores 8` (ou `python scripts/rodar.py rodar hiperparametros`) → `results/benchmark_oficial_tuned/` |
| 5.4 | Detecção nos motores nativos do FD004 | `monitoring_repeated_spc_summary.csv` (EWMA T50), `monitoring_repeated_spc_mv_summary.csv` (MEWMA), `native_mewma_offset_resumo.csv` (`engine_offset = False`: MEWMA sem correção), `native_offset_fora_janela_resumo.csv` (T50 sem correção) | `python -m pdm_cmaps.monitoring.layer`; `python -m pdm_cmaps.experiments.native_mewma_offset`; `python -m pdm_cmaps.experiments.native_offset_fora_janela` |
| 5.5 | Deslocamento estimado dentro e fora da janela (FD004) | `native_offset_fora_janela_resumo.csv`; a linha "limite sintético" usa a antecedência de `monitoring_repeated_spc_mv_summary.csv` e a calibração sintética (`results/mewma_calibracao_sintetica/`, não versionada) | `python -m pdm_cmaps.experiments.native_offset_fora_janela`; `python -m pdm_cmaps.experiments.mewma_calibracao_sintetica` |
| 5.6 | Poder em degraus da assinatura de fim de vida (FD004) | `pareado_far_resumo.csv` (`kind = degrau`, magnitudes 0,10 e 0,25) | `python -m pdm_cmaps.experiments.fault_detection_study` |
| 5.7 | MEWMA × EWMA do módulo × EWMA clássica × Isolation Forest (FD004) | `detectores_comparacao.csv` | `python -m pdm_cmaps.experiments.fault_detection_study --extras --windows 25` |
| 5.8 | Diferenças pareadas de poder, IC 95% e p de Holm (FD004) | `contrastes_ic_extras.csv` | `python -m pdm_cmaps.experiments.poder_bootstrap collect --config extras` e `... analyse --config extras` → `results/injecao_deteccao_ic/extras/contrastes_ic.csv` |
| 5.9 | Detectores multivariados com memória (FD004) | `detectores_memoria_comparacao.csv` (poder) + `contrastes_memoria_ic.csv` (contrastes) | `python -m pdm_cmaps.experiments.estudos_complementares --study c` |
| 5.10 | T50 com regime, sem e com a correção do deslocamento (FD004) | com a correção: `pareado_far_resumo.csv`; sem a correção: rodada `--no-engine-offset` do estudo de injeção (não versionada) | `python -m pdm_cmaps.experiments.fault_detection_study` (com e sem `--no-engine-offset`) |
| 5.11 | Decomposição do ganho: brutos × média do regime × Extra Trees (FD004) | `decomposicao_regime_modelo.csv` (`Detector = MEWMA (13 resíduos)`, `Estimator` = `Média do regime (sem ML)` / `Extra Trees (padrão)`) + `pareado_far_resumo.csv` (`MEWMA bruto (sem regime)`) | `python -m pdm_cmaps.experiments.fault_model_study --H 25 ...` (estudos `fault_model_diag13/24` do `update_all`) |
| 5.12 | Ablação da correção do deslocamento (FD004) | correção em todos: `poder_ic_principal.csv` + `contrastes_ic_principal.csv`; correção só no T50: `poder_ic_so_t50.csv` + `contrastes_ic_so_t50.csv` (contraste `regime na MEWMA (com - sem)`) | `python scripts/rodar.py rodar ic_bootstrap` → `results/injecao_deteccao_ic/{principal,so_t50}/` |
| 5.13 | Camada conformal e estabilidade de alerta (FD004) | `conformal_modelo_ajustado_resumo.csv` | `python -m pdm_cmaps.experiments.estudos_complementares --study a` |
| 5.14 | Custo relativo ao oráculo (FD004, LightGBM) | `monitoring_repeated_costs_summary.csv` (`critical<=10` / `critical<=20`) | `python -m pdm_cmaps.monitoring.layer --jobs 3` |

## Apêndice A

| Tabela | Conteúdo | CSV | Como regenerar |
|---|---|---|---|
| A.1 | Testes estatísticos usados (n e p dos três primeiros) | `thesis_hypothesis_tests.csv` | `python -m pdm_cmaps --subset FD004` (semente 42) |
| A.2 | Diagnóstico do ganho negativo do Extra Trees no FD003 | `diagnostico_fd003_ruido_estimacao.csv` | `python scripts/rodar.py rodar diagnostico_fd003` |
| A.3 | Espaço de busca dos hiperparâmetros | (texto) — definido em [`selecao_hiperparametros.py`](src/pdm_cmaps/experiments/selecao_hiperparametros.py) | — |
| A.4 | Hiperparâmetros escolhidos (FD004) | `hiperparametros_escolhidos.csv` (coluna `Parametros`); origem em [`data/hiperparametros/`](data/hiperparametros/) | `python scripts/rodar.py rodar hiperparametros` |
| A.5 | Taxa de recuo do deslocamento do motor | `taxa_fallback_deslocamento.csv` | `python scripts/taxa_fallback_deslocamento.py` |
| A.6 | MEWMA sob três modelos de série saudável (bootstrap em blocos) | `teste_bloco_ganho_regime.csv` (por semente: `teste_bloco_injecao_resumo.csv`) | `python scripts/teste_bloco_injecao.py run <iid\|bloco\|bloco_cal> FD004 <semente>` e depois `... resumir` |

## Apêndice B (versões completas, nos quatro subconjuntos)

| Tabela | Conteúdo | CSV | Como regenerar |
|---|---|---|---|
| B.1 | Variância do T50 explicada pelo regime e R² do sensor virtual | `regime_variance_by_subset.csv` + `sensor_virtual_r2_regime.csv` | `python -m pdm_cmaps.monitoring.layer`; `python -m pdm_cmaps.experiments.exploracao_r2_resumo` |
| B.2 | RMSE no teste oficial, cinco sementes | `rul_oficial_resumo.csv` (`global_w15`) | como 5.1 |
| B.3 | RMSE sem truncar e NASA por motor | `rul_oficial_resumo.csv` + `rul_oficial_boosting_regime.csv` | como 5.1 |
| B.4 | Redes em quatro configurações + XGBoost por regime | `rul_oficial_resumo.csv` + `rul_oficial_boosting_regime.csv` | como 5.1 |
| B.5 | Detecção nos motores nativos | como 5.4 | como 5.4 |
| B.6 | Deslocamento dentro e fora da janela | como 5.5 | como 5.5 |
| B.7 | Poder em degraus | `pareado_far_resumo.csv` | como 5.6 |
| B.8 | T50 com regime, sem e com a correção | como 5.10 | como 5.10 |
| B.9 | Decomposição do ganho | como 5.11 | como 5.11 |
| B.10 | Ablação da correção do deslocamento | como 5.12 | como 5.12 |
| B.11 | Magnitude mínima detectável | `magnitude_minima_detectavel.csv` | `python -m pdm_cmaps.experiments.fault_detection_study` |
| B.12 | Poder por forma de falha (degrau, rampa, viés) | `pareado_far_resumo.csv` (`kind` = `degrau`, `rampa`, `vies_sensor`) | como 5.6 |
| B.13 | Ganho do sensor virtual aprendido sobre a média do regime | `falhas_dificeis_por_subconjunto.csv` (`Mode = exogenous`) | `python -m pdm_cmaps.experiments.fault_hard_study` + `scripts/hard_by_subset.py` (feito pelo `update_all`) |
| B.14 | Intervalos conformais de 90% | `monitoring_repeated_uncertainty_summary.csv` + `conformal_largura_zona_critica.csv` (largura no fim de vida) | `python -m pdm_cmaps.monitoring.layer` |
| B.15 | Custo relativo ao oráculo | `monitoring_repeated_costs_summary.csv` | como 5.14 |
| B.16 | Teste oficial ajustado, quatro subconjuntos | `rul_ajustado_vs_padrao.csv` (ou `rul_ajustado_resumo.csv`) | como 5.3 |
| B.17 | Pontuação da NASA do teste ajustado | `rul_ajustado_vs_padrao.csv` (`Ajustado_NASA_medio`, `Padrao_NASA_medio`) | como 5.3 |
| B.18 | Janela, representação e RMSE da validação | `hiperparametros_escolhidos.csv` (resumo das janelas em `hiperparametros_janela_por_modelo.csv`) | como 5.3 |
| B.19 | Métricas além do RMSE (FD004) | `rul_ajustado_metricas_estendidas.csv` (colunas `_c15`) | como 5.3 |
| B.20 | Qualidade do sensor virtual por modelo | `sensor_virtual_metricas_resumo.csv` (por sensor: `sensor_virtual_metricas_por_sensor.csv`) | `python -m pdm_cmaps.experiments.virtual_sensor_models --subset FDxxx` |
| B.21 | Qualidade dos alertas de criticidade | `alertas_precisao_f1.csv` (`critical<=15`) | `python -m pdm_cmaps.monitoring.layer` |
| B.22 | Comparação de detectores, quatro subconjuntos | `detectores_comparacao.csv` | como 5.7 |

## CSVs que nenhum código grava com o nome versionado

Estes arquivos de `tcc-writing/tables/` não são escritos com esse nome por nenhum script nem pela sincronização (`python -m pdm_cmaps.cli.update_all --so-sincronizar`). Foram copiados ou montados a partir da saída indicada. Ao regenerar, é preciso copiá-los de novo:

| CSV | Saída de origem |
|---|---|
| `rul_oficial_resumo.csv`, `rul_oficial_metricas_por_semente.csv`, `rul_oficial_boosting_regime.csv`, `rul_oficial_testes_por_motor.csv` | `results/benchmark_oficial/` (`resumo_oficial.csv`, `metricas_por_semente.csv`) |
| `rul_ajustado_resumo.csv`, `rul_ajustado_metricas_por_semente.csv`, `rul_ajustado_metricas_estendidas.csv`, `rul_ajustado_vs_padrao.csv` | `results/benchmark_oficial_tuned/` |
| `hiperparametros_escolhidos.csv`, `hiperparametros_janela_por_modelo.csv` | `results/hiperparametros/<FD>/<modelo>/melhores.json` (cópia em `data/hiperparametros/`) |
| `poder_ic_principal.csv`, `poder_ic_so_t50.csv`, `poder_ic_extras.csv`, `contrastes_ic_principal.csv`, `contrastes_ic_so_t50.csv`, `contrastes_ic_extras.csv` | `results/injecao_deteccao_ic/<config>/{poder_ic,contrastes_ic}.csv` |
| `pareado_far_resumo_offset_so_t50.csv`, `regra_padrao_s1_offset_so_t50.csv`, `magnitude_minima_detectavel_offset_so_t50.csv` | estudo de injeção com `--no-mewma-offset` (sufixo `_offset_so_t50`) |
| `native_mewma_offset_resumo.csv`, `native_offset_fora_janela_resumo.csv` | `results/native_mewma_offset/resumo.csv`, `results/native_offset_fora_janela/resumo.csv` |
| `sensor_virtual_metricas_resumo.csv`, `sensor_virtual_metricas_por_sensor.csv` | `results/virtual_sensor_models/<FD>/model_scores.csv` |
| `conformal_largura_zona_critica.csv`, `alertas_precisao_f1.csv` | `results/monitoring_repeated/` |
| `teste_bloco_ganho_regime.csv`, `teste_bloco_autocorrelacao.csv` | `results/teste_bloco/` (a partir de `teste_bloco_injecao_resumo.csv` e do diagnóstico `diag`) |
| `diagnostico_fd003_ruido_estimacao.csv` | `results/diagnostico_fd003/por_subconjunto_{H25,H50}.csv` |
