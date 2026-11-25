# Protocolo primário (congelado em 2026-11-18)

**O que é.** Uma foto do protocolo que gera os números principais da monografia, com as perguntas e as métricas primárias declaradas. **É um congelamento *post hoc*, feito depois de ver os resultados, e não um pré-registro**: várias decisões (o deslocamento do motor em todos os detectores, H = 25 nas injeções, as magnitudes destacadas) foram tomadas olhando os dados. O que o congelamento garante é que, daqui em diante, nenhum padrão muda sem que um teste falhe e sem que o autor decida.

**Onde está.** Os valores estão em `src/pdm_cmaps/protocol.py`. O `tests/test_protocolo.py` confere esses valores contra os padrões das classes e dos `argparse` do código e contra a tabela de contrastes (famílias do Holm). Mudar um valor é uma decisão do autor que invalida as tabelas do capítulo 5 e exige reexecutar os estudos afetados.

## Parâmetros congelados

| Parâmetro | Valor | Observação |
|---|---|---|
| Sementes | 42, 7, 123 | três divisões treino-teste por motor; **não** são réplicas independentes (mesmos motores) |
| Fração de teste | 0,25 | em toda divisão por motor |
| Janela saudável H | 50 (nativo e monitoramento), 25 (injeção) | H = 25 é padrão de trabalho, não o ótimo (Seção 5.4) |
| Deslocamento do motor | ligado, por motor e regime, mín. 3 linhas, em todos os detectores | exceção deliberada: a injeção para a RUL (apêndice) usa a MEWMA sem ele |
| EWMA do \|z\| | λ = 0,10, portão de 3σ | |
| MEWMA | λ = 0,10, α = 0,0027, persistência k = 2 | limite **sintético** nos motores nativos e na camada de monitoramento (emenda *post hoc* de 2026-11-18, validação só parcial); **empírico** (medido na janela) nos estudos de injeção |
| Persistência | 2 ciclos | o padrão da classe `VirtualSensor` é 3 e só vale sem argumento |
| Falhas sorteadas | 2 réplicas por motor | 10 008 falhas |
| Bootstrap | por motor receptor | limiar recalculado em cada réplica |
| Holm | 3 famílias: 16, 12 e 8 contrastes | fora delas, p sem correção (declarado) |

## Perguntas e métricas primárias

- **P1, injeção cega.** Poder de detecção a **falso alarme igualado**, por forma e magnitude da falha, com IC por bootstrap por motor e Holm nas três famílias: o efeito do regime na MEWMA, T50 contra MEWMA e T50 contra T50 bruto. É a métrica que decide as conclusões sobre o regime.
- **P2, motores nativos e camada de monitoramento.** Taxa de motores detectados, antecedência e falso alarme. **A antecedência é descritiva, não é medida de verdade-terreno**: a RUL ≥ 125 não é um rótulo de saúde, então degradação inicial real e falso alarme se misturam depois da janela. O falso alarme nominal é medido em controles independentes e, nos tratamentos de janela dividida, em ciclos reais saudáveis 26 a 50. Incluem-se a cobertura conformal e o custo da decisão.
- **P3, RUL.** RMSE e score da NASA nos protocolos holdout e oficial, **média ± desvio sobre 5 sementes (42, 7, 123, 1, 2; `protocol.SEEDS_RUL`), com as redes e o MiniRocket determinísticos** (`modeling/determinism.py`: mesma semente e mesmos dados dão as mesmas previsões com 1 thread, qualquer que seja o `OMP_NUM_THREADS`). No protocolo oficial o desvio entre sementes é variância pura de treino, porque o conjunto de teste é o mesmo. Testes pareados por motor, com o erro de cada motor promediado sobre as sementes (um ponto por motor), e não combinando p-valores de sementes como se fossem independentes.

## Rótulo de cada tabela publicada

`confirmatório` = responde a P1, P2 ou P3 no protocolo primário. `comparação` = protocolo anterior, usado como contraste. `exploratório` = análise de apoio ou varredura, sem valor confirmatório. `descritivo` = caracteriza os dados. O `tests/test_protocolo.py` falha se uma tabela nova entrar em `tcc-writing/tables/` sem constar aqui.

| Tabela | Rótulo | Pergunta |
|---|---|---|
| `benchmark_repeated_summary.csv` | confirmatório | P3: benchmark de RUL e testes de hipótese (a refazer com as redes determinísticas) |
| `rul_oficial_resumo.csv` | confirmatório | P3: teste oficial de RUL no Colab (13 modelos, 5 sementes; redes com janela de 30 e normalização por regime, 3 sementes): RMSE contra a verdade truncada e sem truncar, MAE e pontuação da NASA (Tabelas de RUL do Cap. 5) |
| `rul_oficial_metricas_por_semente.csv` | confirmatório | P3: as mesmas métricas por semente |
| `rul_oficial_testes_por_motor.csv` | confirmatório | P3: Wilcoxon pareado por motor, XGBoost (+temporal) contra as redes em cada configuração |
| `conformal_largura_zona_critica.csv` | exploratório | largura média do intervalo conformal (Mondrian padrão) no total e na zona crítica (RUL real < 50 ciclos), média e desvio de três sementes, agregada de `results/monitoring_repeated/runs/*/uncertainty_variants.csv` |
| `rul_oficial_boosting_regime.csv` | confirmatório | P3: XGBoost e LightGBM (+temporal) com atributos sobre sensores padronizados por regime, janelas 5/10 e 5/10/30 (Colab, determinístico): RMSE truncado e sem truncar e pontuação da NASA |
| `rul_ajustado_resumo.csv` | exploratório (decidido antes de ver o teste: hiperparâmetros e janela saem só da validação cruzada nos motores de treino) | P3: teste oficial ajustado, 11 modelos x 4 subconjuntos x 3 sementes (média e desvio), com as métricas estendidas; Tabela 5.3 e Apêndice B |
| `rul_ajustado_metricas_por_semente.csv` | exploratório | O mesmo, por semente |
| `rul_ajustado_vs_padrao.csv` | exploratório | RMSE (truncado e sem truncar) e NASA médio do ajustado contra a referência (janela 15, global) e contra a melhor configuração anterior (escolhida com o teste à vista); Tabela 5.3 e B.16/B.17 |
| `hiperparametros_escolhidos.csv` | exploratório | Janela, representação, RMSE e erro-padrão da validação cruzada, tentativas e parâmetros de cada estudo (44); Tabela B.18 |
| `hiperparametros_janela_por_modelo.csv` | exploratório | Janela escolhida por modelo e quantas vezes bateu no limite de 60 ciclos |
| `rul_ajustado_metricas_estendidas.csv` | exploratório | Viés, erro mediano, P90, acerto +-10/+-20 e precisão, revocação, F1 e PR-AUC do estado crítico (c = 10, 15, 20); Tabela B.19 |
| `sensor_virtual_metricas_por_sensor.csv` | exploratório | MAE, NRMSE dentro do regime e ganho sobre a média do regime por sensor e modelo, recalculados dos traces |
| `sensor_virtual_metricas_resumo.csv` | exploratório | Mediana dos sensores por subconjunto e modelo; Tabela B.20 |
| `alertas_precisao_f1.csv` | exploratório | Precisão, revocação e F1 linha a linha de cada estratégia de criticidade, por política e subconjunto; Tabela B.21 |
| `taxa_fallback_deslocamento.csv` | exploratório | Fração de pares (motor, regime) com menos de três ciclos nos primeiros H ciclos (recuo `o_{e,r} = 0`), H = 25 e 50, FD002 e FD004 (`scripts/taxa_fallback_deslocamento.py`) |
| `conformal_modelo_ajustado_resumo.csv` | exploratório | Camada conformal e oscilações de criticidade no FD004 sobre LightGBM (ref.), XGBoost ajustado e CatBoost ajustado: Mondrian (3 tercis), CQR contínuo e histerese (banda de 5 ciclos) (`experiments/estudos_complementares.py`, Exp. A) |
| `conformal_modelo_ajustado_custos.csv` | exploratório | Custo relativo ao oráculo por razão de custo ρ e prazo de logística no FD004 cruzando os modelos ajustados com Mondrian e CQR (`experiments/estudos_complementares.py`, Exp. A) |
| `ablativo_exogeneidade_r2.csv` | exploratório | Ablação de exogeneidade estrita removendo P15 e epr das entradas do sensor virtual nos quatro subconjuntos: R² dentro do regime e fração do teto nos ciclos 26 a 50 (`experiments/estudos_complementares.py`, Exp. B) |
| `ablativo_exogeneidade_poder.csv` | exploratório | Poder de detecção da MEWMA a falso alarme igualado (0,2 por 100 ciclos, H = 25) com entradas canônicas vs. estritamente externas sem P15/epr (`experiments/estudos_complementares.py`, Exp. B) |
| `detectores_memoria_comparacao.csv` | exploratório | Comparação entre MEWMA, PCA-T²/SPE com memória EWMA, Isolation Forest com memória EWMA e Isolation Forest pontual nas mesmas séries cegas a falso alarme igualado (`experiments/estudos_complementares.py`, Exp. C) |
| `contrastes_memoria_ic.csv` | exploratório | Contrastes pareados com bootstrap em cluster (4000 réplicas) e correção de Holm (famílias de 12 comparações) entre MEWMA e os baselines com memória temporal (`experiments/estudos_complementares.py`, Exp. C) |
| `assinaturas_modo_falha_eol.csv` | exploratório | Assinaturas de fim de vida padronizadas por regime e ajustadas no FD003 e FD004 separadas por modo (HPC vs. fan via KMeans K=2) (`experiments/estudos_complementares.py`, Exp. D) |
| `deteccao_nativa_por_modo.csv` | exploratório | Desempenho da detecção nativa (T50 EWMA e MEWMA com limite sintético, H = 50) no holdout do FD003 e FD004 estratificado pelo modo de falha aprendido no treino (`experiments/estudos_complementares.py`, Exp. D) |
| `injecao_assinatura_por_modo.csv` | exploratório | Poder de detecção em degraus (0,05, 0,10 e 0,25) injetando assinatura média, só HPC ou só fan aprendidas via KMeans K=2 no treino do FD003 e FD004 (`experiments/estudos_complementares.py`, Exp. D) |
| `detectores_comparacao.csv` | exploratório (acrescentado depois da banca; mesmas séries cegas e mesmo orçamento) | Poder e atraso da MEWMA e do T50 (protocolo principal) contra a EWMA clássica com sinal e o Isolation Forest sobre os 13 resíduos, H = 25, 0,2 falsos alarmes por 100 ciclos (`fault_detection_study --extras`) |
| `contrastes_ic_extras.csv` | exploratório (acrescentado depois da banca) | Diferenças pareadas de poder com IC 95% por bootstrap sobre os motores e valor-p de Holm (famílias de 12, 12 e 16 comparações): MEWMA contra EWMA clássica com sinal e contra Isolation Forest; EWMA clássica contra EWMA do módulo (`poder_bootstrap --config extras`) |
| `poder_ic_extras.csv` | exploratório | Poder de cada detector (incluindo os dois extras) com IC 95% por bootstrap, mesma configuração |
| `benchmark_repeated_tests.csv` | comparação | combinação de p-valores das sementes pelo método de Fisher (as sementes não são independentes: o método infla a evidência); mantida só por continuidade, não usar |
| `benchmark_repeated_tests_por_motor.csv` | confirmatório | P3: teste pareado por motor, erro de cada motor promediado sobre as sementes (um ponto por motor) |
| `contrastes_ic_principal.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `contrastes_ic_so_t50.csv` | comparação | protocolo anterior (deslocamento só no T50) ou sem deslocamento; serve de contraste, não de resultado |
| `decomposicao_regime_modelo.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `diferenca_vs_media_regime.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `envelope_desvios_casos.csv` | exploratório | análise do envelope e das margens (apoio) |
| `envelope_desvios_resumo.csv` | exploratório | análise do envelope e das margens (apoio) |
| `falhas_dificeis_por_subconjunto.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `healthy_window_summary.csv` | exploratório | varredura da janela saudável H (25, 35, 50, 100) |
| `magnitude_minima_detectavel.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `magnitude_minima_detectavel_offset_so_t50.csv` | comparação | protocolo anterior (deslocamento só no T50) ou sem deslocamento; serve de contraste, não de resultado |
| `margin_check_point_thresholds_raw.csv` | exploratório | análise do envelope e das margens (apoio) |
| `monitoring_repeated_costs_summary.csv` | confirmatório | P2: incerteza conformal e custo da decisão (camada de monitoramento) |
| `monitoring_repeated_spc_mv_summary.csv` | confirmatório | P2: detecção nos motores nativos (taxa, antecedência descritiva, falso alarme) |
| `monitoring_repeated_spc_summary.csv` | confirmatório | P2: detecção nos motores nativos (T50 EWMA) |
| `monitoring_repeated_summary.csv` | confirmatório | P2: detecção nos motores nativos (taxa, antecedência descritiva, falso alarme) |
| `monitoring_repeated_uncertainty_summary.csv` | confirmatório | P2: incerteza conformal e custo da decisão (camada de monitoramento) |
| `monitoring_repeated_uncertainty_variants_summary.csv` | confirmatório | P2: incerteza conformal e custo da decisão (camada de monitoramento) |
| `native_mewma_offset_resumo.csv` | exploratório | análise de apoio dos motores nativos (deslocamento fora da janela, MEWMA) |
| `diagnostico_fd003_ruido_estimacao.csv` | exploratório | diagnóstico do ganho negativo do Extra Trees no FD003: ET padrão, misturas com a média (25% e 50%) e regularizado, H=25 e H=50, FD003 e FD001 (`python scripts/rodar.py rodar diagnostico_fd003`) |
| `native_offset_fora_janela_resumo.csv` | exploratório | análise de apoio dos motores nativos (deslocamento fora da janela, MEWMA) |
| `pareado_far_resumo.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `pareado_far_resumo_offset_so_t50.csv` | comparação | protocolo anterior (deslocamento só no T50) ou sem deslocamento; serve de contraste, não de resultado |
| `poder_ic_principal.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `poder_ic_so_t50.csv` | comparação | protocolo anterior (deslocamento só no T50) ou sem deslocamento; serve de contraste, não de resultado |
| `poder_por_forma.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `poder_por_magnitude.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `regime_variance_by_subset.csv` | descritivo | variância do T50 explicada pelo regime |
| `regra_padrao_s1.csv` | confirmatório | P1: poder da injeção cega a falso alarme igualado |
| `regra_padrao_s1_offset_so_t50.csv` | comparação | protocolo anterior (deslocamento só no T50) ou sem deslocamento; serve de contraste, não de resultado |
| `sensor_virtual_r2_regime.csv` | exploratório | exploração do R² do sensor virtual (teto de ruído e trilhas A, B1, B1s e B2) |
| `thesis_hypothesis_tests.csv` | confirmatório | P3: benchmark de RUL e testes de hipótese (a refazer com as redes determinísticas) |
| `teste_bloco_autocorrelacao.csv` | exploratório | Autocorrelação do T50 padronizado nos ciclos saudáveis reais, i.i.d. e em blocos (`scripts/teste_bloco_injecao.py`; só FD004) |
| `teste_bloco_ganho_regime.csv` | exploratório | Ganho de remover o regime em cada modelo de série saudável (i.i.d., misto, blocos); Apêndice A.6 (`scripts/teste_bloco_injecao.py`; só FD004) |
| `teste_bloco_injecao_por_semente.csv` | exploratório | Poder da injeção por semente com a série saudável em blocos (`scripts/teste_bloco_injecao.py`; só FD004) |
| `teste_bloco_injecao_resumo.csv` | exploratório | Resumo do teste da série saudável com bootstrap em blocos; Tabela A.6 (`scripts/teste_bloco_injecao.py`; só FD004) |

## Pendências do congelamento

- **Calibração do limite da MEWMA (adotada em 2026-11-18, por decisão do autor, depois de ver os resultados).** O limite empírico subestima o T² (medido na janela, onde a média do z é nula) e o falso alarme nominal fica 0,20 a 2,64 por 100 ciclos contra 0,01 a 0,06 sem o deslocamento. A opção `calibration="synthetic"` reproduz o erro de estimar o deslocamento (`experiments/mewma_calibracao_sintetica.py`) e passou a valer **só nos motores nativos e na camada de monitoramento**, como argumento explícito de `monitoring/layer.py` (`MEWMA_CALIBRATION_MONITORING`; a CLI aceita `--mewma-calibration empirical` para reproduzir a rodada anterior, arquivada em `results/monitoring_repeated/runs_limite_empirico` e nas tabelas `*_limite_empirico.csv`). O padrão da classe e os estudos de injeção (Seção 5.5 e a parte de injeção da Seção 5.4) seguem com o limite empírico: mudar isso invalidaria tabelas que não foram reexecutadas. **A validação foi só parcial** (em α = 0,01 reprova nos subconjuntos mais simples; em α = 0,0027 a razão observado/nominal vai de 0 a 1,58): o limite sintético é uma correção adotada, não um limite validado. O `tests/test_protocolo.py` confere que a camada usa a constante e que a injeção não a usa.
- **Por que a injeção fica com o limite empírico (piloto de 2026-11-19).** Rodei `fault_detection_study` pequeno (FD001, semente 42, H = 25, 14 motores, 2 controles, 1 réplica) com `calibration="empirical"` e com `"synthetic"` (padrão da classe trocado só no piloto). Os quatro detectores que não dependem do limite do T² (T50 com e sem portão, T50 bruto, MEWMA bruta) saíram **bit a bit idênticos** (nenhum acoplamento de semente). Na MEWMA com regime, o falso alarme igualado é o mesmo (0,146 por 100 ciclos nos dois), o `s_star` só muda de escala (1,80 contra 0,71, porque a grade de limiares é em unidades do limite nominal) e a potência difere em **um motor** (7,1 pontos) em 2 das 14 células, por escolha de ponto em um patamar da grade `S_GRID`. Conclusão: o limite nominal não decide a comparação a falso alarme igualado; trocá-lo só adicionaria ruído de discretização à Tabela 5.9 e custaria horas de reexecução. Dependem do limite nominal (e portanto seguem empíricos, rotulados assim) só as saídas `regra_padrao_s1.csv` (regra padrão, usada apenas nos painéis) e a varredura de janelas nos motores nativos (`healthy_window_summary.csv`, Seção 5.4, rotulada "limite empírico" no texto).
- **Benchmark de RUL (feito em 2026-11-18).** Rodada completa com as redes determinísticas, 5 sementes, 520 linhas em `results/benchmark_repeated/`. Os 17 modelos não neurais saíram idênticos bit a bit à rodada anterior (|ΔRMSE| = 0 nas 72 linhas agregadas), o que confirma que só as redes e o MiniRocket mudaram. Ficam por refazer as linhas H4 e H5 de `thesis_hypothesis_tests.csv`, que vêm do pipeline de um subconjunto (não do benchmark repetido).

## Busca de hiperparâmetros (2026-11-21)

Fora do congelamento acima (que descreve a detecção), a RUL tem um protocolo próprio: Optuna TPE (semente 42), `GroupKFold` por motor só nos motores de treino (5 dobras; 3 para redes e MiniRocket), objetivo RMSE com RUL truncada em 125, janela em {10, 15, 20, 30, 40, 50, 60} (0 = sem atributos temporais), representação global ou por regime, regra do erro-padrão (menor janela dentro de 1 EP do melhor), orçamentos Ridge/MLP 30, RF/CatBoost 40, XGBoost/XGBoost NASA/LightGBM 60, redes e MiniRocket 20 (Random Forest do FD002 e FD004 parou em 14). Teste oficial: `--part tuned`, 3 sementes (42, 7, 123), refit na CPU. A janela saudável H da detecção (25/35/50) não faz parte desta busca.

