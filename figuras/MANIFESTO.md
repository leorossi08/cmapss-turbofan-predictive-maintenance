# Manifesto das figuras

Cada linha é uma figura do PDF da monografia (numeração do PDF). `python figuras/gerar.py N` roda o script. Saída em `figuras/saida/`. Gerado a partir de `gerar.py` e do `.lof` do PDF.

| Figura | Legenda (início) | Arquivo | Script | Entradas |
|---|---|---|---|---|
| 1.1 | T50 bruto e resíduo do motor 88 do FD004 | `F24_bruto_vs_residuo.png` | `figuras/novas.py`  | results/maintenance_guide_fd004/tables/virtual_sensor_residuals_t50_h50.csv, tcc-writing/tables/margin_check_point_thresholds_raw.csv |
| 4.3 | Evolução do T50 do motor 1 do FD004 dentro de cada regi | `F03_metodo_a_regimes.png` | `figuras/metodo_a_regimes.py`  | dados C-MAPSS (data/CMaps) |
| 4.5 | De um motor real a uma série saudável e ao gêmeo com fa | `F21a_injecao_serie_saudavel.png` | `figuras/injecao_series.py`  | dados C-MAPSS (data/CMaps) |
| 4.6 | As cinco formas de falha injetadas (série saudável do m | `F21b_injecao_formas_de_falha.png` | `figuras/injecao_series.py`  | dados C-MAPSS (data/CMaps) |
| 5.1 | Agrupamento dos seis regimes do FD004 | `F02_kmeans_regimes.png` | `figuras/regimes_kmeans.py`  | dados C-MAPSS (data/CMaps) |
| 5.2 | Comparação dos nove melhores modelos de RUL no FD004 | `F06_comparacao_modelos_ic_fd004.png` | `figuras/consistencia.py`  | tcc-writing/tables/ (várias), results/maintenance_guide_fd004/tables/{thesis_bootstrap_ci,alert_timeline}.csv, results/estudos_complementares/exp_a_trajetorias_seed42.csv |
| 5.3 | RUL prevista contra RUL real do XGBoost com atributos t | `F15_dispersao_predito_real_xgboost_temporal.png` | `figuras/revisao.py` `--so` `f15` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.4 | Resíduo do sensor virtual do T50 e sua EWMA no motor 17 | `F05_carta_controle_residuo_fd004.png` | `figuras/revisao.py` `--so` `f05` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.5 | Sensores do motor 179 do FD004 ao redor do ciclo 242 | `F04_motor179_queda_ciclo242.png` | `figuras/motor179_queda.py`  | dados C-MAPSS (data/CMaps) |
| 5.6 | RUL prevista pelos modelos ao longo da vida do motor 17 | `F14_linha1.png` | `figuras/revisao.py` `--so` `f14` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.7 | RUL prevista pelos modelos ao longo da vida do motor 17 | `F14_linha2.png` | `figuras/revisao.py` `--so` `f14` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.8 | RUL prevista pelos modelos ao longo da vida do motor 17 | `F14_linha3.png` | `figuras/revisao.py` `--so` `f14` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.9 | RUL prevista pelos modelos ao longo da vida do motor 17 | `F14_linha4.png` | `figuras/revisao.py` `--so` `f14` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.10 | Motor 116 do FD004 com desvios isolados e alerta | `F17a_envelope_motor_completo.png` | `figuras/revisao.py` `--so` `f17` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.11 | Três casos ampliados de desvio (motores 206, 179 e 116  | `F17b_envelope_desvios_isolados.png` | `figuras/revisao.py` `--so` `f17` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.12 | Tamanho dos desvios e falsos alertas com e sem persistê | `F17c_envelope_persistencia_frota.png` | `figuras/revisao.py` `--so` `f17` | results/maintenance_guide_fd004/tables/{rul_holdout_predictions,envelope_scored_trace_h50}.csv |
| 5.13 | Decomposição do ganho: sensores brutos, média do regime | `F23_decomposicao_ganho.png` | `figuras/consistencia.py`  | tcc-writing/tables/ (várias), results/maintenance_guide_fd004/tables/{thesis_bootstrap_ci,alert_timeline}.csv, results/estudos_complementares/exp_a_trajetorias_seed42.csv |
| 5.16 | Curvas operacionais no FD004 | `F11_curvas_operacionais.png` | `figuras/deteccao.py`  | results/injecao_deteccao/runs/FD004_*_H25_*.csv |
| 5.17 | Detecção acumulada após o início da falha | `F12_deteccao_acumulada.png` | `figuras/deteccao.py`  | results/injecao_deteccao/runs/FD004_*_H25_*.csv |
| 5.18 | Trajetórias de previsão e intervalo conformal nos motor | `F07_trajetorias_intervalo_conformal.png` | `figuras/consistencia.py`  | tcc-writing/tables/ (várias), results/maintenance_guide_fd004/tables/{thesis_bootstrap_ci,alert_timeline}.csv, results/estudos_complementares/exp_a_trajetorias_seed42.csv |
| 5.19 | Comparação entre intervalos de Mondrian por 3 tercis e  | `F27_conformal_ajustado_cqr.png` | `figuras/consistencia.py`  | tcc-writing/tables/ (várias), results/maintenance_guide_fd004/tables/{thesis_bootstrap_ci,alert_timeline}.csv, results/estudos_complementares/exp_a_trajetorias_seed42.csv |
| 5.20 | Custo por ciclo relativo ao oráculo conforme o limiar c | `F26_custo_por_limiar.png` | `figuras/novas.py`  | results/maintenance_guide_fd004/tables/virtual_sensor_residuals_t50_h50.csv, tcc-writing/tables/margin_check_point_thresholds_raw.csv |
| B.8 |  | `F18_ganho_ml_vs_media_regime.png` | `figuras/consistencia.py`  | tcc-writing/tables/ (várias), results/maintenance_guide_fd004/tables/{thesis_bootstrap_ci,alert_timeline}.csv, results/estudos_complementares/exp_a_trajetorias_seed42.csv |
| B.11 |  | `F08_criticidade_tradeoff.png` | `figuras/consistencia.py`  | tcc-writing/tables/ (várias), results/maintenance_guide_fd004/tables/{thesis_bootstrap_ci,alert_timeline}.csv, results/estudos_complementares/exp_a_trajetorias_seed42.csv |

## Fora deste repositório

- **Figura A.1** (`F16_apendice_transplante_rul_fd004.png`): saída de `python -m pdm_cmaps.experiments.fault_injection --subset FD004 --stage score` (`fig_rul_transplante.png`); exige a rodada de injeção completa.
- **Fotos e diagramas** (Figuras 2.1 a 2.4, 2.6 a 2.11, 4.1, 4.2, 4.4, 5.14, 5.15): ficam no repositório do texto, em `diagrams/` (fonte SVG e PNG); as fotos têm créditos em `figures/CREDITOS-imagens.md`.
