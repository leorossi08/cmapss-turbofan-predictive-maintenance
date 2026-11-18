# tcc-writing/

Material de apoio à escrita da monografia (não faz parte do pacote).

- `tables/`: CSVs-resumo dos estudos, **versionados** e sincronizados de `results/` por `python -m pdm_cmaps.cli.update_all`. É daqui que saem os números do texto e do site de painéis.
- `hard_by_subset.py`: agrega as falhas sorteadas por subconjunto (ganho do ML sobre a média do regime, com IC por bootstrap de motores). Uso: `python tcc-writing/hard_by_subset.py --results results/falhas_dificeis --out tcc-writing/tables/falhas_dificeis_por_subconjunto.csv` (o `update_all` já o chama).
