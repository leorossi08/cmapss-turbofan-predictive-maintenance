# data/hiperparametros/

Resultado da busca de hiperparâmetros (Optuna, validação cruzada por motor só nos motores de treino): para cada subconjunto e modelo,
`melhores.json` (parâmetros escolhidos, janela, representação, RMSE e erro-padrão da validação cruzada) e `tentativas.csv` (todas as tentativas).
Os bancos do Optuna (`estudo.db`) não estão aqui. Serve de entrada para `benchmark_oficial_completo --part tuned --params-dir data/hiperparametros`.
Para refazer a busca: `python scripts/rodar.py rodar hiperparametros` (escreve em `results/hiperparametros/`).
