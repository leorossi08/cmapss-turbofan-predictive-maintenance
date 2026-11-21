#!/bin/bash
# Colab: busca de hiperparâmetros de UM subconjunto (arg 1) e GRUPO de modelos (arg 2: tab = tabulares, rede = redes e MiniRocket).
# Uso (de /content/pdm):  bash scripts/colab_hiperparametros.sh FD004 tab [dispositivo]
# Uma máquina por subconjunto para os tabulares (CPU, processos presos a núcleos); as redes numa máquina com GPU (dispositivo cuda; só na busca).
# Retomável: cada estudo é um SQLite em /content/out_hp/<subconjunto>/<modelo>/estudo.db; cada modelo roda como um processo.
FD=${1:?subconjunto}; GRUPO=${2:?tab|rede}; DEV=${3:-cpu}
cd /content/pdm || exit 1
mkdir -p .venv/bin /content/logs && ln -sf "$(command -v python3)" .venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=1 PDM_TORCH_THREADS=1
TAB=("Ridge" "MLP" "Random Forest" "XGBoost" "XGBoost (NASA Loss)" "LightGBM" "CatBoost")
REDE=("PyTorch RNN (LSTM)" "PyTorch CNN" "PyTorch Transformer" "MiniRocket + RidgeCV")
if [ "$GRUPO" = tab ]; then MODELOS=("${TAB[@]}"); else MODELOS=("${REDE[@]}"); fi
NC=$(nproc --all); i=0  # nproc sozinho respeita OMP_NUM_THREADS (=1 aqui) e dava NC=1
for M in "${MODELOS[@]}"; do
  c=$(( (i * 2) % NC )); i=$((i+1))
  ( taskset -c $c,$(( (c+1) % NC )) python -m pdm_cmaps.experiments.selecao_hiperparametros --subsets $FD --models "$M" --device $DEV --n-jobs 2 \
      --output-dir /content/out_hp >> "/content/logs/hp_${FD}_${GRUPO}.log" 2>&1; echo "FIM $FD $M rc=$?" >> /content/logs/fim_hp.log ) &
  # no máximo (núcleos/2) processos ao mesmo tempo
  # espera por uma vaga contando os processos da busca (jobs dentro de $( ) não enxerga os filhos em segundo plano)
  while [ "$(pgrep -fc "selecao_hiperparametros --subsets")" -ge $(( NC / 2 )) ]; do sleep 5; done
done
wait
echo "TUDO $FD $GRUPO" >> /content/logs/fim_hp.log
