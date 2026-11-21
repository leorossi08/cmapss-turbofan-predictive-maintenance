#!/bin/bash
# Colab (uma máquina por semente): teste oficial com os hiperparâmetros e a janela escolhidos por validação cruzada (`--part tuned`).
# Uso: bash scripts/colab_tuned.sh SEMENTE   (de /content/pdm). 4 raias (uma por subconjunto), cada uma presa a 2 núcleos; refit sempre na CPU.
SEED=$1
cd /content/pdm || exit 1
mkdir -p .venv/bin /content/logs && ln -sf "$(command -v python3)" .venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=1 PDM_TORCH_THREADS=1
NC=$(nproc --all); i=0
for FD in FD004 FD002 FD003 FD001; do
  c=$(( (i * 2) % NC )); i=$((i+1))
  ( taskset -c $c,$(( (c+1) % NC )) python -m pdm_cmaps.experiments.benchmark_oficial_completo --subsets $FD --part tuned --seeds $SEED \
      --params-dir data/hiperparametros --output-dir /content/out_tuned/${FD}_s$SEED >> /content/logs/tuned_${FD}_s$SEED.log 2>&1
    echo "FIM $FD seed=$SEED rc=$?" >> /content/logs/fim_tuned.log ) &
done
wait
echo "TUDO seed=$SEED" >> /content/logs/fim_tuned.log
