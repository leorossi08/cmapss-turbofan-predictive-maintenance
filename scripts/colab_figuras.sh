#!/bin/bash
# Colab: roda os testes e regera todas as figuras da monografia. Uso (de /content/pdm): bash scripts/colab_figuras.sh
# Grava /content/logs/figuras.log; a última linha é "FIM figuras rc=0" ou "ERRO figuras rc=N" (para o Monitor).
cd /content/pdm || exit 1
mkdir -p /content/logs
fc-list | grep -qi "inter" || { apt-get install -y -qq fonts-inter >/dev/null 2>&1; rm -rf ~/.cache/matplotlib; }  # fonte do estilo das figuras
{
  python -m pytest -q -x tests 2>&1 | tail -n 15
  [ "${PIPESTATUS[0]}" -eq 0 ] && python figuras/gerar.py --todas
} > /content/logs/figuras.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "FIM figuras rc=0" >> /content/logs/figuras.log; else echo "ERRO figuras rc=$rc" >> /content/logs/figuras.log; fi
