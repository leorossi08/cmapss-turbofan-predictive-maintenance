# Atalhos para as tarefas mais comuns. `make help` lista os alvos.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help install test pipeline tudo graficos

help:        ## lista os alvos
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:     ## instala o pacote (com PyTorch) e o pytest em modo editável
	$(PY) -m pip install -e ".[dev]"

test:        ## roda a suíte de testes (os marcados como `slow` ficam de fora)
	$(PY) -m pytest

pipeline:    ## pipeline de um subconjunto: make pipeline SUBSET=FD004
	$(PY) -m pdm_cmaps --subset $(or $(SUBSET),FD004)

tudo:        ## pytest + pipeline + todos os estudos + tabelas (horas; retomável): scripts/rodar.py
	$(PY) scripts/rodar.py rodar tudo

graficos:    ## reconstrói e abre os gráficos Plotly a partir de results/
	$(PY) -m pdm_cmaps.reporting.plotly_hub --open
