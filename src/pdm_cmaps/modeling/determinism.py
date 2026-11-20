"""Execução determinística das redes PyTorch (mesma semente + mesmos dados = mesmas previsões, em qualquer máquina e contagem de threads).

PROBLEMA MEDIDO (2026-11-18, FD001, semente 42, RMSE no holdout): a CNN e o Transformer repetem bit a bit com a mesma contagem de threads,
mas mudam quando ela muda (CNN 18,43 com 2 threads e 18,77 com 4); o LSTM nem repete na mesma configuração (19,18 e 16,68 com 2 threads).
Causa: a ordem de redução de ponto flutuante entre threads (e, no LSTM, o kernel recorrente) muda o caminho do otimizador, e o *early
stopping* amplifica diferenças minúsculas. Semear `torch`/`numpy` não basta.

CORREÇÃO: dentro de `fit` e `predict` (a) semear python, numpy e torch, (b) fixar o número de threads (padrão 1, ver PDM_TORCH_THREADS),
(c) ligar `torch.use_deterministic_algorithms` (warn_only: um operador sem versão determinística avisa, não derruba a rodada) e
(d) restaurar o estado anterior ao sair, para não alterar o resto do processo. O custo é tempo: 1 thread é mais lento que 4.
"""
from __future__ import annotations

import contextlib
import functools
import os
import random
import warnings

import numpy as np
import torch

DEFAULT_THREADS = 1


def deterministic_threads() -> int:
    return max(1, int(os.environ.get("PDM_TORCH_THREADS", DEFAULT_THREADS)))


@contextlib.contextmanager
def deterministic_torch(seed: int, threads: int | None = None):
    """Semeia tudo, fixa as threads e liga os algoritmos determinísticos; restaura o estado anterior ao sair."""
    threads = deterministic_threads() if threads is None else int(threads)
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # exigido pelo CUDA para o determinismo; sem efeito na CPU
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)


def deterministic(method):
    """Decorador para `fit`/`predict` de um modelo com atributo `random_state`."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with deterministic_torch(int(getattr(self, "random_state", 42) or 42)):
            return method(self, *args, **kwargs)
    return wrapper
