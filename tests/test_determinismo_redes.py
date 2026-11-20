"""As redes PyTorch repetem as previsões bit a bit, qualquer que seja a contagem de threads do processo (ver modeling/determinism.py).

Medido antes da correção (FD001, semente 42): CNN 18,43 → 18,77 só por mudar de 2 para 4 threads, e LSTM 19,18 contra 16,68 em duas
execuções idênticas. Estes testes usam dados sintéticos pequenos e poucas épocas (rápidos); a verificação entre processos, com
OMP_NUM_THREADS diferentes e dados reais, está em results/logs/determinismo/.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pdm_cmaps.modeling.determinism import deterministic_torch
from pdm_cmaps.modeling.torch_models import PyTorchCNNModel, PyTorchRNNModel, PyTorchTransformerModel
from factories import make_synthetic_cmapss

FEATURES = ["T24", "T30", "T50", "P30", "Nf", "Nc"]


def _frame():
    df = make_synthetic_cmapss(n_engines=8, min_cycles=40, max_cycles=60, seed=3)
    df["RUL"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    return df


def _models(seed):
    kw = dict(seq_length=5, max_epochs=3, patience=2, random_state=seed, device="cpu")
    return {"CNN": PyTorchCNNModel(**kw), "LSTM": PyTorchRNNModel(hidden_dim=16, num_layers=1, **kw),
            "Transformer": PyTorchTransformerModel(**kw)}


@pytest.mark.parametrize("name", ["CNN", "LSTM", "Transformer"])
def test_network_predictions_do_not_depend_on_the_process_thread_count(name):
    df = _frame()
    out = []
    prev = torch.get_num_threads()
    for threads in (1, 3):
        torch.set_num_threads(threads)  # o que o processo tinha antes; fit e predict devem ignorar
        m = _models(5)[name].fit(df, FEATURES, "RUL")
        out.append(m.predict(df))
    torch.set_num_threads(prev)
    assert np.array_equal(out[0], out[1]), f"{name}: previsões diferem entre 1 e 3 threads (máx {np.abs(out[0] - out[1]).max():.3g})"


@pytest.mark.parametrize("name", ["CNN", "LSTM", "Transformer"])
def test_network_predictions_repeat_with_the_same_seed_and_differ_across_seeds(name):
    df = _frame()
    a = _models(7)[name].fit(df, FEATURES, "RUL").predict(df)
    b = _models(7)[name].fit(df, FEATURES, "RUL").predict(df)
    c = _models(8)[name].fit(df, FEATURES, "RUL").predict(df)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c), "sementes diferentes deram o mesmo resultado: a semente não está sendo usada"


def test_deterministic_context_restores_the_process_state():
    prev_threads, prev_det = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    with deterministic_torch(1, threads=1):
        assert torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == prev_threads
    assert torch.are_deterministic_algorithms_enabled() == prev_det


def test_minirocket_predictions_do_not_depend_on_the_process_thread_count():
    """O MiniRocket daqui é em PyTorch (convoluções): somas multi-thread mudam a PPV quando cruzam um viés."""
    from pdm_cmaps.modeling.minirocket import MiniRocketRULModel
    df = _frame()
    out = []
    prev = torch.get_num_threads()
    for threads in (1, 3):
        torch.set_num_threads(threads)
        m = MiniRocketRULModel(seq_length=8, num_kernels=200, random_state=4).fit(df, FEATURES, "RUL")
        out.append(m.predict_values(df))
    torch.set_num_threads(prev)
    assert np.array_equal(out[0], out[1]), f"MiniRocket: diferença máxima {np.abs(out[0] - out[1]).max():.3g}"
