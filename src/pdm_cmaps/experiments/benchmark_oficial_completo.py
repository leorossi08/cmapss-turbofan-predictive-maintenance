"""Teste oficial de RUL com mais métricas e redes com normalização por regime.

WHAT   No protocolo oficial (reajuste em todo o treino; último ciclo de cada motor de test_FDxxx), guarda a PREVISÃO de cada motor
       (e a RUL verdadeira bruta, sem teto, além da truncada em 125), para calcular depois RMSE contra as duas verdades, MAE e a pontuação da NASA.
         part=main    os 13 modelos da Tabela de RUL (mesma receita de benchmark_repeated; as redes são determinísticas, então
                      repetem as previsões da rodada anterior; serve de conferência).
         part=boosting XGBoost e LightGBM com atributos temporais calculados sobre os sensores padronizados por regime (mais o regime
                      em one-hot), com janelas móveis de 5 e 10 ciclos (regime_t10) ou de 5, 10 e 30 (regime_t30); global_t30 usa a janela de 30 sem padronizar por regime (separa janela de regime).
         part=extras  as três redes (CNN, LSTM, Transformer) com janela de 30 ciclos e/ou normalização por regime:
                      configuração  ref_w15      (referência da Tabela de RUL repetida na mesma máquina das demais)
                                    global_w30  (normalização global, janela 30)
                                    regime_w15  (cada sensor padronizado dentro do regime, mais o regime em one-hot, janela 15)
                                    regime_w30  (idem, janela 30)
                      Todas as configurações são reportadas (nenhuma escolhida no teste).
WHY    Comparar redes e boosting em pé de igualdade: o boosting ainda vence quando as redes recebem normalização por regime e janela maior?
       E guardar o RMSE também contra a verdade sem truncar e a pontuação da NASA.
[LEAKAGE] regimes, estatísticas por regime e escalonadores são ajustados só nos motores de treino.

Rodar da raiz (uma raia por subconjunto, com monitor):
    .venv/bin/python -m pdm_cmaps.experiments.benchmark_oficial_completo --subsets FD001 --part main --output-dir results/benchmark_oficial/FD001
    .venv/bin/python -m pdm_cmaps.experiments.benchmark_oficial_completo --aggregate-only --output-dir results/benchmark_oficial
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdm_cmaps.config import DEFAULT_RESULTS_DIR
from pdm_cmaps.data.features import CausalFeatureEngineer
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.domain.rul import compute_piecewise_rul, evaluate_rul_predictions
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RUL_CAP, SETTING_COLUMNS
from pdm_cmaps.experiments.benchmark_repeated import _models, _prepare
from pdm_cmaps.modeling.registry import get_model_pool
from pdm_cmaps.modeling.torch_models import PyTorchCNNModel, PyTorchRNNModel, PyTorchTransformerModel
from pdm_cmaps.pipeline.extra_models import fit_predict

NETS = {"PyTorch CNN": PyTorchCNNModel, "PyTorch RNN (LSTM)": PyTorchRNNModel, "PyTorch Transformer": PyTorchTransformerModel}
EXTRA_CONFIGS = (("ref_w15", False, 15), ("global_w30", False, 30), ("regime_w15", True, 15), ("regime_w30", True, 30))  # ref_w15 repete a referência na mesma máquina das demais
BOOST_CONFIGS = (("regime_t10", (5, 10), True), ("regime_t30", (5, 10, 30), True), ("global_t30", (5, 10, 30), False))  # (nome, janelas dos atributos temporais, sensores padronizados por regime?); global_t30 separa o efeito da janela do efeito do regime
BOOST_MODELS = ("XGBoost", "LightGBM")
SEEDS_MAIN = (42, 7, 123, 1, 2)
SEEDS_EXTRAS = (42, 7, 123)


def _regime_frames(train: pd.DataFrame, evalf: pd.DataFrame, sensors: list[str], n_regimes: int):
    """Sensores padronizados dentro do regime (estatísticas do treino) e regime em one-hot; devolve (train, eval, colunas)."""
    dec = RegimeDecoupler(n_clusters=n_regimes).fit(train)
    tr, ev = dec.transform(train), dec.transform(evalf)
    dec.fit_sensor_scaler(tr, sensors)
    tr, ev = dec.standardize_sensors(tr, sensors), dec.standardize_sensors(ev, sensors)
    cols = [f"{s}_norm" for s in sensors]
    for r in range(n_regimes if n_regimes > 1 else 0):
        for f in (tr, ev):
            f[f"regime_{r}"] = (f["regime_id"] == r).astype(float)
        cols.append(f"regime_{r}")
    return tr, ev, [*cols, *SETTING_COLUMNS, "cycle"]


def _last(pred: np.ndarray, units: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(pred, dtype=float)).groupby(units).last().sort_index()


def run(subsets, seeds, part: str, out: Path, only=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "previsoes_oficial.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path, usecols=["Subset", "Seed", "Model", "Config"]).drop_duplicates()
        done = set(zip(prev["Subset"], prev["Seed"], prev["Model"], prev["Config"]))
    loader = CMapssLoader()
    for subset in subsets:
        frame = loader.load(subset)
        frame["RUL"] = compute_piecewise_rul(frame)
        off_test, true_rul = loader.load_test(subset)
        y_raw = true_rul.to_numpy(dtype=float)
        n_regimes = 6 if subset in ("FD002", "FD004") else 1
        sensors = loader.informative_sensors(frame)
        for seed in seeds:
            if part == "main":
                base, temporal, tr_fe, ev_fe = _prepare(frame, frame, off_test, sensors)
                jobs = [(name, "global_w15", kind, model, use_t) for name, kind, model, use_t in _models(seed)]
                data = lambda use_t: (tr_fe, ev_fe, temporal) if use_t else (frame, off_test, base)  # noqa: E731
            elif part == "boosting":
                rtr, rev, rcols = _regime_frames(frame, off_test, sensors, n_regimes)
                normed = [c for c in rcols if c.endswith("_norm")]
                pool = get_model_pool(seed)
                jobs, frames = [], {}
                for cfg, windows, by_regime in BOOST_CONFIGS:
                    src_tr, src_ev, src_cols, sens = (rtr, rev, rcols, normed) if by_regime else (frame, off_test, [*sensors, *SETTING_COLUMNS, "cycle"], sensors)
                    eng = CausalFeatureEngineer(sens, windows=windows).fit(src_tr, DEFAULT_HEALTHY_CYCLE_LIMIT)
                    tr_fe, ev_fe = eng.transform(src_tr), eng.transform(src_ev)
                    frames[cfg] = (tr_fe, ev_fe, [*src_cols, *[c for c in tr_fe.columns if c not in src_tr.columns]])
                    jobs += [(f"{name} (+temporal)", cfg, "sklearn", pool[name], cfg) for name in BOOST_MODELS]
                data = lambda cfg: frames[cfg]  # noqa: E731
            else:
                rtr, rev, rcols = _regime_frames(frame, off_test, sensors, n_regimes)
                base = [*sensors, *SETTING_COLUMNS, "cycle"]
                jobs = []
                for cfg, use_regime, window in EXTRA_CONFIGS:
                    for name, cls in NETS.items():
                        jobs.append((name, cfg, "deep", cls(seq_length=window, random_state=seed), use_regime))
                data = lambda use_regime: (rtr, rev, rcols) if use_regime else (frame, off_test, base)  # noqa: E731
            for name, cfg, kind, model, flag in jobs:
                if (subset, seed, name, cfg) in done or (only and name not in only):
                    continue
                tr, ev, cols = data(flag)
                t0 = time.perf_counter()
                pred = fit_predict(kind, model, tr, cols, ev)
                last = _last(pred, ev["unit_id"].to_numpy())
                rows = pd.DataFrame({"Subset": subset, "Seed": seed, "Model": name, "Config": cfg, "unit_id": last.index,
                                     "pred": last.to_numpy(), "rul_raw": y_raw, "rul_cap": np.clip(y_raw, 0.0, float(DEFAULT_RUL_CAP))})
                rows.to_csv(path, mode="a", header=not path.exists(), index=False)
                rm = float(np.sqrt(np.mean((last.to_numpy() - rows["rul_cap"].to_numpy()) ** 2)))
                print(f"[oficial] {subset} seed={seed} {cfg:10s} {name:24s} RMSE(cap) {rm:.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)


def aggregate(base: Path) -> pd.DataFrame:
    """Une as raias, calcula as métricas por (subconjunto, modelo, configuração, semente) e resume em média e desvio entre sementes."""
    parts = [pd.read_csv(f) for f in sorted(base.glob("*/previsoes_oficial.csv"))] + ([pd.read_csv(base / "previsoes_oficial.csv")] if (base / "previsoes_oficial.csv").exists() else [])
    allp = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["Subset", "Seed", "Model", "Config", "unit_id"])
    rows = []
    for (sub, seed, model, cfg), g in allp.groupby(["Subset", "Seed", "Model", "Config"]):
        pred = g["pred"].to_numpy()
        cap, raw = g["rul_cap"].to_numpy(), g["rul_raw"].to_numpy()
        m_cap = evaluate_rul_predictions(cap, pred)
        m_raw = evaluate_rul_predictions(raw, pred)
        rows.append({"Subset": sub, "Seed": seed, "Model": model, "Config": cfg, "RMSE_cap": m_cap["RMSE"], "RMSE_raw": m_raw["RMSE"],
                     "MAE_cap": m_cap["MAE"], "NASA_cap": m_cap["NASA_Score"], "Mean_NASA_cap": m_cap["Mean_NASA_Score"],
                     "NASA_raw": m_raw["NASA_Score"], "Late_pct_cap": m_cap["Late_Pred_Pct"], "n": len(g)})
    per = pd.DataFrame(rows)
    per.to_csv(base / "metricas_por_semente.csv", index=False)
    metrics = ["RMSE_cap", "RMSE_raw", "MAE_cap", "NASA_cap", "Mean_NASA_cap", "NASA_raw", "Late_pct_cap"]
    summ = per.groupby(["Subset", "Model", "Config"])[metrics].agg(["mean", "std"]).reset_index()
    summ.columns = ["_".join(c).rstrip("_") for c in summ.columns]
    summ["Seeds"] = per.groupby(["Subset", "Model", "Config"]).size().to_numpy()
    summ.to_csv(base / "resumo_oficial.csv", index=False)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--part", choices=["main", "extras", "boosting"], default="main")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "benchmark_oficial")
    ap.add_argument("--models", nargs="+", default=None, help="só estes modelos (nomes exatos)")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.aggregate_only:
        run(a.subsets, tuple(a.seeds) if a.seeds else {"main": SEEDS_MAIN, "extras": SEEDS_EXTRAS, "boosting": (42,)}[a.part], a.part, a.output_dir, a.models)
    else:
        print(aggregate(a.output_dir).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
