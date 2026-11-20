"""Teste pareado por motor do benchmark de RUL (experiments/benchmark_repeated.engine_level_tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm_cmaps.experiments.benchmark_repeated import engine_level_tests, merge_lanes


def _errors(n_seeds, n_engines=30, delta=2.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sd in range(n_seeds):
        for u in range(n_engines):
            base = rng.gamma(4, 2)
            for model, add in (("XGBoost (+temporal)", 0.0), ("PyTorch CNN", delta)):
                rows.append({"Protocol": "official", "Subset": "FDX", "Seed": sd, "Model": model, "unit_id": u, "abs_err": base + add + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_each_engine_counts_once_however_many_seeds():
    """Repetir as sementes não pode inflar o número de unidades nem encolher o p-valor: um ponto por motor."""
    few, many = engine_level_tests(_errors(1)), engine_level_tests(_errors(5))
    for t in (few, many):
        row = t[t["Comparison"].str.endswith("PyTorch CNN")].iloc[0]
        assert row["n_engines"] == 30
    assert many.iloc[0]["Seeds_per_engine_mean"] == 5 and few.iloc[0]["Seeds_per_engine_mean"] == 1
    assert many.iloc[0]["Median_err_diff_other_minus_ref"] > 1.5          # a diferença real (2,0) aparece
    assert few.iloc[0]["p_wilcoxon_engines"] < 1e-3 and many.iloc[0]["p_wilcoxon_engines"] < 1e-3


def test_no_difference_gives_a_large_p_value():
    t = engine_level_tests(_errors(3, delta=0.0), others=["PyTorch CNN"])
    assert t.iloc[0]["p_wilcoxon_engines"] > 0.05


def test_merge_lanes_requires_every_lane(tmp_path):
    for sub in ("FD001", "FD002"):
        (tmp_path / sub).mkdir()
        for name in ("benchmark_rows.csv", "benchmark_engine_errors.csv"):
            pd.DataFrame({"Subset": [sub], "x": [1]}).to_csv(tmp_path / sub / name, index=False)
    merge_lanes(tmp_path, ("FD001", "FD002"))
    assert len(pd.read_csv(tmp_path / "benchmark_rows.csv")) == 2
    try:
        merge_lanes(tmp_path, ("FD001", "FD002", "FD003"))
    except FileNotFoundError as e:
        assert "FD003" in str(e)
    else:
        raise AssertionError("faltou erro para raia ausente")


def test_merge_lanes_accepts_several_folders_per_subset_and_rejects_duplicates(tmp_path):
    def write(folder, seed, model="M"):
        (tmp_path / folder).mkdir(exist_ok=True)
        pd.DataFrame({"Subset": ["FD002"], "Seed": [seed], "Protocol": ["holdout"], "Model": [model]}).to_csv(tmp_path / folder / "benchmark_rows.csv", index=False)
        pd.DataFrame({"Subset": ["FD002"], "Seed": [seed], "Protocol": ["holdout"], "Model": [model], "unit_id": [1]}).to_csv(tmp_path / folder / "benchmark_engine_errors.csv", index=False)
    write("FD002", 42)
    write("FD002_s12", 1)
    merge_lanes(tmp_path, ("FD002",))
    assert sorted(pd.read_csv(tmp_path / "benchmark_rows.csv")["Seed"]) == [1, 42]
    write("FD002_s12", 42)  # mesma semente nas duas pastas
    import pytest
    with pytest.raises(ValueError, match="duplicadas"):
        merge_lanes(tmp_path, ("FD002",))
