"""Leitura, divisão por motor (sem vazamento) e atributos causais."""

import numpy as np
import pandas as pd
import pytest
from pdm_cmaps.data.features import (
    CausalFeatureEngineer,
    engineer_causal_temporal_features,
)
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.data.splits import split_engine_units
from pdm_cmaps.domain.schema import COLUMNS
from pdm_cmaps.modeling.registry import get_model

from factories import make_synthetic_cmapss, sample_frame


def test_loader_reads_named_fd001_schema():
    frame = CMapssLoader().load("FD001")
    assert list(frame.columns) == COLUMNS
    assert frame["unit_id"].nunique() == 100


def test_engine_holdout_has_no_shared_units():
    frame = sample_frame()
    train, test = split_engine_units(frame, test_ratio=0.3, random_state=4)
    assert not set(train).intersection(test)
    assert len(train) + len(test) == frame["unit_id"].nunique()


def test_temporal_features_do_not_cross_engine_boundaries():
    frame = sample_frame(engine_count=2)
    features = engineer_causal_temporal_features(frame, ["T50"], windows=(3,))
    first_rows = features.groupby("unit_id", sort=False).head(1)
    assert (first_rows["T50_lag1"] == first_rows["T50"]).all()
    assert (first_rows["T50_delta"] == 0.0).all()


def test_loader_rejects_duplicate_engine_cycle_rows():
    frame = sample_frame(engine_count=2)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate engine-cycle"):
        CMapssLoader.validate(duplicated)


def test_split_engine_units_eol_stratified():
    """Verify EOL-stratified holdout split produces matching EOL distributions without leakage."""
    df = make_synthetic_cmapss(n_engines=24, min_cycles=50, max_cycles=150, seed=42)
    train_units, test_units = split_engine_units(df, test_ratio=0.25, random_state=42, stratify_eol=True)
    assert len(set(train_units).intersection(test_units)) == 0
    assert len(train_units) + len(test_units) == df["unit_id"].nunique()

    # Compare EOL medians between train and test
    eol_map = df.groupby("unit_id")["cycle"].max()
    train_eols = eol_map.loc[train_units]
    test_eols = eol_map.loc[test_units]
    # KS-test or median difference should confirm similar distributions
    assert abs(train_eols.median() - test_eols.median()) < 25.0


def test_lag2_boundary_fix_at_cycle_2():
    """Verify lag2 at t=2 is filled with lag1 (x at t=1), not current x at t=2."""
    df = pd.DataFrame({
        "unit_id": [1, 1, 1],
        "cycle": [1, 2, 3],
        "T50": [100.0, 110.0, 120.0],
    })
    res = engineer_causal_temporal_features(df, ["T50"], windows=(3,))
    # At t=1: lag1=100, lag2=100
    assert res.loc[0, "T50_lag1"] == 100.0
    assert res.loc[0, "T50_lag2"] == 100.0
    # At t=2: lag1=100, lag2 must be 100 (filled with lag1), NOT 110!
    assert res.loc[1, "T50_lag1"] == 100.0
    assert res.loc[1, "T50_lag2"] == 100.0
    # At t=3: lag1=110, lag2=100
    assert res.loc[2, "T50_lag1"] == 110.0
    assert res.loc[2, "T50_lag2"] == 100.0


def test_causal_feature_engineer_startup_and_imputation():
    """Verify CausalFeatureEngineer provides is_startup indicator and healthy-median rollstd imputation."""
    train_df = make_synthetic_cmapss(n_engines=6, seed=42)
    test_df = make_synthetic_cmapss(n_engines=2, seed=99)
    engineer = CausalFeatureEngineer(sensor_cols=["T50", "T24"], windows=(3, 5))
    engineer.fit(train_df, healthy_cycle_limit=30)
    transformed = engineer.transform(test_df)
    assert "is_startup" in transformed.columns
    # cycles 1 and 2 should have is_startup == 1.0
    assert (transformed.loc[transformed["cycle"] <= 2, "is_startup"] == 1.0).all()
    assert (transformed.loc[transformed["cycle"] > 2, "is_startup"] == 0.0).all()
    # rollstd at cycle 1 should be imputed with healthy median, not rigid 0.0
    cycle1_std = transformed.loc[transformed["cycle"] == 1, "T50_rollstd_3"].to_numpy()
    assert (cycle1_std > 0.0).all()


def test_data_loader_load_test():
    """Verify CMapssLoader.load_test loads official test and RUL files."""
    loader = CMapssLoader()
    # Check if FD004 test exists on disk
    test_file = loader.data_dir / "test_FD004.txt"
    rul_file = loader.data_dir / "RUL_FD004.txt"
    if test_file.is_file() and rul_file.is_file():
        test_df, rul_series = loader.load_test("FD004")
        assert list(test_df.columns) == COLUMNS
        assert len(rul_series) == test_df["unit_id"].nunique()
        assert (rul_series >= 0).all()


def test_notebook_insights_slopes_and_nasa_objective():
    """Notebook Insights: Verify rolling causal slopes and XGBoost NASA asymmetric objective."""
    df = pd.DataFrame({
        "unit_id": [1] * 25,
        "cycle": np.arange(1, 26),
        "T50": np.linspace(1400.0, 1450.0, 25),
    })
    feat_eng = CausalFeatureEngineer(sensor_cols=["T50"], windows=(5, 10))
    feat_eng.fit(df, healthy_cycle_limit=10)
    trans = feat_eng.transform(df)

    assert "T50_slope_5" in trans.columns
    assert "T50_slope_10" in trans.columns
    # T50 is linearly increasing, slope should be positive
    assert (trans.loc[trans["cycle"] > 10, "T50_slope_5"] > 0).all()

    # Test get_model("xgboost_nasa_loss")
    xgb_nasa = get_model("xgboost_nasa_loss", n_estimators=5, max_depth=2, random_state=42)
    X = np.random.randn(30, 4)
    y = np.random.uniform(10, 100, 30)
    xgb_nasa.fit(X, y)
    preds = xgb_nasa.predict(X)
    assert len(preds) == len(y)
    assert np.isfinite(preds).all()
