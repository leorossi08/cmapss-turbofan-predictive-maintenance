"""Desacoplamento de regimes operacionais."""


import numpy as np
import pandas as pd
from pdm_cmaps.detection.regime import RegimeDecoupler

from factories import make_synthetic_cmapss, sample_frame


def test_variance_decomposition_adds_to_total():
    frame = sample_frame()
    frame["regime_id"] = frame["unit_id"] % 3
    result = RegimeDecoupler.variance_decomposition(frame, "T50")
    assert np.isclose(result["total_var"], result["between_regime_var"] + result["within_regime_var"])
    assert np.isclose(result["pct_between_regime"] + result["pct_within_regime"], 100.0)
    assert result["Between_regime_pct"] == result["pct_between_regime"]
    assert result["Within_regime_pct"] == result["pct_within_regime"]
    assert 0.0 <= result["regime_baseline_r2"] <= 1.0


def test_regime_k_sweep_diagnostics():
    """Verify empirical k-selection evaluation (Silhouette, Calinski-Harabasz, BIC)."""
    df = make_synthetic_cmapss(n_engines=8, seed=42)
    decoupler = RegimeDecoupler()
    sweep = decoupler.evaluate_k_sweep(df, k_range=range(2, 6))
    assert isinstance(sweep, pd.DataFrame)
    assert "k" in sweep.columns
    assert "silhouette" in sweep.columns
    assert "calinski_harabasz" in sweep.columns
    assert "bic" in sweep.columns
    assert len(sweep) == 4
    assert (sweep["silhouette"] >= -1.0).all() and (sweep["silhouette"] <= 1.0).all()
    assert (sweep["calinski_harabasz"] > 0).all()


def test_regime_lexicographical_centroid_sorting():
    """Verify cluster ordering is physical and lexicographically stable across runs."""
    df = make_synthetic_cmapss(n_engines=10, seed=42)
    r1 = RegimeDecoupler(n_clusters=6, random_state=42).fit(df)
    r2 = RegimeDecoupler(n_clusters=6, random_state=123).fit(df)
    # Centroid coordinates sorted by (altitude, mach, TRA) should be strictly monotonic in altitude
    centers1 = r1.kmeans.cluster_centers_
    centers_phys1 = r1.scaler.inverse_transform(centers1)
    # Check that regime_ids map to lexicographically ordered physical centers
    ordered_centers = np.zeros_like(centers_phys1)
    for cl_idx, r_id in r1.cluster_to_regime_id.items():
        ordered_centers[r_id] = centers_phys1[cl_idx]
    # Altitude (setting_1) should be non-decreasing across ordered regime IDs
    assert np.all(np.diff(ordered_centers[:, 0]) >= -1e-3)
    # ... and the same must hold for another random seed: the ordering is a property of the data, not of the initialisation
    phys2 = r2.scaler.inverse_transform(r2.kmeans.cluster_centers_)
    ordered2 = np.zeros_like(phys2)
    for cl_idx, r_id in r2.cluster_to_regime_id.items():
        ordered2[r_id] = phys2[cl_idx]
    assert np.all(np.diff(ordered2[:, 0]) >= -1e-3)


def test_regime_distance_margin_and_relative_margin():
    """Verify distance_margin includes absolute and dimensionless relative margins."""
    df = make_synthetic_cmapss(n_engines=4, seed=42)
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42).fit(df)
    margins = decoupler.distance_margin(df)
    assert "nearest_distance" in margins.columns
    assert "runner_up_distance" in margins.columns
    assert "distance_margin" in margins.columns
    assert "relative_margin" in margins.columns
    assert (margins["nearest_distance"] <= margins["runner_up_distance"] + 1e-6).all()
    assert (margins["distance_margin"] >= -1e-6).all()
    assert (margins["relative_margin"] >= -1e-6).all() and (margins["relative_margin"] <= 1.0 + 1e-6).all()


def test_regime_standardization_and_out_of_sample_variance():
    """Verify regime-wise standardization and clean separation of descriptive vs OOS variance."""
    train_df = make_synthetic_cmapss(n_engines=8, seed=42)
    test_df = make_synthetic_cmapss(n_engines=4, seed=99)
    decoupler = RegimeDecoupler(n_clusters=6, random_state=42).fit(train_df)
    train_trans = decoupler.transform(train_df)
    test_trans = decoupler.transform(test_df)

    # Standardize by regime
    decoupler.fit_sensor_scaler(train_trans, ["T50", "P30"])
    std_train = decoupler.standardize_sensors(train_trans, ["T50", "P30"])
    # Per-regime training mean of standardized features should be ~0.0
    for r_id in train_trans["regime_id"].unique():
        sub = std_train.loc[std_train["regime_id"] == r_id, "T50_norm"]
        assert abs(sub.mean()) < 1e-3
        assert abs(sub.std() - 1.0) < 1e-2 or sub.std() == 0.0

    # Variance decomposition separating descriptive vs OOS
    decomp = RegimeDecoupler.variance_decomposition(test_trans, "T50", reference_frame=train_trans)
    assert "descriptive_between_pct" in decomp or "pct_between_regime" in decomp
    assert "oos_r2" in decomp or "regime_baseline_r2" in decomp
    assert decomp["baseline_evaluation"] == "out_of_sample"
