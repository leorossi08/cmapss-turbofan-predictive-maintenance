"""Uncertainty for RUL predictions: split-conformal prediction intervals.

WHAT   Turns a point prediction "RUL = 42 cycles" into an interval "RUL in [28, 61] with ~90%
       coverage". The interval, not the point, is what an engineer should act on.
WHY    Research question RQ-B: how should uncertainty be presented and used? A conservative
       (lower-bound) rule needs an interval whose coverage is verified on held-out engines.
CONCEPTS
  [CALIBRATION]  split conformal prediction: absolute/signed errors on a CALIBRATION set give
                 the interval half-widths. No distributional assumption on the errors.
  [LEAKAGE]      the calibration set must be out-of-fold predictions on TRAINING engines
                 (models.grouped_oof). Never calibrate on the engines used to report coverage.
  [PSEUDOREPLICATION] cycles of one engine are correlated, so the 90% guarantee is only
                 approximate; we therefore REPORT empirical coverage on held-out engines
                 (overall, in the critical zone, and per engine) instead of trusting the theory.
  Mondrian bins  RUL errors are larger when the true RUL is large (early life). Calibrating
                 separately in bins of the PREDICTED RUL gives locally valid widths; an optional
                 second key (e.g. the operating regime) gives per-regime validity.
  Normalized     NormalizedConformalRULInterval scales residuals by a causal difficulty score
                 (recent prediction spread of the engine) so widths adapt engine by engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RUL_CAP = 125.0


class ConformalRULInterval:
    """Asymmetric split-conformal intervals with Mondrian bins on the predicted RUL."""

    def __init__(self, alpha: float = 0.10, n_bins: int = 3, rul_cap: float = DEFAULT_RUL_CAP):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.rul_cap = float(rul_cap)
        self.bin_edges_: np.ndarray | None = None
        self.offsets_: list[tuple[float, float]] = []

    def _bin_index(self, pred: np.ndarray) -> np.ndarray:
        assert self.bin_edges_ is not None
        return np.clip(np.searchsorted(self.bin_edges_, pred, side="right") - 1, 0, self.n_bins - 1)

    def fit(self, y_cal, pred_cal, groups=None) -> "ConformalRULInterval":
        """Learn per-bin signed-residual quantiles from calibration (out-of-fold) predictions.

        ``groups`` (optional, e.g. the operating regime of each row) adds a second Mondrian key: offsets are
        learned per (group, RUL bin) and fall back to the RUL bin alone when a cell has < 20 rows.
        """
        y = np.asarray(y_cal, dtype=float)
        pred = np.asarray(pred_cal, dtype=float)
        if y.shape != pred.shape or y.size < 20 * self.n_bins:
            raise ValueError("need matching calibration arrays with >= 20 points per bin")
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_edges_ = np.quantile(pred, quantiles)
        self.bin_edges_[0], self.bin_edges_[-1] = -np.inf, np.inf
        idx = self._bin_index(pred)
        residual = y - pred
        self.offsets_ = [self._offsets(residual[idx == b]) for b in range(self.n_bins)]
        self.group_offsets_ = {}
        if groups is not None:
            g = np.asarray(groups)
            for key in np.unique(g):
                for b in range(self.n_bins):
                    r = residual[(g == key) & (idx == b)]
                    if r.size >= 20:
                        self.group_offsets_[(key, b)] = self._offsets(r)
        return self

    def _offsets(self, r: np.ndarray) -> tuple[float, float]:
        n = r.size
        # finite-sample conformal levels (Vovk et al.); 'higher'/'lower' keep the guarantee conservative
        hi_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha / 2.0)) / n)
        lo_level = max(0.0, np.floor((n + 1) * (self.alpha / 2.0)) / n)
        return float(np.quantile(r, lo_level, method="lower")), float(np.quantile(r, hi_level, method="higher"))

    def predict_interval(self, pred, groups=None) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) clipped to the physical RUL range [0, cap]."""
        if self.bin_edges_ is None:
            raise RuntimeError("fit the interval model before predicting")
        p = np.asarray(pred, dtype=float)
        idx = self._bin_index(p)
        g = np.asarray(groups) if groups is not None else None
        offs = [
            getattr(self, "group_offsets_", {}).get((g[k], i), self.offsets_[i]) if g is not None else self.offsets_[i]
            for k, i in enumerate(idx)
        ]
        lo = np.array([o[0] for o in offs])
        hi = np.array([o[1] for o in offs])
        return np.clip(p + lo, 0.0, self.rul_cap), np.clip(p + hi, 0.0, self.rul_cap)


def causal_local_spread(pred, units, window: int = 10) -> np.ndarray:
    """Per-row difficulty score from the PAST of each engine: rolling std of its last ``window`` predictions.

    [CAUSALITY] uses only predictions up to the current cycle of the same engine (rows must be sorted by
    unit and cycle). Unstable recent predictions signal a harder case and get a wider interval.
    """
    s = pd.Series(np.asarray(pred, dtype=float))
    out = s.groupby(np.asarray(units)).transform(lambda v: v.rolling(window, min_periods=2).std())
    return out.fillna(0.0).to_numpy()


class NormalizedConformalRULInterval(ConformalRULInterval):
    """Locally adaptive split conformal: residuals are divided by a difficulty score d = spread + floor.

    Interval = pred + q * d, with q the per-bin conformal quantiles of r / d. Same finite-sample marginal
    guarantee as the plain version; widths now vary engine by engine (Lei et al., 2018, section 5.2).
    The floor (median calibration spread) keeps d away from zero on flat, early-life predictions.
    """

    def fit(self, y_cal, pred_cal, spread_cal) -> "NormalizedConformalRULInterval":
        spread = np.asarray(spread_cal, dtype=float)
        self.floor_ = float(max(np.median(spread), 1e-6))
        d = spread + self.floor_
        y = np.asarray(y_cal, dtype=float)
        p = np.asarray(pred_cal, dtype=float)
        # reuse the parent machinery on normalized residuals: pseudo-target p + (y - p) / d
        super().fit(p + (y - p) / d, p)
        return self

    def predict_interval(self, pred, spread) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(pred, dtype=float)
        d = np.asarray(spread, dtype=float) + self.floor_
        idx = self._bin_index(p)
        lo = np.array([self.offsets_[i][0] for i in idx]) * d
        hi = np.array([self.offsets_[i][1] for i in idx]) * d
        return np.clip(p + lo, 0.0, self.rul_cap), np.clip(p + hi, 0.0, self.rul_cap)


def interval_metrics(y_true, lower, upper, critical_threshold: float = 50.0) -> dict[str, float]:
    """Empirical coverage and width, overall and in the critical zone (true RUL < threshold)."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    inside = (y >= lo) & (y <= hi)
    crit = y < critical_threshold
    return {
        "coverage": float(inside.mean()),
        "mean_width": float((hi - lo).mean()),
        "coverage_critical_zone": float(inside[crit].mean()) if crit.any() else float("nan"),
        "mean_width_critical_zone": float((hi - lo)[crit].mean()) if crit.any() else float("nan"),
    }
