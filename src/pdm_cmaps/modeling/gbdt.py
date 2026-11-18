"""XGBoost trained with the asymmetric NASA/PHM loss (objective + sklearn-style wrapper).

The model pool that uses it lives in registry.py; the package overview is in modeling/__init__.py."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def nasa_xgboost_objective(y_true, y_pred):
    """Gradient/Hessian of the asymmetric PHM loss for XGBoost (d = predicted - true).

    loss = exp(-d/13) - 1 for d < 0 (too early),  exp(d/10) - 1 for d >= 0 (too late).
    d is clipped to +-30 so exp() cannot explode far from the optimum.
    """
    diff = np.clip(y_pred - y_true, -30.0, 30.0)
    grad = np.where(diff < 0, -(1.0 / 13.0) * np.exp(-diff / 13.0), (1.0 / 10.0) * np.exp(diff / 10.0))
    hess = np.where(diff < 0, (1.0 / 169.0) * np.exp(-diff / 13.0), (1.0 / 100.0) * np.exp(diff / 10.0))
    return grad, hess


class XGBoostNasaRegressor(BaseEstimator, RegressorMixin):
    """XGBoost trained with the asymmetric NASA/PHM loss.

    Why a wrapper: with a custom objective XGBoost starts every prediction at base_score=0.5, but
    RUL is ~60-125. The clipped gradient then moves each prediction only ~0.65 cycles per tree, so
    80 trees never reach the target (R² was NEGATIVE, worse than predicting the mean). Even from the mean,
    learning_rate=0.05 underfits (R² 0.69 vs 0.78 at 0.1) because the asymmetric gradients are small. Starting
    from the training-mean RUL (the optimum of a constant model) fixes this; it must be computed
    inside fit() so each CV fold / refit uses only its own training data [LEAKAGE].
    """

    def __init__(self, n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, extra_params=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.extra_params = extra_params

    def fit(self, X, y):
        from xgboost import XGBRegressor

        y_arr = np.asarray(y, dtype=float)
        self.model_ = XGBRegressor(
            objective=nasa_xgboost_objective,
            base_score=float(np.mean(y_arr)),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **(self.extra_params or {}),
        )
        self.model_.fit(X, y_arr)
        return self

    def predict(self, X):
        return self.model_.predict(X)
