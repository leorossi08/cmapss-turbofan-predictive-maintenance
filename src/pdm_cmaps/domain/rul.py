"""RUL targets and evaluation metrics (RMSE, MAE, R2, NASA score)."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score

from pdm_cmaps.domain.schema import DEFAULT_RUL_CAP



def compute_piecewise_rul(
    frame: pd.DataFrame,
    rul_max: int = DEFAULT_RUL_CAP,
    unit_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> pd.Series:
    """Compute min(RUL cap, engine final cycle minus current cycle).

    [PIECEWISE-RUL] Early in life an engine shows no measurable wear, so "200 cycles left" and
    "150 cycles left" look identical in the sensors. A linear RUL target would force the model
    to chase noise there. Capping at 125 (the standard C-MAPSS convention) says: "while healthy,
    just predict 125". It also means scores computed on capped RUL are NOT comparable with
    papers that use uncapped targets; always state the cap.
    """
    if rul_max < 1:
        raise ValueError("rul_max must be positive")
    end_of_life = frame.groupby(unit_col)[cycle_col].transform("max")
    return (end_of_life - frame[cycle_col]).clip(lower=0, upper=rul_max).astype(float)

def clip_predictions(y_pred, lower: float = 0.0, upper: float = float(DEFAULT_RUL_CAP)) -> np.ndarray:
    """Clip RUL predictions to physical bounds [lower, upper]."""
    return np.clip(np.asarray(y_pred, dtype=float), lower, upper)

def nasa_scoring_function(y_true, y_pred) -> tuple[float, float]:
    """Return total and mean asymmetric PHM score with overflow protection.

    [METRIC-CHOICE] Error d = predicted - true. Predicting too LATE (d > 0, engine fails before
    the alarm) is penalised with exp(d/10); too EARLY (d < 0, wasted life) with exp(-d/13).
    Maintenance logic: a missed failure costs far more than an early inspection. The TOTAL
    score grows with the number of engines, so compare MEAN score across different fleets.
    """
    difference = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    diff_clipped = np.clip(difference, -500.0, 500.0)
    per_row = np.where(
        diff_clipped < 0,
        np.exp(-diff_clipped / 13.0) - 1.0,
        np.exp(diff_clipped / 10.0) - 1.0,
    )
    return float(per_row.sum()), float(per_row.mean())

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return R², RMSE, MAE, and target variance on the scored observations."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "target_variance": float(np.var(truth)),
    }

def evaluate_rul_predictions(
    y_true,
    y_pred,
    clip: bool = False,
    rul_cap: float = float(DEFAULT_RUL_CAP),
) -> dict[str, float]:
    """Calculate symmetric regression metrics and asymmetric RUL costs with optional clipping."""
    truth = np.asarray(y_true, dtype=float)
    prediction = clip_predictions(y_pred, 0.0, rul_cap) if clip else np.asarray(y_pred, dtype=float)
    total, mean = nasa_scoring_function(truth, prediction)
    return {
        **{key: round(value, 6) for key, value in regression_metrics(truth, prediction).items()},
        "NASA_Score": round(total, 2),
        "Mean_NASA_Score": round(mean, 3),
        "Late_Pred_Pct": round(float(np.mean(prediction > truth) * 100.0), 1),
    }

def evaluate_stratified_rul(
    y_true,
    y_pred,
    critical_threshold: float = 50.0,
    rul_cap: float = float(DEFAULT_RUL_CAP),
) -> dict[str, dict[str, float]]:
    """Evaluate RUL predictions across full holdout and critical degradation zone (RUL < critical_threshold)."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    overall = evaluate_rul_predictions(truth, prediction, clip=False)
    overall["N_samples"] = len(truth)

    critical_mask = truth < critical_threshold
    if critical_mask.any():
        critical_zone = evaluate_rul_predictions(truth[critical_mask], prediction[critical_mask], clip=False)
        critical_zone["N_samples"] = int(np.sum(critical_mask))
    else:
        critical_zone = {"N_samples": 0}

    return {
        "overall": overall,
        "critical_zone": critical_zone,
    }


CRITICAL_THRESHOLDS = (10, 15, 20)


def extended_rul_metrics(y_true, y_pred, thresholds=CRITICAL_THRESHOLDS, hits=(10, 20)) -> dict[str, float]:
    """Metrics that complement RMSE, MAE, R2 and the NASA score, for a set of (true, predicted) RUL pairs.

    Error d = predicted - true (positive = too LATE, the dangerous side).
      Bias            mean d (a positive value means the model is optimistic on average).
      Median_AE, P90_AE   median and 90th percentile of |d| (robust to a few very bad engines).
      Hit_within_k    share of predictions with |d| <= k cycles (an operational reading of the error).
    Critical-state classification, for each c in ``thresholds``: the positive class is "true RUL <= c" and the model
    "predicts critical" when its predicted RUL <= c. Precision, recall and F1 are valid here because the positive class is
    rare (accuracy would reward always saying "not critical"). PR_AUC_c uses the continuous score -prediction, so it does
    not depend on the cut c of the prediction, only on the true class; it is NaN when the class has no member.
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    d = pred - truth
    out = {"Bias": float(np.mean(d)), "Median_AE": float(np.median(np.abs(d))), "P90_AE": float(np.percentile(np.abs(d), 90))}
    for k in hits:
        out[f"Hit_within_{k}"] = float(np.mean(np.abs(d) <= k))
    for c in thresholds:
        positive = truth <= c
        called = pred <= c
        tp = float(np.sum(positive & called))
        fp = float(np.sum(~positive & called))
        fn = float(np.sum(positive & ~called))
        precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
        recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan")
        out[f"Precision_c{c}"], out[f"Recall_c{c}"], out[f"F1_c{c}"] = precision, recall, f1
        out[f"PR_AUC_c{c}"] = float(average_precision_score(positive, -pred)) if positive.any() and not positive.all() else float("nan")
    return out
