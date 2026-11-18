"""All figures: pipeline plots and thesis diagnostic figures.

Plots never compute results; they only draw tables/arrays that the
other modules produced, so you can ignore this file until you want to change a figure.
Useful reading guide for the figures:
  parity plot       predicted vs actual; points on the diagonal = perfect; the +-3 sigma band
                    shows the calibrated tolerance.
  envelope plots    healthy-calibrated corridor; a sustained exit from it is the alarm.
  figure_1..4       FD004 thesis figures (noise vs damage, physical cascade, autocorrelation,
                    fleet persistence trade-off); they read tables written earlier in the run.
"""

from __future__ import annotations

from pdm_cmaps.config import DEFAULT_PATHS

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from pdm_cmaps.domain.schema import SENSOR_UNITS


# ============================================================================
# plotting
# ============================================================================

matplotlib.use("Agg")

MIN_SENSOR_R2_FOR_PLOT = 0.75


def save_figure(figure, path: Path) -> Path:
    """Write a figure after ensuring its destination directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_baseline(summary: pd.DataFrame, path: Path) -> Path:
    """Plot pooled grouped-CV R² with fold standard deviation bars."""
    ordered = summary.sort_values("R2_pooled_OOF", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(ordered["Subpart"], ordered["R2_pooled_OOF"], yerr=ordered["R2_fold_std"], capsize=4, color="#2a7a8c")
    for index, value in enumerate(ordered["R2_pooled_OOF"]):
        ax.text(index, value + 0.012, f"{value:.3f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pooled out-of-fold R² (error bars: fold SD)")
    ax.set_title("C-MAPSS capped-RUL grouped-CV baseline")
    return save_figure(fig, path)


def plot_regime_scores(scores: pd.DataFrame, path: Path) -> Path:
    """Plot healthy-row virtual-sensor R² for each learned regime."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(scores["Regime_ID"].astype(str), scores["R2"], color="#c9742a")
    ax.set_xlabel("Learned flight regime")
    ax.set_ylabel("Held-out healthy-row R²")
    ax.set_title("FD004 healthy T50 virtual sensor by regime")
    return save_figure(fig, path)


def plot_learning_curves(summary: pd.DataFrame, path: Path) -> Path:
    """Plot repeated grouped-holdout R² as training-engine count grows."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for subset, group in summary.groupby("Subpart"):
        ax.errorbar(group["Train_engine_fraction"], group["Mean_R2"], yerr=group["Std_R2"], marker="o", capsize=3, label=subset)
    ax.set_xlabel("Fraction of training engines")
    ax.set_ylabel("Held-out R²")
    ax.set_title("Learning curves by subset")
    ax.legend()
    return save_figure(fig, path)


def plot_sensor_variance(results: pd.DataFrame, path: Path) -> Path:
    """Plot between- and within-regime sensor variance shares."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(results))
    ax.bar(positions, results["Between_regime_pct"], label="Between regime", color="#2a7a8c")
    ax.bar(positions, results["Within_regime_pct"], bottom=results["Between_regime_pct"], label="Within regime", color="#c9742a")
    ax.set_xticks(positions, results["Sensor"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of total variance (%)")
    ax.set_title("FD004 sensor variance by operating regime")
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_failure_detection(
    scored: pd.DataFrame,
    engines: pd.DataFrame,
    path: Path,
    healthy_cycle_limit: int = 50,
    sensor_name: str = "T50",
) -> Path:
    """Plot a representative engine's sensor residual alarm and first threshold crossing with calibrated baseline."""
    detected = engines.loc[engines["detected"]]
    if detected.empty:
        engine_id = int(engines.iloc[0]["unit_id"])
    else:
        engine_id = int(detected.sort_values("lead_time_cycles", ascending=False).iloc[0]["unit_id"])
    trace = scored.loc[scored["unit_id"] == engine_id].sort_values("cycle")
    summary = engines.loc[engines["unit_id"] == engine_id].iloc[0]
    figure, (sensor_ax, alarm_ax) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)

    unit_label = SENSOR_UNITS.get(sensor_name, "")
    unit_str = f" ({unit_label})" if unit_label else ""

    max_cycle = float(trace["cycle"].max())
    # Calibrated shading covering the full interval from 0 to healthy_cycle_limit and post-calibration
    for axis in (sensor_ax, alarm_ax):
        axis.axvspan(0, healthy_cycle_limit, color="#55b982", alpha=0.15, label=f"Healthy calibration (cycles 0–{healthy_cycle_limit})")
        axis.axvspan(healthy_cycle_limit, max_cycle + 5, color="#e46f70", alpha=0.08, label="Online monitoring (post-cal)")
        axis.axvline(healthy_cycle_limit, color="#2e7d32", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Calibration limit (cycle {healthy_cycle_limit})")

    # If multi-regime, plot points by regime to avoid false oscillatory continuous lines
    has_regimes = "regime_id" in trace.columns and trace["regime_id"].nunique() > 1
    if has_regimes:
        cmap = plt.get_cmap("tab10")
        for r_id, r_rows in trace.groupby("regime_id"):
            sensor_ax.scatter(
                r_rows["cycle"], r_rows[sensor_name],
                color=cmap(int(r_id) % 10), s=18, alpha=0.75, label=f"Regime {r_id}" if int(r_id) < 6 else "",
                zorder=3
            )
        sensor_ax.plot(trace["cycle"], trace["prediction"], color="#326bff", linewidth=1.5, linestyle="--", label="Virtual sensor (Exogenous)")
    else:
        sensor_ax.plot(trace["cycle"], trace[sensor_name], color="#26364a", linewidth=1.6, label=f"Measured {sensor_name}")
        sensor_ax.plot(trace["cycle"], trace["prediction"], color="#326bff", linewidth=1.5, linestyle="--", label="Virtual sensor")

    sensor_ax.set_ylabel(f"{sensor_name}{unit_str}")
    sensor_ax.set_title(f"Held-out engine {engine_id}: measured and virtual {sensor_name} with calibrated baseline")
    sensor_ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    sensor_ax.set_xlim(0, max_cycle + 5)
    sensor_ax.grid(True, linestyle=":", alpha=0.5)

    alarm_ax.plot(trace["cycle"], trace["ewma_abs_z"], color="#7444ed", linewidth=1.6, label="EWMA absolute Z-score (α = 0.10)")
    if "ewma_ucl" in trace.columns:
        alarm_ax.plot(trace["cycle"], trace["ewma_ucl"], color="#d62728", linestyle=":", linewidth=1.8, label="Dynamic finite-horizon UCL (3σ)")
    else:
        alarm_ax.axhline(3.0, color="#d62728", linestyle=":", linewidth=1.8, label="Alarm threshold (3σ)")
    alarm_cycle = summary["first_alarm_cycle"]
    if pd.notna(alarm_cycle):
        alarm_row = trace.loc[trace["cycle"] == alarm_cycle]
        if not alarm_row.empty:
            alarm_val = float(alarm_row["ewma_abs_z"].iloc[0])
            alarm_ax.scatter([alarm_cycle], [alarm_val], marker="*", s=180, color="#bd1e1e", zorder=5, label=f"First alarm (cycle {int(alarm_cycle)})")
            alarm_ax.annotate(
                f"First alarm: cycle {int(alarm_cycle)}\nLead time: {int(summary['lead_time_cycles'])} cycles",
                xy=(alarm_cycle, alarm_val),
                xytext=(max(10, alarm_cycle - 50), min(alarm_val + 0.4, 4.3)),
                arrowprops=dict(arrowstyle="->", color="#bd1e1e", lw=1.3),
                fontsize=9, fontweight="bold", color="#8b0000",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#bd1e1e", alpha=0.9),
            )
    alarm_ax.set_xlabel("Operational cycle")
    alarm_ax.set_ylabel("EWMA |Z|")
    alarm_ax.set_title(
        f"First persistent alarm: cycle {int(alarm_cycle)}; lead time {int(summary['lead_time_cycles'])} cycles"
        if pd.notna(alarm_cycle) else "No alarm crossed the threshold"
    )
    alarm_ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9)
    alarm_ax.set_xlim(0, max_cycle + 5)
    alarm_ax.set_ylim(bottom=0.0)
    alarm_ax.grid(True, linestyle=":", alpha=0.5)
    figure.tight_layout()
    return save_figure(figure, path)


def plot_actual_vs_predicted(y_true, y_pred, path: Path, title: str, value_label: str) -> Path:
    """Plot held-out actual-versus-predicted values with identity line and R²."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    score = float(r2_score(truth, prediction))
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    padding = (upper - lower) * 0.04 or 1.0
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(truth, prediction, s=10, alpha=0.25, color="#326bff", edgecolors="none")
    axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1.5, label="Perfect prediction")
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel(f"Actual {value_label}")
    axis.set_ylabel(f"Predicted {value_label}")
    axis.set_title(f"{title}\nHeld-out R² = {score:.3f}")
    axis.legend(frameon=False)
    return save_figure(figure, path)


def plot_model_r2(summary: pd.DataFrame, path: Path, score_column: str = "R2", min_r2_cutoff: float = 0.75) -> Path:
    """Rank model holdout R² values, filtering for models meeting the high-fidelity cutoff (R² >= 0.75)."""
    eligible = summary.loc[summary[score_column] >= min_r2_cutoff]
    if eligible.empty:
        eligible = summary
    ordered = eligible.sort_values(score_column, ascending=True)
    best_model = ordered.iloc[-1]["Model"]
    colors = ["#16a34a" if name == best_model else "#3b82f6" for name in ordered["Model"]]
    figure, axis = plt.subplots(figsize=(8.5, max(4.0, 0.52 * len(ordered))))
    positions = np.arange(len(ordered))
    axis.barh(positions, ordered[score_column], color=colors, height=0.65)
    axis.axvline(min_r2_cutoff, color="#dc2626", linestyle="--", linewidth=1.4, label=f"Limiar de Qualidade (R² ≥ {min_r2_cutoff:.2f})")
    axis.set_yticks(positions, ordered["Model"], fontsize=10, fontweight="bold")
    axis.set_xlabel("Held-out R²", fontsize=10, fontweight="bold")
    axis.set_xlim(left=min(0.70, float(ordered[score_column].min()) - 0.03), right=max(1.0, float(ordered[score_column].max()) + 0.05))
    axis.set_title(f"Modelos de RUL com Alta Fidelidade (R² ≥ {min_r2_cutoff:.2f}) | Melhor: {best_model}", fontsize=11, fontweight="bold")
    axis.legend(loc="lower right", frameon=True, fontsize=9)
    axis.grid(True, linestyle=":", alpha=0.5, axis="x")
    for position, value in zip(positions, ordered[score_column]):
        axis.text(value + 0.005, position, f"{value:.4f}", va="center", ha="left", fontsize=9.5, fontweight="bold")
    figure.tight_layout()
    return save_figure(figure, path)


def plot_sensor_r2(results: pd.DataFrame, path: Path) -> Path | None:
    """Rank held-out virtual-sensor R² scores across physical channels."""
    eligible = results.loc[results["R2"] >= MIN_SENSOR_R2_FOR_PLOT]
    if eligible.empty:
        return None
    ordered = eligible.sort_values("R2", ascending=True)
    best_sensor = ordered.iloc[-1]["Sensor"]
    colors = ["#c9742a" if sensor == best_sensor else "#ddb18a" for sensor in ordered["Sensor"]]
    figure, axis = plt.subplots(figsize=(8, max(6.0, 0.36 * len(ordered))))
    positions = np.arange(len(ordered))
    axis.barh(positions, ordered["R2"], color=colors)
    axis.axvline(0.0, color="#26364a", linewidth=0.8)
    axis.axvline(MIN_SENSOR_R2_FOR_PLOT, color="#bd1e1e", linestyle=":", linewidth=1.2, label=f"Display cutoff ({MIN_SENSOR_R2_FOR_PLOT:.2f})")
    axis.set_yticks(positions, ordered["Sensor"])
    axis.set_xlabel("Held-out R²")
    axis.set_title("Pooled full-lifecycle R² by sensor (all regimes)")
    axis.legend(frameon=False)
    return save_figure(figure, path)


def plot_sensor_parity_grid(predictions: pd.DataFrame, path: Path) -> Path | None:
    """Plot actual-vs-predicted held-out values, colored by operating regime."""
    sensor_scores = {
        sensor: float(r2_score(rows["Actual"], rows["Predicted"]))
        for sensor, rows in predictions.groupby("Sensor", sort=False)
        if rows["Actual"].var() > 1e-12
    }
    sensors = [sensor for sensor in dict.fromkeys(predictions["Sensor"]) if sensor_scores.get(sensor, -np.inf) >= MIN_SENSOR_R2_FOR_PLOT]
    if not sensors:
        return None
    columns = 4
    rows = int(np.ceil(len(sensors) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.2 * rows), squeeze=False)
    best_sensor = max(sensor_scores, key=sensor_scores.get) if sensor_scores else None
    for axis, sensor in zip(axes.flat, sensors):
        rows_for_sensor = predictions.loc[predictions["Sensor"] == sensor]
        truth = rows_for_sensor["Actual"].to_numpy(dtype=float)
        estimate = rows_for_sensor["Predicted"].to_numpy(dtype=float)
        score = float(r2_score(truth, estimate)) if np.var(truth) > 1e-12 else np.nan
        lower = float(min(truth.min(), estimate.min()))
        upper = float(max(truth.max(), estimate.max()))
        padding = (upper - lower) * 0.04 or 1.0
        if "regime_id" in rows_for_sensor:
            for regime_id, regime_rows in rows_for_sensor.groupby("regime_id", sort=True):
                axis.scatter(
                    regime_rows["Actual"], regime_rows["Predicted"], s=5, alpha=0.35,
                    color=plt.get_cmap("tab10")(int(regime_id) % 10), edgecolors="none",
                    rasterized=True,
                )
        else:
            axis.scatter(truth, estimate, s=5, alpha=0.2, color="#326bff", edgecolors="none", rasterized=True)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1)
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        score_label = f"{score:.3f}" if np.isfinite(score) else "n/a"
        axis.set_title(f"{sensor}{' (best)' if sensor == best_sensor else ''}: R² {score_label}")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
    for axis in axes.flat[len(sensors):]:
        axis.set_visible(False)
    regime_ids = sorted(predictions["regime_id"].dropna().unique()) if "regime_id" in predictions else []
    if regime_ids:
        handles = [Patch(facecolor=plt.get_cmap("tab10")(int(regime_id) % 10), label=f"Regime {regime_id}") for regime_id in regime_ids]
        figure.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False)
    figure.suptitle("Held-out actual vs predicted sensor values by operating regime", y=1.0)
    figure.tight_layout(rect=(0, 0.04 if regime_ids else 0, 1, 0.98))
    return save_figure(figure, path)


def plot_sensor_parity_by_channel(
    predictions: pd.DataFrame,
    directory: Path,
    scope_label: str = "held-out",
) -> list[Path]:
    """Save a separate actual-vs-predicted panel for each regime and sensor."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for sensor, rows in predictions.groupby("Sensor", sort=False):
        regime_rows = [
            (regime_id, rows.loc[rows["regime_id"] == regime_id])
            for regime_id in sorted(rows["regime_id"].dropna().unique())
        ] if "regime_id" in rows else [(None, rows)]
        regimes = []
        for regime_id, regime_frame in regime_rows:
            if len(regime_frame) < 2 or np.var(regime_frame["Actual"].to_numpy(dtype=float)) <= 1e-12:
                continue
            if r2_score(regime_frame["Actual"], regime_frame["Predicted"]) >= MIN_SENSOR_R2_FOR_PLOT:
                regimes.append(regime_id)
        if not regimes:
            continue
        columns = min(3, len(regimes))
        figure, axes = plt.subplots(
            int(np.ceil(len(regimes) / columns)), columns,
            figsize=(5.5 * columns, 4.8 * int(np.ceil(len(regimes) / columns))),
            squeeze=False,
        )
        for axis, regime_id in zip(axes.flat, regimes):
            regime_rows = rows if regime_id is None else rows.loc[rows["regime_id"] == regime_id]
            regime_truth = regime_rows["Actual"].to_numpy(dtype=float)
            regime_prediction = regime_rows["Predicted"].to_numpy(dtype=float)
            score = float(r2_score(regime_truth, regime_prediction)) if len(regime_truth) > 1 and np.var(regime_truth) > 1e-12 else np.nan
            if not np.isfinite(score) or score < MIN_SENSOR_R2_FOR_PLOT:
                axis.set_visible(False)
                continue
            lower = float(min(regime_truth.min(), regime_prediction.min()))
            upper = float(max(regime_truth.max(), regime_prediction.max()))
            padding = (upper - lower) * 0.06 or 1.0
            axis.scatter(regime_truth, regime_prediction, s=8, alpha=0.25, color="#326bff", edgecolors="none", rasterized=True)
            axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1.2)
            axis.set_xlim(lower - padding, upper + padding)
            axis.set_ylim(lower - padding, upper + padding)
            axis.set_xlabel(f"Actual {sensor}")
            axis.set_ylabel(f"Predicted {sensor}")
            score_label = f"R² {score:.3f}" if np.isfinite(score) else "R² n/a"
            axis.set_title(f"Regime {regime_id}: {score_label}, n={len(regime_rows)}")
        for axis in axes.flat[len(regimes):]:
            axis.set_visible(False)
        figure.suptitle(f"{sensor}: {scope_label} actual vs predicted, split by regime")
        figure.tight_layout()
        paths.append(save_figure(figure, Path(directory) / f"{sensor.lower()}_actual_vs_predicted.png"))
    return paths


def plot_sensor_cycle_windows_by_regime(
    predictions: pd.DataFrame,
    directory: Path,
    window_size: int = 50,
    ) -> tuple[list[Path], pd.DataFrame]:
    """Plot per-regime sensor timelines in short cycle windows with engine spread bands."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sensors = list(dict.fromkeys(predictions["Sensor"]))
    windowed = predictions.copy()
    windowed["Cycle_start"] = ((windowed["cycle"] - 1) // window_size) * window_size + 1
    windowed["Cycle_end"] = windowed["Cycle_start"] + window_size - 1
    paths = []
    metric_rows = []
    columns = 4
    rows = int(np.ceil(len(sensors) / columns))

    for (regime_id, cycle_start, cycle_end), window_rows in windowed.groupby(
        ["regime_id", "Cycle_start", "Cycle_end"], sort=True
    ):
        figure, axes = plt.subplots(rows, columns, figsize=(16, 3.0 * rows), squeeze=False)
        plotted_sensors = 0
        for axis, sensor in zip(axes.flat, sensors):
            sensor_rows = window_rows.loc[window_rows["Sensor"] == sensor]
            if sensor_rows.empty:
                axis.set_visible(False)
                continue
            actual = sensor_rows["Actual"].to_numpy(dtype=float)
            predicted = sensor_rows["Predicted"].to_numpy(dtype=float)
            score = float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 1e-12 else np.nan
            errors = actual - predicted
            metrics = {
                "R2": score,
                "RMSE": float(np.sqrt(np.mean(errors ** 2))),
                "MAE": float(np.mean(np.abs(errors))),
                "target_variance": float(np.var(actual)),
            }
            metric_rows.append({
                "Regime_ID": int(regime_id),
                "Sensor": sensor,
                "Cycle_start": int(cycle_start),
                "Cycle_end": int(cycle_end),
                "Rows": len(sensor_rows),
                "Engines": int(sensor_rows["unit_id"].nunique()),
                **metrics,
            })
            if not np.isfinite(score) or score < MIN_SENSOR_R2_FOR_PLOT:
                axis.set_visible(False)
                continue

            cycle_summary = sensor_rows.groupby("cycle").agg(
                Actual_mean=("Actual", "mean"),
                Actual_std=("Actual", "std"),
                Predicted_mean=("Predicted", "mean"),
                Predicted_std=("Predicted", "std"),
            ).fillna(0.0)
            cycles = cycle_summary.index.to_numpy(dtype=int)
            actual_mean = cycle_summary["Actual_mean"].to_numpy(dtype=float)
            actual_std = cycle_summary["Actual_std"].to_numpy(dtype=float)
            predicted_mean = cycle_summary["Predicted_mean"].to_numpy(dtype=float)
            predicted_std = cycle_summary["Predicted_std"].to_numpy(dtype=float)
            axis.plot(cycles, actual_mean, color="#26364a", linewidth=1.4, label="Actual")
            axis.fill_between(cycles, actual_mean - actual_std, actual_mean + actual_std, color="#26364a", alpha=0.12)
            axis.plot(cycles, predicted_mean, color="#326bff", linewidth=1.4, linestyle="--", label="Predicted")
            axis.fill_between(cycles, predicted_mean - predicted_std, predicted_mean + predicted_std, color="#326bff", alpha=0.14)
            score_label = f"R² {score:.3f}" if np.isfinite(score) else "R² n/a"
            axis.set_title(f"{sensor}: {score_label}")
            axis.set_xlim(cycle_start, cycle_end)
            axis.set_xlabel("Cycle")
            axis.set_ylabel("Sensor value")
            axis.grid(alpha=0.2)
            plotted_sensors += 1

        for axis in axes.flat[len(sensors):]:
            axis.set_visible(False)
        if plotted_sensors == 0:
            plt.close(figure)
            continue
        figure.suptitle(f"Regime {int(regime_id)} sensor actual vs predicted | cycles {int(cycle_start)}-{int(cycle_end)}")
        figure.legend(
            handles=[
                Line2D([0], [0], color="#26364a", label="Actual mean"),
                Line2D([0], [0], color="#326bff", linestyle="--", label="Predicted mean"),
            ],
            loc="lower center", ncol=2, frameon=False,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.97))
        paths.append(save_figure(
            figure,
            directory / f"regime_{int(regime_id)}_cycles_{int(cycle_start):03d}-{int(cycle_end):03d}.png",
        ))

    return paths, pd.DataFrame(metric_rows)


def plot_subset_rul_parity(datasets: dict[str, pd.DataFrame], predictions: dict[str, np.ndarray], path: Path) -> Path:
    """Plot pooled grouped-CV RUL actual-vs-predicted scores by subset."""
    subsets = list(datasets)
    columns = 2
    rows = int(np.ceil(len(subsets) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 5.5 * rows), squeeze=False)
    for axis, subset in zip(axes.flat, subsets):
        truth = datasets[subset]["RUL"].to_numpy(dtype=float)
        estimate = np.asarray(predictions[subset], dtype=float)
        score = float(r2_score(truth, estimate))
        lower = float(min(truth.min(), estimate.min()))
        upper = float(max(truth.max(), estimate.max()))
        padding = (upper - lower) * 0.04 or 1.0
        axis.scatter(truth, estimate, s=5, alpha=0.15, color="#326bff", edgecolors="none", rasterized=True)
        axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#26364a", linestyle="--", linewidth=1.2)
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("Actual RUL (cycles)")
        axis.set_ylabel("Predicted RUL (cycles)")
        axis.set_title(f"{subset}: grouped OOF R² {score:.3f}")
    figure.suptitle("C-MAPSS RUL actual vs predicted by subset", y=1.0)
    figure.tight_layout()
    return save_figure(figure, path)


def plot_signal_filtering_comparison(
    frame: pd.DataFrame,
    unit_id: int,
    sensor: str,
    path: Path,
    alphas: tuple[float, ...] = (0.1, 0.35, 0.7),
) -> Path:
    """Plot raw sensor data vs EWMA low-pass filtering at multiple alphas, mirroring Padua's slide 17."""
    trace = frame.loc[frame["unit_id"] == unit_id].sort_values("cycle").copy().reset_index(drop=True)
    cycles = trace["cycle"].to_numpy()
    raw = trace[sensor].to_numpy(dtype=float)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    # Panel 1: Sinal cru (Sem Filtro)
    axes[0].plot(cycles, raw, color="#bd1e1e", linewidth=1.5, label=f"{sensor} cru")
    axes[0].set_title(f"{sensor} cru (Sem filtro) - Motor {unit_id}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Ciclo operacional")
    axes[0].set_ylabel(f"{sensor} medido")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", frameon=True)

    # Panel 2: EWMA com alpha = 0.1 (alta inércia / filtro suave)
    ewma_slow = pd.Series(raw).ewm(alpha=alphas[0], adjust=False).mean().to_numpy()
    axes[1].plot(cycles, ewma_slow, color="#2e7d32", linewidth=1.8, label=f"EWMA (α = {alphas[0]})")
    axes[1].set_title(f"{sensor} EWMA (α = {alphas[0]}) - Suave", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Ciclo operacional")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper left", frameon=True)

    # Panel 3: EWMA com alpha = 0.7 (baixa inércia / mais responsivo)
    ewma_fast = pd.Series(raw).ewm(alpha=alphas[-1], adjust=False).mean().to_numpy()
    axes[2].plot(cycles, ewma_fast, color="#326bff", linewidth=1.8, label=f"EWMA (α = {alphas[-1]})")
    axes[2].set_title(f"{sensor} EWMA (α = {alphas[-1]}) - Responsivo", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Ciclo operacional")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper left", frameon=True)

    figure.suptitle(
        f"Enriquecimento Temporal: Comparativo com e sem Filtros EWMA ({sensor})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, path)


def plot_minirocket_feature_filtering(
    frame: pd.DataFrame,
    unit_id: int,
    path: Path,
    sensor: str = "T50",
    random_state: int = 42,
) -> Path:
    """Plot MiniRocket convolutional kernel response, PPV thresholding, and feature trajectory (slides 19 & 21)."""
    trace = frame.loc[frame["unit_id"] == unit_id].sort_values("cycle").copy().reset_index(drop=True)
    cycles = trace["cycle"].to_numpy()
    raw = trace[sensor].to_numpy(dtype=float)
    rul = trace["RUL"].to_numpy(dtype=float) if "RUL" in trace else None

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1 (Slide 21 left): Sinal Cru e RUL
    ax0_twin = axes[0].twinx()
    p1 = axes[0].plot(cycles, raw, color="#bd1e1e", linewidth=1.4, label=f"{sensor} cru")
    p2 = []
    if rul is not None:
        p2 = ax0_twin.plot(cycles, rul, color="#7444ed", linestyle="--", linewidth=1.4, label="RUL real")
        ax0_twin.set_ylabel("RUL (ciclos)", color="#7444ed")
        ax0_twin.tick_params(axis="y", labelcolor="#7444ed")
    axes[0].set_title(f"{sensor} cru e Degradação (RUL) - Motor {unit_id}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Ciclo operacional")
    axes[0].set_ylabel(f"{sensor} medido")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    lines = p1 + p2
    axes[0].legend(lines, [l.get_label() for l in lines], loc="upper right", frameon=True)

    # Panel 2 (Slide 19): Convoluções & Limiar PPV (demonstração com kernel determinístico L=9)
    np.random.seed(random_state)
    mid = len(raw) // 2
    window = raw[max(0, mid - 15): min(len(raw), mid + 15)]
    kernel = np.array([-1, -1, 2, -1, 2, -1, -1, 2, -1], dtype=float)
    conv_resp = np.convolve(window, kernel, mode="valid")
    conv_resp = (conv_resp - conv_resp.mean()) / (conv_resp.std() or 1.0)
    m_resp = len(conv_resp)
    b = 0.5
    colors = ["#2e7d32" if c > b else "#888888" for c in conv_resp]
    axes[1].bar(range(1, m_resp + 1), conv_resp, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
    axes[1].axhline(b, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Limiar b = {b}")
    pos_count = int((conv_resp > b).sum())
    ppv_val = pos_count / m_resp
    axes[1].set_title(f"MiniRocket PPV: {pos_count}/{m_resp} respostas > b (PPV = {ppv_val:.2f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Posição do filtro convolucional (j)")
    axes[1].set_ylabel("Resposta convolucional (Cj)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)

    # Panel 3 (Slide 21 right): Trajetória de Ativação PPV convolucional
    window_size = 15
    all_sub_c = []
    for end in range(window_size, len(raw) + 1):
        sub_w = raw[end - window_size:end]
        sub_c = np.convolve(sub_w, kernel, mode="valid")
        all_sub_c.append(sub_c)
    all_sub_c = np.vstack(all_sub_c)
    b_healthy = float(np.quantile(all_sub_c[:min(35, len(all_sub_c))], 0.45))
    ppv_trajectory = (all_sub_c > b_healthy).mean(axis=1)
    ppv_smooth = pd.Series(ppv_trajectory).ewm(alpha=0.15, adjust=False).mean().to_numpy()

    cycles_w = cycles[window_size - 1:]
    axes[2].plot(cycles_w, ppv_trajectory, color="#02818a", alpha=0.35, linewidth=1.2, label="PPV instantâneo")
    axes[2].plot(cycles_w, ppv_smooth, color="#01665e", linewidth=2.2, label="PPV suavizado (tendência)")
    axes[2].set_title(f"Atributo PPV Convolucional vs Degradação - Motor {unit_id}", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Ciclo operacional")
    axes[2].set_ylabel(f"PPV(C, b={b_healthy:.1f}) ∈ [0, 1]")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper left", frameon=True)

    figure.suptitle(
        "Enriquecimento Convolucional MiniRocket: Resposta de Filtros e PPV",
        fontsize=12, fontweight="bold", y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, path)


def plot_time_series_envelope_detection(
    scored_trace: pd.DataFrame,
    summary: dict,
    path: Path,
    title: str = "Detecção de Anomalia por Envelope de Série Temporal (Estilo Power BI)",
    sensor_name: str = "T50",
    healthy_cycle_limit: int = 50,
) -> Path:
    """Plot time-series prediction envelope ('embrulho'), highlighting breaches, first alarm, and product breakage."""
    trace = scored_trace.sort_values("cycle").copy().reset_index(drop=True)
    figure, (ax_main, ax_breach) = plt.subplots(2, 1, figsize=(12, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    cycles = trace["cycle"].to_numpy()
    actual = trace[sensor_name].to_numpy()
    pred = trace["prediction"].to_numpy()
    upper = trace["envelope_upper"].to_numpy()
    lower = trace["envelope_lower"].to_numpy()
    anomalies = trace["out_of_envelope"].to_numpy(dtype=bool)
    breach_dist = trace["breach_distance"].to_numpy()

    max_cycle = float(cycles.max())
    eol_cycle = int(summary["eol_cycle"])
    alarm_cycle = summary["first_alarm_cycle"]
    lead_time = summary["lead_time_cycles"]

    # Shading for healthy calibration and monitoring
    for ax in (ax_main, ax_breach):
        ax.axvspan(0, healthy_cycle_limit, color="#10b981", alpha=0.12, label=f"Calibração Saudável (0–{healthy_cycle_limit})")
        ax.axvspan(healthy_cycle_limit, max_cycle + 5, color="#f43f5e", alpha=0.06, label="Monitoramento Online")
        ax.axvline(healthy_cycle_limit, color="#059669", linestyle="--", linewidth=1.2, alpha=0.85, label="Limite de Calibração")

    # Main plot
    unit_label = SENSOR_UNITS.get(sensor_name, "")
    unit_str = f" ({unit_label})" if unit_label else ""

    has_regimes = "regime_id" in trace.columns and trace["regime_id"].nunique() > 1
    if has_regimes:
        cmap = plt.get_cmap("tab10")
        for r_id, r_rows in trace.groupby("regime_id"):
            ax_main.scatter(
                r_rows["cycle"], r_rows[sensor_name],
                color=cmap(int(r_id) % 10), s=16, alpha=0.75, label=f"Regime {r_id}" if int(r_id) < 6 else "",
                zorder=3
            )
        ax_main.plot(cycles, pred, color="#2563eb", linewidth=1.6, linestyle="--", label="Sensor Virtual (Predição)", zorder=2)
    else:
        ax_main.plot(cycles, actual, color="#1e293b", linewidth=1.7, label=f"Sinal Medido ({sensor_name})", zorder=3)
        ax_main.plot(cycles, pred, color="#2563eb", linewidth=1.6, linestyle="--", label="Sensor Virtual (Predição)", zorder=3)

    # Shaded Envelope ("Embrulho")
    ax_main.fill_between(
        cycles, lower, upper,
        color="#94a3b8", alpha=0.35, label=f"Envelope de Predição ('Embrulho' ±3σ: {summary.get('envelope_margin', 0):.2f})", zorder=2,
    )
    ax_main.plot(cycles, upper, color="#64748b", linestyle=":", linewidth=1.0, alpha=0.7)
    ax_main.plot(cycles, lower, color="#64748b", linestyle=":", linewidth=1.0, alpha=0.7)

    # Highlight anomalous points outside the envelope
    if np.any(anomalies):
        ax_main.scatter(
            cycles[anomalies], actual[anomalies],
            color="#ef4444", s=32, zorder=4, label=f"Anomalia Estrutural (Fora do Embrulho, n={int(np.sum(anomalies))})",
            edgecolors="#991b1b", linewidth=0.6,
        )

    # Mark First Alarm
    if alarm_cycle is not None:
        alarm_idx = np.where(cycles == alarm_cycle)[0]
        if len(alarm_idx):
            alarm_val = actual[alarm_idx[0]]
            ax_main.scatter([alarm_cycle], [alarm_val], marker="*", s=260, color="#dc2626", edgecolors="#7f1d1d", linewidth=1.2, zorder=6, label="PRIMEIRO ALARME")
            ax_main.annotate(
                f"PRIMEIRO ALARME: Ciclo {int(alarm_cycle)}\nLead Time: {int(lead_time)} ciclos de antecedência",
                xy=(alarm_cycle, alarm_val),
                xytext=(max(15, alarm_cycle - 55), alarm_val + (upper.max() - lower.min()) * 0.18),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8),
                fontsize=9.5, fontweight="bold", color="#991b1b",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#dc2626", lw=1.2),
            )

    # Mark Product Breakage (Quebra do Produto / EOL)
    eol_idx = np.where(cycles == eol_cycle)[0]
    if len(eol_idx):
        eol_val = actual[eol_idx[0]]
        ax_main.scatter([eol_cycle], [eol_val], marker="X", s=220, color="#1e1b4b", edgecolors="#dc2626", linewidth=1.5, zorder=6, label="QUEBRA DO PRODUTO (EOL)")
        ax_main.annotate(
            f"QUEBRA DO PRODUTO (EOL)\nCiclo {int(eol_cycle)} (Falha do Motor)",
            xy=(eol_cycle, eol_val),
            xytext=(max(20, eol_cycle - 45), eol_val - (upper.max() - lower.min()) * 0.22),
            arrowprops=dict(arrowstyle="->", color="#1e1b4b", lw=1.8),
            fontsize=9.5, fontweight="bold", color="#1e1b4b",
            bbox=dict(boxstyle="round,pad=0.35", fc="#e0e7ff", ec="#3730a3", lw=1.2),
        )

    ax_main.set_ylabel(f"{sensor_name}{unit_str}", fontsize=10, fontweight="bold")
    coverage_pct = summary.get("in_envelope_pct", 100.0)
    ax_main.set_title(
        f"{title} - Motor {summary.get('unit_id', 'Holdout')}\n"
        f"Cobertura dentro do embrulho: {coverage_pct:.1f}% | Alarme no Ciclo {alarm_cycle} (Antecedência: {lead_time} ciclos da Quebra)",
        fontsize=11, fontweight="bold",
    )
    ax_main.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.5)
    ax_main.grid(True, linestyle=":", alpha=0.5)
    ax_main.set_xlim(0, max_cycle + 5)

    # Breach distance plot
    ax_breach.plot(cycles, breach_dist, color="#dc2626", linewidth=1.4, label="Distância Fora do Envelope (|Δ| > Margem)")
    ax_breach.fill_between(cycles, 0, breach_dist, color="#ef4444", alpha=0.25)
    ax_breach.axhline(0.0, color="#64748b", linestyle="-", linewidth=0.8)

    if alarm_cycle is not None and len(alarm_idx):
        ax_breach.scatter([alarm_cycle], [breach_dist[alarm_idx[0]]], marker="*", s=200, color="#dc2626", zorder=5)
    if len(eol_idx):
        ax_breach.scatter([eol_cycle], [breach_dist[eol_idx[0]]], marker="X", s=180, color="#1e1b4b", edgecolors="#dc2626", zorder=5)

    ax_breach.set_xlabel("Ciclo Operacional", fontsize=10, fontweight="bold")
    ax_breach.set_ylabel("Rompimento do Envelope", fontsize=10, fontweight="bold")
    ax_breach.set_ylim(bottom=0.0)
    ax_breach.grid(True, linestyle=":", alpha=0.5)
    ax_breach.legend(loc="upper left", frameon=True, fontsize=8.5)

    figure.tight_layout()
    return save_figure(figure, path)


def plot_geometric_envelope_detection(
    scored_geom: pd.DataFrame,
    summary: dict,
    path: Path,
    title: str = "Envelope Geométrico de Sinal Digital (Rolling Peak/Valley Hull)",
    sensor_name: str = "T50",
    healthy_cycle_limit: int = 50,
) -> Path:
    """Plot digital signal geometric envelope wrapping peaks and valleys, highlighting modulation and breakage."""
    df = scored_geom.sort_values("cycle").copy().reset_index(drop=True)
    figure, (ax_main, ax_amp) = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    cycles = df["cycle"].to_numpy()
    signal = df["geom_signal"].to_numpy() if "geom_signal" in df.columns else df[sensor_name].to_numpy()
    upper_hull = df["geom_upper_hull"].to_numpy()
    lower_hull = df["geom_lower_hull"].to_numpy()
    amp = df["geom_amplitude"].to_numpy()

    unit_label = SENSOR_UNITS.get(sensor_name, "")
    unit_str = f" ({unit_label})" if unit_label else ""

    max_cycle = float(cycles.max())
    eol_cycle = int(summary["eol_cycle"])
    alarm_cycle = summary["first_alarm_cycle"]
    lead_time = summary["lead_time_cycles"]
    amp_thresh = summary["amp_threshold"]
    upper_ceil = summary["upper_ceiling"]

    for ax in (ax_main, ax_amp):
        ax.axvspan(0, healthy_cycle_limit, color="#10b981", alpha=0.12, label=f"Calibração (0–{healthy_cycle_limit})")
        ax.axvspan(healthy_cycle_limit, max_cycle + 5, color="#f43f5e", alpha=0.06, label="Operação Contínua")
        ax.axvline(healthy_cycle_limit, color="#059669", linestyle="--", linewidth=1.2, alpha=0.85)

    # Main plot: Signal and geometric envelope
    signal_label = f"Resíduo Desacoplado ({sensor_name})" if "geom_signal" in df.columns else f"Sinal Bruto ({sensor_name})"
    ax_main.plot(cycles, signal, color="#334155", linewidth=1.2, alpha=0.75, label=signal_label)
    ax_main.plot(cycles, upper_hull, color="#0284c7", linewidth=1.6, label="Envelope Geométrico Superior (Upper Hull)")
    ax_main.plot(cycles, lower_hull, color="#0284c7", linewidth=1.6, linestyle="-.", label="Envelope Geométrico Inferior (Lower Hull)")
    ax_main.fill_between(cycles, lower_hull, upper_hull, color="#38bdf8", alpha=0.22, label="Embrulho Geométrico")
    ax_main.axhline(upper_ceil, color="#b91c1c", linestyle=":", linewidth=1.5, label=f"Teto Superior ({upper_ceil:.2f})")

    # Mark First Alarm and Breakage
    if alarm_cycle is not None:
        a_idx = np.where(cycles == alarm_cycle)[0]
        if len(a_idx):
            val = signal[a_idx[0]]
            ax_main.scatter([alarm_cycle], [val], marker="*", s=260, color="#dc2626", edgecolors="#7f1d1d", zorder=6, label="PRIMEIRO ALARME")
            ax_main.annotate(
                f"PRIMEIRO ALARME: Ciclo {int(alarm_cycle)}\nModulação anômala do envelope ({lead_time} ciclos antes da falha)",
                xy=(alarm_cycle, val),
                xytext=(max(15, alarm_cycle - 50), val + (upper_hull.max() - lower_hull.min()) * 0.15),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.6),
                fontsize=9.5, fontweight="bold", color="#991b1b",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec="#dc2626", lw=1.2),
            )

    e_idx = np.where(cycles == eol_cycle)[0]
    if len(e_idx):
        e_val = signal[e_idx[0]]
        ax_main.scatter([eol_cycle], [e_val], marker="X", s=220, color="#1e1b4b", edgecolors="#dc2626", linewidth=1.5, zorder=6, label="QUEBRA DO PRODUTO (EOL)")
        ax_main.annotate(
            f"QUEBRA DO PRODUTO (EOL)\nCiclo {int(eol_cycle)}",
            xy=(eol_cycle, e_val),
            xytext=(max(20, eol_cycle - 40), e_val - (upper_hull.max() - lower_hull.min()) * 0.20),
            arrowprops=dict(arrowstyle="->", color="#1e1b4b", lw=1.6),
            fontsize=9.5, fontweight="bold", color="#1e1b4b",
            bbox=dict(boxstyle="round,pad=0.35", fc="#e0e7ff", ec="#3730a3", lw=1.2),
        )

    ax_main.set_ylabel(f"{sensor_name}{unit_str}", fontsize=10, fontweight="bold")
    ax_main.set_title(
        f"{title} - Motor {summary.get('unit_id', 'Holdout')}\n"
        f"Alarme no Ciclo {alarm_cycle} | Quebra no Ciclo {eol_cycle} | Lead Time: {lead_time} ciclos",
        fontsize=11, fontweight="bold",
    )
    ax_main.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.5)
    ax_main.grid(True, linestyle=":", alpha=0.5)
    ax_main.set_xlim(0, max_cycle + 5)

    # Amplitude modulation plot
    ax_amp.plot(cycles, amp, color="#0369a1", linewidth=1.6, label="Largura do Envelope Geométrico (Pico - Vale)")
    ax_amp.axhline(amp_thresh, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Limiar de Modulação ({amp_thresh:.2f})")
    ax_amp.fill_between(cycles, amp_thresh, np.maximum(amp, amp_thresh), where=(amp > amp_thresh), color="#ef4444", alpha=0.35)

    if alarm_cycle is not None and len(a_idx):
        ax_amp.scatter([alarm_cycle], [amp[a_idx[0]]], marker="*", s=200, color="#dc2626", zorder=5)
    if len(e_idx):
        ax_amp.scatter([eol_cycle], [amp[e_idx[0]]], marker="X", s=180, color="#1e1b4b", edgecolors="#dc2626", zorder=5)

    ax_amp.set_xlabel("Ciclo Operacional", fontsize=10, fontweight="bold")
    ax_amp.set_ylabel("Largura do Envelope", fontsize=10, fontweight="bold")
    ax_amp.grid(True, linestyle=":", alpha=0.5)
    ax_amp.legend(loc="upper left", frameon=True, fontsize=8.5)

    figure.tight_layout()
    return save_figure(figure, path)


def plot_actual_vs_predicted_with_envelope(
    y_true,
    y_pred,
    path: Path,
    title: str,
    value_label: str = "RUL (ciclos)",
    k_sigma: float = 3.0,
    cal_sigma: float | None = None,
) -> Path:
    """Plot actual-vs-predicted scatter with identity line and shaded prediction envelope corridor ('embrulho')."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    residuals = truth - prediction
    score = float(r2_score(truth, prediction))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    sigma = cal_sigma if cal_sigma is not None and cal_sigma > 0 else float(np.std(residuals))
    margin = k_sigma * sigma

    out_of_envelope = np.abs(residuals) > margin
    in_envelope_pct = float(np.mean(~out_of_envelope) * 100.0)
    out_pct = float(np.mean(out_of_envelope) * 100.0)

    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    padding = (upper - lower) * 0.05 or 1.0

    figure, ax = plt.subplots(figsize=(7.5, 7.0))
    grid = np.linspace(lower - padding, upper + padding, 200)

    # Shaded Envelope Corridor along identity line
    ax.fill_between(
        grid, grid - margin, grid + margin,
        color="#cbd5e1", alpha=0.55, label=f"Embrulho de Tolerância (±{k_sigma:.1f}σ: ±{margin:.1f} {value_label})",
    )
    ax.plot(grid, grid + margin, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax.plot(grid, grid - margin, color="#94a3b8", linestyle=":", linewidth=1.1)
    ax.plot(grid, grid, color="#1e293b", linestyle="--", linewidth=1.6, label="Paridade Perfeita (y = x)")

    # Scatter nominal vs anomalous
    ax.scatter(
        truth[~out_of_envelope], prediction[~out_of_envelope],
        s=16, alpha=0.45, color="#2563eb", edgecolors="none", label=f"Nominal no Embrulho ({in_envelope_pct:.1f}%)",
    )
    if np.any(out_of_envelope):
        ax.scatter(
            truth[out_of_envelope], prediction[out_of_envelope],
            s=28, alpha=0.85, color="#ef4444", edgecolors="#991b1b", linewidth=0.5,
            label=f"Anomalia / Fora do Embrulho ({out_pct:.1f}%, n={int(np.sum(out_of_envelope))})",
        )

    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel(f"Real {value_label}", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Predito {value_label}", fontsize=10, fontweight="bold")
    ax.set_title(
        f"{title}\nR² = {score:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}\n"
        f"Confiabilidade do Embrulho: {in_envelope_pct:.1f}% dentro da banda",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.5)

    figure.tight_layout()
    return save_figure(figure, path)


# ============================================================================
# generate_thesis_figures
# ============================================================================

#!/usr/bin/env python3




def compute_acf(x: np.ndarray, nlags: int = 25) -> np.ndarray:
    """Compute sample autocorrelation function using numpy."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    var = np.sum(x**2)
    if var == 0:
        return np.ones(nlags + 1)
    r = np.correlate(x, x, mode="full")
    mid = len(x) - 1
    acf_vals = r[mid:mid + nlags + 1] / var
    return acf_vals

# Formatting
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8

PALETTE = {
    "bg_dark": "#0b0f19",
    "surface": "#131c2e",
    "card": "#1e293b",
    "border": "#334155",
    "cyan": "#06b6d4",
    "cyan_light": "#38bdf8",
    "amber": "#f59e0b",
    "amber_light": "#fbbf24",
    "red_alert": "#ef4444",
    "red_dark": "#b91c1c",
    "emerald": "#10b981",
    "purple": "#8b5cf6",
    "gray_text": "#94a3b8",
    "white": "#f8fafc",
}

FIGURES_DIR = DEFAULT_PATHS.guide_dir("FD004") / "figures"
TABLES_DIR = DEFAULT_PATHS.guide_dir("FD004") / "tables"


def _thesis_style():
    """Light print style shared with monitoring_figures (tokens in viz_style.py)."""
    from pdm_cmaps.reporting.viz_style import apply_style
    apply_style()


def _first_persistent(mask: np.ndarray, cycles: np.ndarray, k: int, after: int) -> int | None:
    """First cycle > ``after`` that starts ``k`` consecutive True values (Western-Electric style persistence)."""
    run = 0
    for i, flag in enumerate(mask):
        run = run + 1 if (flag and cycles[i] > after) else 0
        if run >= k:
            return int(cycles[i - k + 1])
    return None


def figure_1_noise_vs_damage(output_dir: Path | None = None, healthy_cycle_limit: int = 50, k: int = 2) -> Path:
    """Figure 1: an isolated envelope breach (noise) vs the first sustained run of breaches, on the sample engine.

    Every cycle and value printed in the figure is computed from envelope_scored_trace_t50_h50.csv.
    """
    from pdm_cmaps.reporting.viz_style import INK, INK_2, LINE_W, SERIES, STATUS, style_axes
    _thesis_style()
    figures_dir = Path(output_dir) / "figures" if output_dir else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(output_dir) / "tables" if output_dir else TABLES_DIR
    trace = pd.read_csv(tables_dir / "envelope_scored_trace_t50_h50.csv").sort_values("cycle").reset_index(drop=True)
    unit = int(trace["unit_id"].iloc[0])
    cyc = trace["cycle"].to_numpy()
    out = trace["out_of_envelope"].to_numpy(dtype=bool)
    sustained = _first_persistent(out, cyc, k, healthy_cycle_limit)
    prev_ = np.r_[False, out[:-1]]
    next_ = np.r_[out[1:], False]
    isolated = np.flatnonzero(out & ~prev_ & ~next_ & (cyc > healthy_cycle_limit))
    if sustained is not None:
        before = isolated[cyc[isolated] < sustained]
        isolated = before if before.size else isolated
    iso_cycle = int(cyc[isolated[0]]) if isolated.size else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    panels = [(iso_cycle, "Violação isolada", "um ciclo fora do envelope, sem continuidade"),
              (sustained, f"Primeira sequência de {k}+ violações", "o padrão que a regra de persistência procura")]
    for ax, (center, title, subtitle) in zip(axes, panels):
        style_axes(ax, "y")
        if center is None:
            ax.text(0.5, 0.5, "não ocorreu neste motor", transform=ax.transAxes, ha="center", color=INK_2)
            ax.set_title(title)
            continue
        win = trace.loc[(trace["cycle"] >= center - 8) & (trace["cycle"] <= center + 12)]
        ax.fill_between(win["cycle"], win["envelope_lower"], win["envelope_upper"], color=SERIES[0], alpha=0.15, linewidth=0)
        ax.plot(win["cycle"], win["prediction"], color=SERIES[0], lw=LINE_W, linestyle=(0, (4, 2)))
        ax.plot(win["cycle"], win["T50"], color=INK, lw=1.4, marker="o", markersize=4)
        hit = win.loc[win["out_of_envelope"]]
        ax.plot(hit["cycle"], hit["T50"], linestyle="none", marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=STATUS[3], markeredgewidth=2)
        row = trace.loc[trace["cycle"] == center].iloc[0]
        excess = float(max(row["T50"] - row["envelope_upper"], row["envelope_lower"] - row["T50"]))
        ax.set_title(f"{title} (ciclo {center})\n{subtitle}; excesso de {excess:.2f} °R".replace(".", ","), fontsize=10)
        ax.set_xlabel("Ciclo")
        ax.set_ylabel("T50 (°R)")
    handles = [Line2D([], [], color=INK, lw=1.4, marker="o", markersize=4), Line2D([], [], color=SERIES[0], lw=LINE_W, linestyle=(0, (4, 2))),
               Patch(color=SERIES[0], alpha=0.15), Line2D([], [], linestyle="none", marker="o", markersize=10, markerfacecolor="none", markeredgecolor=STATUS[3], markeredgewidth=2)]
    fig.legend(handles, ["T50 medido", "Linha de base saudável (sensor virtual)", "Envelope ±3σ", "Fora do envelope"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Ativo {unit} (FD004, motor de teste): ruído isolado × violação sustentada", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out_path = figures_dir / "thesis_anomaly_noise_vs_damage.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def figure_2_physical_cascade(output_dir: Path | None = None, healthy_cycle_limit: int = 50, k: int = 2) -> Path:
    """Figure 2: residuals of four sensors on the sample engine and the first persistent envelope alarm of each.

    Alarm cycles are computed (k consecutive envelope breaches after the healthy window). The order is a
    temporal sequence on ONE engine, not evidence of causality.
    """
    from pdm_cmaps.reporting.viz_style import INK, MUTED, SERIES, STATUS, style_axes
    _thesis_style()
    figures_dir = Path(output_dir) / "figures" if output_dir else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(output_dir) / "tables" if output_dir else TABLES_DIR
    meta = [("nc", "Nc, rotação do núcleo (rpm)"), ("p30", "P30, pressão de saída do HPC (psia)"),
            ("bpr", "BPR, razão de bypass"), ("t50", "T50, temperatura de saída da LPT (°R)")]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9.5), sharex=True)
    rows, unit, eol = [], None, None
    for ax, (key, label) in zip(axes, meta):
        style_axes(ax, "y")
        df = pd.read_csv(tables_dir / f"envelope_scored_trace_{key}_h50.csv").sort_values("cycle")
        unit, eol = int(df["unit_id"].iloc[0]), int(df["cycle"].max())
        cyc = df["cycle"].to_numpy()
        res = df["residual"].to_numpy()
        half = (df["envelope_upper"] - df["prediction"]).to_numpy()
        lo = (df["envelope_lower"] - df["prediction"]).to_numpy()
        ax.fill_between(cyc, lo, half, color=SERIES[0], alpha=0.12, linewidth=0)
        ax.plot(cyc, res, color=SERIES[0], lw=1.2)
        ax.axhline(0, color=MUTED, lw=0.8)
        alarm = _first_persistent(df["out_of_envelope"].to_numpy(dtype=bool), cyc, k, healthy_cycle_limit)
        if alarm is not None:
            ax.axvline(alarm, color=STATUS[3], lw=1.6, linestyle=(0, (4, 2)))
            ax.text(alarm + 3, ax.get_ylim()[1] * 0.85, f"1º alarme persistente: ciclo {alarm}", color=INK, fontsize=9, va="top")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Resíduo")
        rows.append({"sensor": key, "first_persistent_alarm": alarm})
    axes[-1].set_xlabel(f"Ciclo (fim de vida no ciclo {eol})")
    names = {"nc": "Nc", "p30": "P30", "bpr": "BPR", "t50": "T50"}
    order = " → ".join(f"{names[r['sensor']]} ({r['first_persistent_alarm']})" for r in sorted(rows, key=lambda r: (r["first_persistent_alarm"] is None, r["first_persistent_alarm"] or 0)))
    fig.suptitle(f"Ativo {unit}: resíduo de quatro sensores e envelope ±3σ\nOrdem do primeiro alarme: {order}\n(um motor; é uma sequência no tempo, não prova causalidade)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = figures_dir / "thesis_multisensor_physical_cascade.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(tables_dir / "thesis_multisensor_first_alarms.csv", index=False)
    print(f"Saved: {out_path}")
    return out_path


def figure_3_autocorrelation_evidence(output_dir: Path | None = None, healthy_cycle_limit: int = 50, degraded_window: int = 50) -> Path:
    """Figure 3: residual distribution and ACF in the healthy window vs the last cycles of life (sample engine).

    Windows match hypothesis tests H2/H3: healthy = cycle <= H; degraded = last ``degraded_window`` cycles.
    """
    from pdm_cmaps.reporting.viz_style import INK, MUTED, SERIES, style_axes
    _thesis_style()
    figures_dir = Path(output_dir) / "figures" if output_dir else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(output_dir) / "tables" if output_dir else TABLES_DIR
    trace = pd.read_csv(tables_dir / "envelope_scored_trace_t50_h50.csv").sort_values("cycle")
    unit, eol = int(trace["unit_id"].iloc[0]), int(trace["cycle"].max())
    start_deg = max(healthy_cycle_limit, eol - degraded_window)
    healthy = trace.loc[trace["cycle"] <= healthy_cycle_limit, "residual"].dropna().to_numpy()
    degraded = trace.loc[trace["cycle"] > start_deg, "residual"].dropna().to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    lags = 15
    stats_rows = []
    for col, (vals, name, color) in enumerate([(healthy, f"fase saudável (ciclos 1–{healthy_cycle_limit})", SERIES[0]),
                                               (degraded, f"fim de vida (ciclos {start_deg + 1}–{eol})", SERIES[1])]):
        ax = axes[0, col]
        style_axes(ax, "y")
        bins = np.linspace(min(healthy.min(), degraded.min()), max(healthy.max(), degraded.max()), 26)
        ax.hist(vals, bins=bins, color=color, edgecolor="#ffffff", linewidth=1.0)
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        ax.axvline(mu, color=INK, lw=1.2)
        ax.set_title(f"Resíduo do T50, {name}\nmédia {mu:.1f} °R, desvio {sd:.1f} °R, n = {len(vals)}".replace(".", ","), fontsize=10)
        ax.set_xlabel("Resíduo (°R)")
        ax.set_ylabel("Ciclos")
        ax = axes[1, col]
        style_axes(ax, "y")
        acf = compute_acf(vals, nlags=lags)
        conf = 1.96 / np.sqrt(len(vals))
        ax.bar(range(1, lags + 1), acf[1:], color=color, width=0.6)
        ax.axhspan(-conf, conf, color=MUTED, alpha=0.15, linewidth=0)
        ax.axhline(0, color=MUTED, lw=0.8)
        outside = int(np.sum(np.abs(acf[1:]) > conf))
        ax.set_title(f"Autocorrelação, {name}\nr₁ = {acf[1]:.2f}; {outside} de {lags} defasagens fora da faixa de 95 %".replace(".", ","), fontsize=10)
        ax.set_xlabel("Defasagem (ciclos)")
        ax.set_ylabel("r_k")
        ax.set_ylim(-0.4, 1.0)
        stats_rows.append({"window": name, "n": len(vals), "mean": mu, "std": sd, "r1": float(acf[1]), "lags_outside_95": outside})
    fig.suptitle(f"Ativo {unit}: o resíduo muda de nível e ganha memória no fim de vida\n(um motor; o teste de frota é o H3, Ljung–Box por motor + Fisher)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = figures_dir / "thesis_residual_autocorrelation_evidence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(tables_dir / "thesis_residual_window_stats.csv", index=False)
    print(f"Saved: {out_path}")
    return out_path


def figure_4_fleet_persistence_tradeoff(output_dir: Path | None = None, healthy_cycle_limit: int = 50, k_values=(1, 2, 3, 5)) -> Path:
    """Figure 4: effect of the persistence rule k on the fleet, COMPUTED from virtual_sensor_residuals_t50.csv.

    A cycle is a breach when |z| >= 3 and EWMA(|z|) >= UCL_t (same rule as StatisticalAnomalyDetector).
    For each k: detection = % engines with k consecutive breaches after the healthy window; false alarm =
    % engines with k consecutive breaches INSIDE the healthy window; lead time = EOL - first alarm.
    """
    from pdm_cmaps.reporting.viz_style import INK_2, MARKER_PT, SERIES, SURFACE, style_axes
    _thesis_style()
    figures_dir = Path(output_dir) / "figures" if output_dir else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(output_dir) / "tables" if output_dir else TABLES_DIR
    res = pd.read_csv(tables_dir / "virtual_sensor_residuals_t50.csv").sort_values(["unit_id", "cycle"])
    res["breach"] = (res["z_score"].abs() >= 3.0) & (res["ewma_abs_z"] >= res["ewma_ucl"])
    rows = []
    for k in k_values:
        det, fa, leads = 0, 0, []
        engines = 0
        for _, eng in res.groupby("unit_id"):
            engines += 1
            cyc = eng["cycle"].to_numpy()
            b = eng["breach"].to_numpy(dtype=bool)
            alarm = _first_persistent(b, cyc, k, healthy_cycle_limit)
            healthy_alarm = _first_persistent(b & (cyc <= healthy_cycle_limit), np.where(cyc <= healthy_cycle_limit, cyc, -1), k, 0)
            if alarm is not None:
                det += 1
                leads.append(int(cyc.max()) - alarm)
            if healthy_alarm is not None:
                fa += 1
        rows.append({"k": k, "engines": engines, "detection_pct": 100 * det / engines,
                     "healthy_false_alarm_engines_pct": 100 * fa / engines,
                     "mean_lead_time_cycles": float(np.mean(leads)) if leads else float("nan")})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    specs = [("detection_pct", "Detecção (% dos motores)"), ("healthy_false_alarm_engines_pct", "Motores com alarme na fase saudável (%)"),
             ("mean_lead_time_cycles", "Antecedência média (ciclos)")]
    for ax, (col, title) in zip(axes, specs):
        style_axes(ax, "y")
        ax.plot(table["k"], table[col], color=SERIES[0], lw=2.0, marker="o", markersize=MARKER_PT, markeredgecolor=SURFACE, markeredgewidth=2)
        for x, y in zip(table["k"], table[col]):
            ax.annotate(f"{y:.1f}".replace(".", ","), (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(k_values))
        ax.set_xlabel("k (violações consecutivas exigidas)")
        ax.margins(y=0.25)
    n = int(table["engines"].iloc[0])
    fig.suptitle(f"Regra de persistência na frota de teste ({n} motores, semente desta execução): k maior troca antecedência por menos alarmes",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = figures_dir / "thesis_persistence_tradeoff_fleet.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    table.to_csv(tables_dir / "thesis_persistence_tradeoff_fleet.csv", index=False)
    print(f"Saved: {out_path}")
    return out_path


def main() -> int:
    print("[thesis_figures] Generating high-resolution thesis proof figures...")
    figure_1_noise_vs_damage()
    figure_2_physical_cascade()
    figure_3_autocorrelation_evidence()
    figure_4_fleet_persistence_tradeoff()
    print("[thesis_figures] All 4 thesis figures successfully generated!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
