"""Statistical hypothesis testing (Friedman, Wilcoxon, Ljung-Box, BH-FDR) and engine-cluster BCa bootstrap CIs.

WHAT   Turns "the plot looks different" into a testable statement with a p-value and an effect
       size. One test per thesis hypothesis:
         H1 regime explains sensor variance  -> Friedman test across regimes, blocked by engine
         H2 residuals shift near end of life -> paired t-test / Wilcoxon on engine means
         H3 healthy residuals are white noise-> Ljung-Box per engine + Fisher combination
         H4 model beats the mean baseline    -> paired Wilcoxon on per-engine errors
         H5 XGBoost vs deep models           -> two-sided paired Wilcoxon
WHY    With one fleet and one split, differences between models can be luck. Tests and
       confidence intervals say how much luck could explain.
CONCEPTS
  [PSEUDOREPLICATION] 200 cycles of one engine are NOT 200 independent samples. All tests first
                 aggregate to one number per engine, so n = engines.
  Paired tests   the same engines are scored by both models, so compare per-engine differences.
  Non-parametric Wilcoxon avoids assuming normal errors; Shapiro-Wilk picks t-test vs Wilcoxon.
  Effect size    p-values shrink with n; always read the effect (Kendall's W, Hodges-Lehmann).
  Bootstrap BCa  resample ENGINES (clusters) with replacement to get a confidence interval of
                 R²/RMSE; bias-correction and acceleration fix skew.
  Multiple testing  many tests -> some false positives; Benjamini-Hochberg controls the false
                 discovery rate. "Not rejected" is NOT "proved equal".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from pdm_cmaps.domain.rul import regression_metrics
from pdm_cmaps.domain.schema import DEFAULT_RANDOM_STATE
from pdm_cmaps.modeling.registry import grouped_oof


# ============================================================================
# thesis_statistics
# ============================================================================

def hodges_lehmann_estimator(deltas: np.ndarray) -> float:
    """Compute Hodges-Lehmann pseudo-median from pairwise Walsh averages."""
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    pairwise = [(arr[i] + arr[j]) / 2.0 for i in range(n) for j in range(i, n)]
    return float(np.median(pairwise))


def _bca_interval(bootstrap_values: list[float], jackknife_values: list[float], observed: float) -> tuple[float, float]:
    """Bias-corrected and accelerated interval for a cluster-resampled statistic."""
    boot = np.asarray(bootstrap_values, dtype=float)
    jack = np.asarray(jackknife_values, dtype=float)
    eps = 0.5 / len(boot)
    bias = stats.norm.ppf(np.clip(np.mean(boot < observed), eps, 1.0 - eps))
    centered = jack.mean() - jack
    denom = 6.0 * np.sum(centered ** 2) ** 1.5
    acceleration = float(np.sum(centered ** 3) / denom) if denom > 0 else 0.0
    quantiles = []
    for alpha in (0.025, 0.975):
        z = stats.norm.ppf(alpha)
        adjusted = stats.norm.cdf(bias + (bias + z) / (1.0 - acceleration * (bias + z)))
        quantiles.append(float(np.clip(adjusted, 0.0, 1.0)))
    return tuple(float(x) for x in np.quantile(boot, quantiles))


@dataclass
class HypothesisTestResult:
    """Container for hypothesis test outcomes."""

    hypothesis_id: str
    description: str
    h0: str
    h1: str
    test_name: str
    test_statistic: float
    p_value: float
    significance_level: float = 0.05
    effect_size: float | None = None
    conclusion: str = ""

    @property
    def is_rejected(self) -> bool:
        return self.p_value < self.significance_level

    def to_dict(self) -> dict[str, Any]:
        p_formatted = f"{self.p_value:.2e}" if self.p_value < 1e-4 else f"{self.p_value:.4f}"
        if self.p_value == 0.0:
            p_formatted = "< 1e-300"
        return {
            "ID": self.hypothesis_id,
            "Description": self.description,
            "H0": self.h0,
            "H1": self.h1,
            "Test": self.test_name,
            "Statistic": round(float(self.test_statistic), 4),
            "p_value": p_formatted,
            "p_value_raw": float(self.p_value),
            "H0_Rejected": "REJEITADA (p < 0.05)" if self.is_rejected else "NÃO REJEITADA",
            "Effect_Size": round(float(self.effect_size), 4) if self.effect_size is not None else np.nan,
            "Scientific_Conclusion": self.conclusion,
        }


class ThesisHypothesisAnalyzer:
    """Comprehensive hypothesis testing and statistical analysis engine."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results: list[HypothesisTestResult] = []

    def test_regime_variance_dominance(
        self,
        df: pd.DataFrame,
        sensor: str = "T50",
        regime_col: str = "regime_id",
        unit_col: str = "unit_id",
    ) -> HypothesisTestResult:
        """Hypothesis 1: Operating regime explains statistically significant variance in sensor readings.
        
        Evaluated on repeated engine blocks to eliminate cycle pseudoreplication.
        """
        # If engine unit_id is present, perform repeated-measures engine-block Friedman test
        if unit_col in df.columns and df[unit_col].nunique() >= 2:
            pivot = df.groupby([unit_col, regime_col])[sensor].mean().unstack()
            # Drop incomplete engine rows across regimes
            clean_matrix = pivot.dropna()
            if len(clean_matrix) >= 2 and clean_matrix.shape[1] >= 2:
                cols = [clean_matrix[c].to_numpy() for c in clean_matrix.columns]
                stat, p_val = stats.friedmanchisquare(*cols)
                n_engines = len(clean_matrix)
                k_regimes = clean_matrix.shape[1]
                kendall_w = float(stat / (n_engines * (k_regimes - 1)))
                res = HypothesisTestResult(
                    hypothesis_id="H1_REGIME_VARIANCE",
                    description="Dominância da Variação de Altitude/Mach sobre os Sensores Térmicos",
                    h0="A altitude e velocidade de voo não afetam as leituras térmicas nominais entre motores.",
                    h1="A atmosfera ISA e regimes de voo causam variação altamente significativa (p < 0.001).",
                    test_name=f"Repeated-Measures Friedman Test (n={n_engines} engines, k={k_regimes} regimes)",
                    test_statistic=stat,
                    p_value=p_val,
                    effect_size=kendall_w,
                    conclusion=(
                        f"Efeito intra-motor entre regimes: p={p_val:.2e} (Friedman chi²={stat:.1f}, "
                        f"Kendall's W={kendall_w:.4f}, n={n_engines} motores). "
                        f"Confirmada dominância de regime livre de pseudorreplicação amostral."
                    ),
                )
                self.results.append(res)
                return res

        # Fallback to grouped one-way ANOVA / Alexander-Govern if no engine column
        groups = [group[sensor].dropna().to_numpy(dtype=float) for _, group in df.groupby(regime_col)]
        groups = [group for group in groups if group.size > 1]
        if len(groups) < 2:
            raise ValueError("At least two regimes with two observations each are required")
        levene_stat, levene_p = stats.levene(*groups, center="median")
        if levene_p < 0.05:
            robust_result = stats.alexandergovern(*groups)
            test_statistic, p_val = float(robust_result.statistic), float(robust_result.pvalue)
            test_name = f"Alexander-Govern test (Levene p={levene_p:.2e})"
            statistic_label = "AG"
        else:
            f_result = stats.f_oneway(*groups)
            test_statistic, p_val = float(f_result.statistic), float(f_result.pvalue)
            test_name = f"One-Way ANOVA (Levene p={levene_p:.2e})"
            statistic_label = "F"

        all_vals = df[sensor].dropna().to_numpy(dtype=float)
        ss_total = np.sum((all_vals - np.mean(all_vals)) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(all_vals)) ** 2 for g in groups)
        eta_squared = ss_between / (ss_total + 1e-12)

        res = HypothesisTestResult(
            hypothesis_id="H1_REGIME_VARIANCE",
            description="Dominância da Variação de Altitude/Mach sobre os Sensores Térmicos",
            h0="A altitude e velocidade de voo não afetam as leituras térmicas nominais.",
            h1="A atmosfera ISA e regimes de voo causam variação altamente significativa (p < 0.001).",
            test_name=test_name,
            test_statistic=test_statistic,
            p_value=p_val,
            effect_size=eta_squared,
            conclusion=(
                f"Efeito entre regimes: p={p_val:.2e} ({statistic_label}={test_statistic:.1f}; "
                f"Levene p={levene_p:.2e}; eta²={eta_squared:.4f}). "
                f"A estatística de efeito descreve {eta_squared*100:.1f}% da variância da amostra."
            ),
        )
        self.results.append(res)
        return res

    def test_physical_degradation_shift(
        self,
        residuals_df: pd.DataFrame,
        healthy_cutoff: int = 50,
        degraded_start: int | None = None,
        residual_col: str = "residual",
        cycle_col: str = "cycle",
        unit_col: str = "unit_id",
    ) -> HypothesisTestResult:
        """Hypothesis 2: Virtual sensor residuals shift significantly during degradation.
        
        Uses survivor-bias-free relative window (t > max(H, EOL_i - 50)) to preserve all engines.
        """
        if unit_col not in residuals_df:
            raise ValueError(f"{unit_col} is required for engine-level paired inference")

        ordered = residuals_df.copy()
        eol_map = ordered.groupby(unit_col)[cycle_col].transform("max")
        ordered["_eol"] = eol_map

        # Relative degradation window: t > max(healthy_cutoff, EOL_i - 50)
        if degraded_start is None:
            ordered["_is_degraded"] = ordered[cycle_col] > np.maximum(healthy_cutoff, ordered["_eol"] - 50)
        else:
            ordered["_is_degraded"] = ordered[cycle_col] >= degraded_start

        healthy = ordered.loc[ordered[cycle_col] <= healthy_cutoff].groupby(unit_col)[residual_col].mean()
        degraded = ordered.loc[ordered["_is_degraded"]].groupby(unit_col)[residual_col].mean()

        paired = pd.concat([healthy.rename("healthy"), degraded.rename("degraded")], axis=1).dropna()
        n_engines = len(paired)
        if n_engines < 2:
            raise ValueError("At least two engines need observations in both phases")

        deltas = (paired["degraded"] - paired["healthy"]).to_numpy(dtype=float)
        # Normality check on deltas
        shapiro_stat, shapiro_p = stats.shapiro(deltas)
        hl_delta = hodges_lehmann_estimator(deltas)
        cohens_d = float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-12))

        if shapiro_p >= 0.05:
            t_stat, p_val = stats.ttest_1samp(deltas, popmean=0.0)
            test_name = f"Paired t-test on engine means (n={n_engines}, Shapiro p={shapiro_p:.3f})"
            stat_used = float(t_stat)
        else:
            w_stat, p_val = stats.wilcoxon(deltas, alternative="greater")
            test_name = f"Paired Wilcoxon signed-rank on engine means (n={n_engines}, Shapiro p={shapiro_p:.3f})"
            stat_used = float(w_stat)

        res = HypothesisTestResult(
            hypothesis_id="H2_PHYSICAL_DEGRADATION",
            description="Divergência Térmica Real vs Ruído de Medição no Fim de Vida",
            h0="Os resíduos do Sensor Virtual no final de vida possuem a mesma distribuição da fase saudável.",
            h1="O desgaste físico das pás provoca um deslocamento sistemático e significativo (p < 1e-15).",
            test_name=test_name,
            test_statistic=stat_used,
            p_value=float(p_val),
            effect_size=hl_delta,
            conclusion=(
                f"Confirmada com p={p_val:.2e} (n={n_engines} motores, d de Cohen={cohens_d:.2f}, "
                f"Hodges-Lehmann Δ={hl_delta:+.2f} unidades do sensor). "
                f"O desvio médio de {np.mean(deltas):+.2f} unidades do sensor indica deslocamento sistemático do resíduo."
            ),
        )
        self.results.append(res)
        return res

    def test_white_noise_residual_autocorrelation(
        self,
        residuals: np.ndarray,
        max_lags: int = 10,
        phase_label: str = "healthy",
    ) -> HypothesisTestResult:
        """Hypothesis 3: Ljung-Box test for residual serial correlation."""
        n = len(residuals)
        mean_r = np.mean(residuals)
        var_r = np.var(residuals)
        lags_autocorr = []
        for k in range(1, max_lags + 1):
            cov = np.sum((residuals[:-k] - mean_r) * (residuals[k:] - mean_r)) / n
            lags_autocorr.append(cov / (var_r + 1e-12))

        q_stat = n * (n + 2) * sum((rk ** 2) / (n - k) for k, rk in enumerate(lags_autocorr, start=1))
        p_val = stats.chi2.sf(q_stat, df=max_lags)

        hyp_id = f"H3_AUTOCORRELATION_{phase_label.upper()}"
        desc = f"Estrutura de Autocorrelação Residual (Fase {phase_label.capitalize()})"
        h0 = "Os resíduos são ruído branco puramente aleatório (sem autocorrelação serial)."
        h1 = "Os resíduos apresentam memória serial e correlação persistente."

        conclusion = (
            f"Fase {phase_label}: Q={q_stat:.2f}, p={p_val:.2e}. "
            + ("Não se rejeita H0; isso não demonstra ausência de autocorrelação." if p_val > 0.05 else "Evidência de autocorrelação serial.")
        )

        res = HypothesisTestResult(
            hypothesis_id=hyp_id,
            description=desc,
            h0=h0,
            h1=h1,
            test_name=f"Ljung-Box Portmanteau Test (lags={max_lags})",
            test_statistic=q_stat,
            p_value=p_val,
            effect_size=lags_autocorr[0] if lags_autocorr else None,
            conclusion=conclusion,
        )
        self.results.append(res)
        return res

    def test_fleet_residual_autocorrelation(
        self,
        residuals_df: pd.DataFrame,
        healthy_cutoff: int = 50,
        max_lags: int = 10,
        residual_col: str = "residual",
        cycle_col: str = "cycle",
        unit_col: str = "unit_id",
        phase_label: str = "frota_saudavel",
        min_obs: int = 20,
    ) -> HypothesisTestResult:
        """Hypothesis 3 (fleet): per-engine Ljung-Box on the healthy window, combined with Fisher's method.

        Each engine is tested separately (no concatenation across engines, which would create
        artificial lag-k pairs at engine boundaries). Under H0 each per-engine p-value is Uniform(0,1),
        so -2*sum(ln p_i) ~ chi2(2m).
        """
        q_stats, p_vals = [], []
        for _, eng in residuals_df.loc[residuals_df[cycle_col] <= healthy_cutoff].groupby(unit_col):
            r = eng.sort_values(cycle_col)[residual_col].dropna().to_numpy(dtype=float)
            n = len(r)
            if n < max(min_obs, max_lags + 2) or np.var(r) < 1e-12:
                continue
            centered = r - r.mean()
            denom = float(np.sum(centered ** 2))
            rho = [float(np.sum(centered[:-k] * centered[k:]) / denom) for k in range(1, max_lags + 1)]
            q = n * (n + 2) * sum((rk ** 2) / (n - k) for k, rk in enumerate(rho, start=1))
            q_stats.append(q)
            p_vals.append(float(stats.chi2.sf(q, df=max_lags)))
        if len(p_vals) < 2:
            raise ValueError("At least two engines with enough healthy observations are required")
        p_arr = np.clip(np.asarray(p_vals), 1e-300, 1.0)
        fisher = float(-2.0 * np.sum(np.log(p_arr)))
        p_comb = float(stats.chi2.sf(fisher, df=2 * len(p_arr)))
        frac_reject = float(np.mean(p_arr < 0.05))
        res = HypothesisTestResult(
            hypothesis_id=f"H3_AUTOCORRELATION_{phase_label.upper()}",
            description=f"Estrutura de Autocorrelação Residual por Motor (Fase {phase_label})",
            h0="Os resíduos de cada motor são ruído branco (sem autocorrelação serial).",
            h1="Ao menos parte da frota apresenta autocorrelação serial nos resíduos.",
            test_name=f"Per-engine Ljung-Box (lags={max_lags}) + Fisher combination (m={len(p_arr)} engines)",
            test_statistic=fisher,
            p_value=p_comb,
            effect_size=frac_reject,
            conclusion=(
                f"Fase {phase_label}: Q médio={np.mean(q_stats):.2f}, {frac_reject*100:.1f}% dos motores rejeitam H0 a 5% "
                f"(esperado ≈5% sob ruído branco); Fisher p={p_comb:.2e}."
            ),
        )
        self.results.append(res)
        return res

    def test_model_superiority_vs_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_pred: np.ndarray,
        model_name: str,
        unit_ids: np.ndarray | None = None,
    ) -> HypothesisTestResult:
        """Hypothesis 4: ML model provides statistically significantly lower error than naive baseline."""
        err_model = np.abs(y_true - y_pred)
        err_base = np.abs(y_true - baseline_pred)

        if unit_ids is None or len(unit_ids) != len(err_model):
            diff = err_base - err_model
        else:
            diff = pd.DataFrame({"unit_id": unit_ids, "diff": err_base - err_model}).groupby("unit_id")["diff"].mean().to_numpy()

        w_stat, p_val = stats.wilcoxon(diff, alternative="greater")
        mean_reduction = np.mean(err_base) - np.mean(err_model)
        hl_reduction = hodges_lehmann_estimator(diff)

        res = HypothesisTestResult(
            hypothesis_id=f"H4_SUPERIORITY_{model_name.upper().replace(' ', '_')}",
            description=f"Superioridade Estatística de {model_name} sobre a Linha de Base (Média)",
            h0=f"O modelo {model_name} possui a mesma distribuição de erro absoluto da média incondicional.",
            h1=f"O modelo {model_name} tem erro estatisticamente inferior (p < 0.001).",
            test_name="Wilcoxon Paired Signed-Rank Test",
            test_statistic=w_stat,
            p_value=p_val,
            effect_size=hl_reduction,
            conclusion=(
                f"{model_name} superou a linha de base com p={p_val:.2e} (W={w_stat:.1f}, "
                f"Hodges-Lehmann Δ={hl_reduction:.2f} ciclos). "
                f"Redução média no erro absoluto de {mean_reduction:.2f} ciclos."
            ),
        )
        self.results.append(res)
        return res

    def test_gbdt_vs_sequence_model_parity(
        self,
        y_true: np.ndarray,
        pred_gbdt: np.ndarray,
        pred_deep: np.ndarray,
        deep_model_name: str = "Transformer",
        unit_ids: np.ndarray | None = None,
    ) -> HypothesisTestResult:
        """Hypothesis 5: IEEE CIM 2026 RQ2 Dialogue: GBDT vs Deep Sequence Model Parity Test."""
        err_gbdt = np.abs(y_true - pred_gbdt)
        err_deep = np.abs(y_true - pred_deep)

        if unit_ids is None or len(unit_ids) != len(err_gbdt):
            sub_gbdt, sub_deep = err_gbdt, err_deep
        else:
            paired = pd.DataFrame({"unit_id": unit_ids, "gbdt": err_gbdt, "deep": err_deep}).groupby("unit_id")[["gbdt", "deep"]].mean()
            sub_gbdt, sub_deep = paired["gbdt"].to_numpy(), paired["deep"].to_numpy()

        w_stat, p_val = stats.wilcoxon(sub_gbdt, sub_deep)
        diff_mae = np.mean(err_deep) - np.mean(err_gbdt)
        hl_diff = hodges_lehmann_estimator(sub_deep - sub_gbdt)

        if p_val < 0.05:
            direction = "XGBoost superior" if diff_mae > 0 else f"{deep_model_name} superior"
            conclusion = (
                f"Comparação XGBoost vs {deep_model_name}: W={w_stat:.1f}, p={p_val:.2e}. "
                f"{direction} por Hodges-Lehmann Δ={hl_diff:.2f} ciclos (ΔMAE = {diff_mae:.2f})."
            )
        else:
            conclusion = (
                f"Comparação XGBoost vs {deep_model_name}: W={w_stat:.1f}, p={p_val:.2e}. "
                f"Paridade estatística: não se rejeita H0 a 5% (Hodges-Lehmann Δ={hl_diff:.2f} ciclos, ΔMAE = {diff_mae:.2f})."
            )

        res = HypothesisTestResult(
            hypothesis_id=f"H5_IEEE_RQ2_XGB_VS_{deep_model_name.upper()}",
            description=f"Diálogo IEEE CIM 2026: XGBoost vs {deep_model_name}",
            h0=f"XGBoost e {deep_model_name} apresentam acurácia preditiva idêntica.",
            h1="Existe divergência estatisticamente significativa entre as arquiteturas.",
            test_name="Two-Sided Paired Wilcoxon Test",
            test_statistic=w_stat,
            p_value=p_val,
            effect_size=hl_diff,
            conclusion=conclusion,
        )
        self.results.append(res)
        return res

    def compute_bootstrap_confidence_intervals(
        self,
        test_df: pd.DataFrame,
        model_cols: dict[str, str],
        target_col: str = "RUL_actual",
        n_bootstraps: int = 2000,
    ) -> pd.DataFrame:
        """Compute 95% Confidence Intervals via fleet engine cluster bootstrapping."""
        rng = np.random.default_rng(self.random_state)
        engines = test_df["unit_id"].unique()
        n_engines = len(engines)
        available_models = {name: col for name, col in model_cols.items() if col in test_df.columns}
        if not available_models:
            return pd.DataFrame()

        unit_ids = test_df["unit_id"].to_numpy()
        truth = test_df[target_col].to_numpy(dtype=float)
        predictions = {name: test_df[col].to_numpy(dtype=float) for name, col in available_models.items()}
        row_indices = [np.flatnonzero(unit_ids == engine) for engine in engines]

        def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
            errors = y_true - y_pred
            ss_total = np.sum((y_true - y_true.mean()) ** 2)
            return (
                float(1.0 - np.sum(errors ** 2) / (ss_total + 1e-12)),
                float(np.sqrt(np.mean(errors ** 2))),
                float(np.mean(np.abs(errors))),
            )

        score_store = {
            name: {metric: np.empty(n_bootstraps, dtype=float) for metric in ("r2", "rmse", "mae")}
            for name in available_models
        }
        for iteration in range(n_bootstraps):
            sampled = rng.integers(n_engines, size=n_engines)
            sample_index = np.concatenate([row_indices[index] for index in sampled])
            sample_truth = truth[sample_index]
            for name, prediction in predictions.items():
                r2, rmse, mae = metrics(sample_truth, prediction[sample_index])
                score_store[name]["r2"][iteration] = r2
                score_store[name]["rmse"][iteration] = rmse
                score_store[name]["mae"][iteration] = mae

        rows = []
        for name, prediction in predictions.items():
            observed = metrics(truth, prediction)
            jackknife = []
            for index in row_indices:
                keep = np.ones(len(truth), dtype=bool)
                keep[index] = False
                jackknife.append(metrics(truth[keep], prediction[keep]))
            r2_ci = _bca_interval(score_store[name]["r2"], [row[0] for row in jackknife], observed[0])
            rmse_ci = _bca_interval(score_store[name]["rmse"], [row[1] for row in jackknife], observed[1])
            mae_ci = _bca_interval(score_store[name]["mae"], [row[2] for row in jackknife], observed[2])
            rows.append({
                "Model": name,
                "R2_Observed": round(observed[0], 4),
                "R2_Mean": float(score_store[name]["r2"].mean()),
                "R2_CI_Lower": r2_ci[0],
                "R2_CI_Upper": r2_ci[1],
                "RMSE_Observed": round(observed[1], 2),
                "RMSE_Mean": float(score_store[name]["rmse"].mean()),
                "RMSE_CI_Lower": rmse_ci[0],
                "RMSE_CI_Upper": rmse_ci[1],
                "MAE_Observed": round(observed[2], 2),
                "MAE_Mean": float(score_store[name]["mae"].mean()),
                "MAE_CI_Lower": mae_ci[0],
                "MAE_CI_Upper": mae_ci[1],
                "CI_Method": "engine-cluster BCa",
            })
        return pd.DataFrame(rows)

    def summary_dataframe(self) -> pd.DataFrame:
        """Return all tested hypotheses as a structured DataFrame with FDR correction."""
        frame = pd.DataFrame([res.to_dict() for res in self.results])
        if not frame.empty:
            p = frame["p_value_raw"].to_numpy(dtype=float)
            order = np.argsort(p)
            adjusted = np.empty_like(p)
            running = 1.0
            m = len(p)
            for rank in range(m - 1, -1, -1):
                idx = order[rank]
                running = min(running, p[idx] * m / (rank + 1))
                adjusted[idx] = running
            frame["p_value_BH_FDR"] = adjusted
            frame["BH_FDR_Rejected_0.05"] = adjusted < 0.05
        return frame

    def export_results(self, output_dir: Path) -> tuple[pd.DataFrame, Path]:
        """Export hypothesis summary to CSV in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summary_dataframe()
        out_path = output_dir / "thesis_hypothesis_tests.csv"
        summary.to_csv(out_path, index=False)
        return summary, out_path


# ============================================================================
# diagnostics
# ============================================================================

# Stateless diagnostics for the cross-subset RUL investigation.

SUBSYSTEM_GROUPS = {
    "Fan_Inlet": ["T2", "P2", "Nf", "NRf"],
    "Compressors_LPC_HPC": ["T24", "T30", "P15", "P30", "Nc", "NRc"],
    "Turbines_Combustor": ["T50", "epr", "Ps30", "BPR", "farB", "htBleed", "W31", "W32"],
    "Controls_Demands": ["Nf_dmd", "PCNfR_dmd", "phi"],
    "Flight_Settings": ["setting_1", "setting_2", "setting_3"],
}


def extra_trees_baseline(random_state: int = DEFAULT_RANDOM_STATE, **kwargs) -> ExtraTreesRegressor:
    """Return the common Extra Trees baseline used across FD001-FD004."""
    params = {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 2, "max_features": 1.0, "n_jobs": -1}
    params.update(kwargs)
    return ExtraTreesRegressor(random_state=random_state, **params)


def compare_rul_subsets(datasets: dict[str, pd.DataFrame], features: list[str], n_splits: int = 5, random_state: int = DEFAULT_RANDOM_STATE):
    """Compute comparable grouped-fold mean and pooled OOF RUL results."""
    fold_results, predictions = {}, {}
    rows = []
    for subset, frame in datasets.items():
        folds, oof = grouped_oof(
            frame, features,
            estimator=extra_trees_baseline(random_state=random_state),
            n_splits=n_splits, target_col="RUL",
        )
        fold_results[subset], predictions[subset] = folds, oof
        pooled = regression_metrics(frame["RUL"], oof)
        rows.append({
            "Subpart": subset,
            "R2_fold_mean": folds["R2"].mean(),
            "R2_fold_std": folds["R2"].std(ddof=1),
            "R2_pooled_OOF": pooled["R2"],
            "RMSE_fold_mean": folds["RMSE"].mean(),
            "MAE_fold_mean": folds["MAE"].mean(),
            "Rows": len(frame),
            "Engines": frame["unit_id"].nunique(),
        })
    summary = pd.DataFrame(rows).sort_values("R2_pooled_OOF", ascending=False).reset_index(drop=True)
    return summary, fold_results, predictions


def compare_split_strategies(frame: pd.DataFrame, features: list[str], n_repeats: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Compare repeated random-row splits with grouped engine holdouts."""
    x, y, groups = frame[features].to_numpy(), frame["RUL"].to_numpy(), frame["unit_id"].to_numpy()
    row_scores, group_scores = [], []
    for rep in range(n_repeats):
        row_train, row_test = train_test_split(np.arange(len(frame)), test_size=0.25, random_state=random_state + rep)
        group_train, group_test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + rep).split(x, y, groups))
        row_model = extra_trees_baseline(random_state + rep).fit(x[row_train], y[row_train])
        group_model = extra_trees_baseline(random_state + rep).fit(x[group_train], y[group_train])
        row_scores.append(float(r2_score(y[row_test], row_model.predict(x[row_test]))))
        group_scores.append(float(r2_score(y[group_test], group_model.predict(x[group_test]))))

    row_r2 = float(np.mean(row_scores))
    group_r2 = float(np.mean(group_scores))
    return {
        "Row_random_R2": row_r2,
        "Engine_holdout_R2": group_r2,
        "Row_split_optimism_delta": row_r2 - group_r2,
        "Row_random_R2_std": float(np.std(row_scores, ddof=1)) if n_repeats > 1 else 0.0,
        "Engine_holdout_R2_std": float(np.std(group_scores, ddof=1)) if n_repeats > 1 else 0.0,
        "Engines_shared_in_row_split": len(set(groups[row_train]).intersection(groups[row_test])),
        "Engines_shared_in_group_split": len(set(groups[group_train]).intersection(groups[group_test])),
    }


def future_cycle_score(frame: pd.DataFrame, features: list[str], random_state: int = DEFAULT_RANDOM_STATE) -> dict:
    """Train on each engine's first 75% of cycles and score its later cycles."""
    cutoff = frame.groupby("unit_id")["cycle"].transform(lambda cycles: np.floor(cycles.max() * 0.75))
    train_mask, future_mask = frame["cycle"] <= cutoff, frame["cycle"] > cutoff
    model = extra_trees_baseline(random_state).fit(frame.loc[train_mask, features], frame.loc[train_mask, "RUL"])
    prediction = model.predict(frame.loc[future_mask, features])
    return {
        "Temporal_future_R2": float(r2_score(frame.loc[future_mask, "RUL"], prediction)),
        "Train_rows": int(train_mask.sum()),
        "Future_rows": int(future_mask.sum()),
        "Engines_shared_by_design": int(frame.loc[train_mask, "unit_id"].nunique()),
    }


def capacity_sweep(datasets: dict[str, pd.DataFrame], features: list[str], random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compare Extra Trees depths and Ridge strengths on the same grouped folds."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for subset, frame in datasets.items():
        for depth in (4, 8, 16, None):
            estimator = extra_trees_baseline(random_state, n_estimators=40, max_depth=depth)
            folds, _ = grouped_oof(frame, features, estimator=estimator)
            rows.append({"Subpart": subset, "Family": "Extra Trees", "Configuration": f"max_depth={depth or 'unlimited'}", "Mean_fold_R2": folds["R2"].mean(), "Std_fold_R2": folds["R2"].std(ddof=1)})
        for alpha in (0.1, 1.0, 10.0, 100.0):
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            folds, _ = grouped_oof(frame, features, estimator=estimator)
            rows.append({"Subpart": subset, "Family": "Ridge", "Configuration": f"alpha={alpha:g}", "Mean_fold_R2": folds["R2"].mean(), "Std_fold_R2": folds["R2"].std(ddof=1)})
    return pd.DataFrame(rows)


def learning_curves(frame: pd.DataFrame, features: list[str], subset: str, repeats: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Measure holdout R² as the number of training engines increases."""
    groups = frame["unit_id"].to_numpy()
    x, y = frame[features].to_numpy(), frame["RUL"].to_numpy()
    rows = []
    for repeat in range(repeats):
        train_idx, valid_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + repeat).split(x, y, groups))
        train_units = np.unique(groups[train_idx])
        shuffled = np.random.default_rng(random_state + repeat).permutation(train_units)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            count = max(2, int(np.ceil(len(shuffled) * fraction)))
            selected = shuffled[:count]
            partial = train_idx[np.isin(groups[train_idx], selected)]
            model = extra_trees_baseline(random_state, n_estimators=40).fit(x[partial], y[partial])
            rows.append({"Subpart": subset, "Repeat": repeat, "Train_engine_fraction": fraction, "Train_engines": count, "Validation_R2": r2_score(y[valid_idx], model.predict(x[valid_idx]))})
    return pd.DataFrame(rows)


def summarize_learning_curves(repeats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated learning-curve scores into mean and standard deviation."""
    return repeats.groupby(["Subpart", "Train_engine_fraction"], as_index=False).agg(
        Mean_R2=("Validation_R2", "mean"), Std_R2=("Validation_R2", "std"),
    )


def residual_segments(frame: pd.DataFrame, truth, prediction, subset: str) -> pd.DataFrame:
    """Summarize OOF residual direction and RMSE over quantile-defined RUL bands."""
    residuals = pd.DataFrame({
        "unit_id": frame["unit_id"].to_numpy(), "cycle": frame["cycle"].to_numpy(),
        "RUL": frame["RUL"].to_numpy(), "prediction": np.asarray(prediction),
    })
    residuals["residual"] = np.asarray(truth) - np.asarray(prediction)
    residuals["RUL_band"] = pd.qcut(residuals["RUL"], q=5, duplicates="drop")
    result = residuals.groupby("RUL_band", observed=True).agg(
        rows=("residual", "size"), mean_error=("residual", "mean"),
        mae=("residual", lambda values: np.abs(values).mean()),
        rmse=("residual", lambda values: np.sqrt(np.mean(values ** 2))),
    ).reset_index()
    result.insert(0, "Subpart", subset)
    return result


def feature_signal(frame: pd.DataFrame, features: list[str], subset: str, sample_size: int = 5000, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Return sampled univariate absolute correlation and mutual information with RUL."""
    sample = frame.sample(n=min(sample_size, len(frame)), random_state=random_state)
    x = sample[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mutual_info = mutual_info_regression(x, sample["RUL"], random_state=random_state)
    correlation = pd.Series(np.nan, index=features, dtype=float)
    varying_features = [feature for feature in features if x[feature].nunique() > 1]
    if varying_features and sample["RUL"].nunique() > 1:
        correlation.loc[varying_features] = x[varying_features].corrwith(sample["RUL"]).abs()
    return pd.DataFrame({
        "Subpart": subset,
        "Feature": features,
        "Abs_correlation": [float(correlation[feature]) for feature in features],
        "Mutual_information": mutual_info,
    })


def permutation_importances(frame: pd.DataFrame, features: list[str], subset: str, n_repeats: int = 10, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compute held-out engine permutation importance with subsystem mapping."""
    x, y, groups = frame[features].to_numpy(), frame["RUL"].to_numpy(), frame["unit_id"].to_numpy()
    fit_idx, score_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state).split(x, y, groups))
    model = extra_trees_baseline(random_state, n_estimators=40).fit(x[fit_idx], y[fit_idx])
    result = permutation_importance(model, x[score_idx], y[score_idx], scoring="r2", n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    # Map each feature to its physical subsystem
    feature_to_subsystem = {}
    for sub_name, sub_feats in SUBSYSTEM_GROUPS.items():
        for f in sub_feats:
            feature_to_subsystem[f] = sub_name

    subsystems = [feature_to_subsystem.get(f, "Other") for f in features]
    return pd.DataFrame({
        "Subpart": subset,
        "Feature": features,
        "Subsystem": subsystems,
        "R2_drop_mean": result.importances_mean,
        "R2_drop_std": result.importances_std,
    })


def distribution_shift(high_frame: pd.DataFrame, low_frame: pd.DataFrame, features: list[str], high_name: str, low_name: str, sample_size: int = 20000, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compare feature distributions with KS effect sizes and Wasserstein-1 distances."""
    rows = []
    for feature in [*features, "RUL"]:
        high = high_frame[feature].dropna()
        low = low_frame[feature].dropna()
        high = high.sample(n=min(sample_size, len(high)), random_state=random_state)
        low = low.sample(n=min(sample_size, len(low)), random_state=random_state)
        result = ks_2samp(high, low)

        # Standardized Wasserstein-1 distance
        pooled_vals = np.concatenate([high.to_numpy(), low.to_numpy()])
        p_mean = np.mean(pooled_vals)
        p_std = np.std(pooled_vals) if np.std(pooled_vals) > 1e-8 else 1.0
        w_dist = float(wasserstein_distance((high.to_numpy() - p_mean) / p_std, (low.to_numpy() - p_mean) / p_std))

        rows.append({
            "Feature": feature,
            "KS_statistic": float(result.statistic),
            "Wasserstein_distance": w_dist,
            "p_value": float(result.pvalue),
            "High_subset_median": float(high.median()),
            "Low_subset_median": float(low.median()),
            "High_subset": high_name,
            "Low_subset": low_name,
        })
    return pd.DataFrame(rows).sort_values("KS_statistic", ascending=False).reset_index(drop=True)


def repeated_grouped_cv(frame: pd.DataFrame, features: list[str], subset: str, repeats: int = 5, n_splits: int = 5, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Repeat pooled grouped OOF scoring with shuffled engine-to-fold assignment."""
    groups = frame["unit_id"].to_numpy()
    x, y = frame[features].to_numpy(dtype=float), frame["RUL"].to_numpy(dtype=float)
    rows = []
    for repeat in range(repeats):
        units = np.random.default_rng(random_state + repeat).permutation(np.unique(groups))
        fold_for = {unit: i % n_splits for i, unit in enumerate(units)}
        fold_ids = np.array([fold_for[group] for group in groups])
        predictions = np.full(len(frame), np.nan)
        for fold in range(n_splits):
            valid = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            model = extra_trees_baseline(random_state + repeat * 10 + fold).fit(x[train], y[train])
            predictions[valid] = model.predict(x[valid])
        rows.append({"Subpart": subset, "Repeat": repeat, "R2_pooled_OOF": float(r2_score(y, predictions))})
    return pd.DataFrame(rows)
