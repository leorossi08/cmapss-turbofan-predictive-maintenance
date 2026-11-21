"""Criticality levels and alert generation.

WHAT   Converts model outputs (RUL point, RUL interval, anomaly flag) into a criticality level
       an engineer can act on, compares several automatic strategies, and produces alert events.
WHY    Research question RQ-C: how should criticality be defined automatically? Three strategies
       are compared on held-out engines with the same ground truth:
         point               level from the predicted RUL
         interval            level from the CONSERVATIVE end (lower bound) of the RUL interval
         interval+anomaly    as above, escalated one level when an SPC anomaly alarm is active
CONCEPTS
  [ASSUMPTION]   the level thresholds (<=15 critical, <=30 warning, <=60 attention cycles) are
                 maintenance-policy parameters, not facts about the data. Report sensitivity.
  [METRIC-CHOICE] a missed critical state costs far more than an early warning, so we report
                 missed-critical rate AND false-alarm rate (a trade-off), never accuracy alone.
  [CAUSALITY]    debouncing uses only the past: a level is raised only after `persistence`
                 consecutive cycles at or above it (same idea as the Western-Electric rule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVEL_NAMES = ("NORMAL", "ATTENTION", "WARNING", "CRITICAL")
NORMAL, ATTENTION, WARNING, CRITICAL = range(4)


@dataclass(frozen=True)
class CriticalityPolicy:
    """RUL thresholds (cycles) that define the levels; level k applies when RUL <= threshold k."""

    attention: float = 60.0
    warning: float = 30.0
    critical: float = 15.0
    persistence: int = 2

    def level_from_rul(self, rul) -> np.ndarray:
        r = np.asarray(rul, dtype=float)
        level = np.zeros(r.shape, dtype=int)
        level[r <= self.attention] = ATTENTION
        level[r <= self.warning] = WARNING
        level[r <= self.critical] = CRITICAL
        return level


def debounce(levels: np.ndarray, persistence: int) -> np.ndarray:
    """Causal k-of-k filter: level_t = min(raw levels over the last k cycles)."""
    levels = np.asarray(levels, dtype=int)
    if persistence <= 1:
        return levels.copy()
    out = levels.copy()
    for t in range(len(levels)):
        out[t] = levels[max(0, t - persistence + 1): t + 1].min()
    return out


def build_alert_timeline(
    frame: pd.DataFrame,
    policy: CriticalityPolicy,
    first_alarm_cycle: dict[int, int | None] | None = None,
    extra_alarms: dict[str, dict[int, int | None]] | None = None,
) -> pd.DataFrame:
    """Add true level and one level column per strategy to a per-cycle prediction frame.

    ``frame`` needs: unit_id, cycle, RUL_true, RUL_pred, RUL_lower, RUL_upper.
    ``first_alarm_cycle`` maps unit_id -> first persistent SPC alarm cycle (None = no alarm).
    ``extra_alarms`` adds one strategy ``interval+<name>`` per extra detector (same escalation rule).
    """
    first_alarm_cycle = first_alarm_cycle or {}
    parts = []
    for unit_id, eng in frame.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=False):
        eng = eng.copy()
        alarm_cycle = first_alarm_cycle.get(int(unit_id))
        eng["anomaly_active"] = _active(eng, alarm_cycle)
        eng["level_true"] = policy.level_from_rul(eng["RUL_true"])
        raw = {
            "point": policy.level_from_rul(eng["RUL_pred"]),
            "interval": policy.level_from_rul(eng["RUL_lower"]),
        }
        raw["interval+anomaly"] = _escalate(raw["interval"], eng["anomaly_active"].to_numpy(dtype=int))
        for name, alarm_map in (extra_alarms or {}).items():
            active = _active(eng, alarm_map.get(int(unit_id)))
            eng[f"{name}_active"] = active
            raw[f"interval+{name}"] = _escalate(raw["interval"], np.asarray(active, dtype=int))
        # anomaly escalation only applies when the interval already shows some risk (> NORMAL):
        # a lone SPC alarm in early life must not create a maintenance order by itself.
        for name, levels in raw.items():
            eng[f"level_{name}"] = debounce(levels, policy.persistence)
        parts.append(eng)
    return pd.concat(parts, ignore_index=True)


def _active(eng: pd.DataFrame, alarm_cycle) -> pd.Series | bool:
    if alarm_cycle is None or pd.isna(alarm_cycle):
        return pd.Series(False, index=eng.index)
    return eng["cycle"] >= alarm_cycle


def _escalate(levels: np.ndarray, active: np.ndarray) -> np.ndarray:
    return np.minimum(levels + active * (levels > NORMAL), CRITICAL)


STRATEGIES = ("point", "interval", "interval+anomaly")


def strategy_names(timeline: pd.DataFrame) -> list[str]:
    """Base strategies plus any ``interval+<detector>`` columns added through ``extra_alarms``."""
    extra = [c[len("level_"):] for c in timeline.columns if c.startswith("level_interval+") and c[len("level_"):] not in STRATEGIES]
    return [*STRATEGIES, *extra]


def evaluate_strategies(timeline: pd.DataFrame, policy: CriticalityPolicy, premature_rul: float = 60.0) -> pd.DataFrame:
    """Row-level and engine-level quality of each criticality strategy on held-out engines."""
    rows = []
    true = timeline["level_true"].to_numpy()
    for name in strategy_names(timeline):
        pred = timeline[f"level_{name}"].to_numpy()
        crit = true == CRITICAL
        normal = true == NORMAL
        lead_times, early, missed = [], 0, 0
        n_units = 0
        for _, eng in timeline.groupby("unit_id"):
            n_units += 1
            eng = eng.sort_values("cycle")
            eol = int(eng["cycle"].max())
            hit = eng.loc[eng[f"level_{name}"] >= WARNING]
            if hit.empty:
                missed += 1
                continue
            first = hit.iloc[0]
            if first["RUL_true"] > premature_rul:
                early += 1
            lead_times.append(eol - int(first["cycle"]))
        # classificação linha a linha: positivo = nível verdadeiro >= ALERTA; alerta = nível previsto >= ALERTA
        pos, called = true >= WARNING, pred >= WARNING
        tp, fp, fn = float(np.sum(pos & called)), float(np.sum(~pos & called)), float(np.sum(pos & ~called))
        rows.append({
            "Strategy": name,
            "Alert_precision": tp / (tp + fp) if tp + fp > 0 else float("nan"),
            "Alert_recall": tp / (tp + fn) if tp + fn > 0 else float("nan"),
            "Alert_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan"),
            "Exact_level_accuracy": float((pred == true).mean()),
            "Within_one_level": float((np.abs(pred - true) <= 1).mean()),
            "Missed_critical_rate": float((pred[crit] < WARNING).mean()) if crit.any() else float("nan"),
            "False_alarm_rate_normal": float((pred[normal] >= WARNING).mean()) if normal.any() else float("nan"),
            "Engines": n_units,
            "Engines_never_alerted": missed,
            "Engines_premature_alert": early,
            "Median_warning_lead_time": float(np.median(lead_times)) if lead_times else float("nan"),
        })
    return pd.DataFrame(rows)


def generate_alert_events(timeline: pd.DataFrame, strategy: str = "interval+anomaly") -> pd.DataFrame:
    """One event per level ESCALATION (and per de-escalation to NORMAL) for each engine."""
    events = []
    col = f"level_{strategy}"
    for unit_id, eng in timeline.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=False):
        previous = NORMAL
        for _, row in eng.iterrows():
            level = int(row[col])
            if level > previous:
                events.append({
                    "unit_id": int(unit_id), "cycle": int(row["cycle"]),
                    "from_level": LEVEL_NAMES[previous], "to_level": LEVEL_NAMES[level],
                    "RUL_pred": float(row["RUL_pred"]), "RUL_lower": float(row["RUL_lower"]),
                    "RUL_upper": float(row["RUL_upper"]), "RUL_true": float(row["RUL_true"]),
                })
            previous = level
    return pd.DataFrame(events)



def maintenance_cost(timeline: pd.DataFrame, strategy: str, cost_ratio: float, lead: int = 0, trigger: int = CRITICAL) -> dict:
    """Renewal-reward cost of the rule "order maintenance at the first ``trigger`` level, done ``lead`` cycles later".

    The decision at cycle t is executed at t + lead (parts, crew, slot). If t + lead < EOL the engine is
    replaced as planned (cost 1, cycles used t + lead); otherwise it fails at EOL (cost ``cost_ratio``).
    By the renewal-reward theorem the long-run cost per operating cycle is sum(costs) / sum(cycles used);
    early replacement is penalised implicitly (fewer cycles of use for the same cost). Reported relative
    to an oracle that replaces every engine one cycle before failure (1 = perfect).
    """
    col = f"level_{strategy}"
    cost, used, oracle_used, failures, unused = 0.0, 0.0, 0.0, 0, []
    for _, eng in timeline.groupby("unit_id"):
        eng = eng.sort_values("cycle")
        eol = int(eng["cycle"].max())
        hit = eng.loc[eng[col] >= trigger, "cycle"]
        done = int(hit.iloc[0]) + lead if not hit.empty else None
        if done is None or done >= eol:
            cost += cost_ratio
            used += eol
            failures += 1
        else:
            cost += 1.0
            used += done
            unused.append(eol - done)
        oracle_used += eol - 1
    n = timeline["unit_id"].nunique()
    rate = cost / used
    oracle = n / oracle_used
    return {"Strategy": strategy, "Cost_ratio": cost_ratio, "Lead": lead, "Cost_rate": rate, "Relative_to_oracle": rate / oracle,
            "Failures": failures, "Engines": n, "Mean_unused_cycles": float(np.mean(unused)) if unused else float("nan")}


def cost_table(timeline: pd.DataFrame, ratios=(2, 5, 10, 20, 50, 100), leads=(0, 5, 10), trigger: int = CRITICAL) -> pd.DataFrame:
    """maintenance_cost for every strategy, failure/planned cost ratio and logistics lead time."""
    return pd.DataFrame([maintenance_cost(timeline, s, r, L, trigger) for s in strategy_names(timeline) for r in ratios for L in leads])
