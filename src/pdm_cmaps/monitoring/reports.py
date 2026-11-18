"""Alert facts and deterministic report text for the maintenance engineer.

WHAT   Packs everything the models know about one asset at one cycle into a small, causal facts
       dictionary (RUL estimate and interval, trend, anomaly status, criticality level, action)
       and renders it as a short Portuguese message used in alert e-mails and by the portal.
WHY    The text is generated from the facts by a fixed template, so every number in it comes
       straight from a model output: nothing is estimated or invented at this stage.
CONCEPTS
  [CAUSALITY]    facts use only rows up to the alert cycle (no future information).
  [DECISION-POLICY] the recommended action is a fixed mapping from the criticality level, a
                 maintenance-policy choice that the engineer can change in ACTIONS_PT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm_cmaps.monitoring.alerts import LEVEL_NAMES, CriticalityPolicy

LEVEL_LABELS_PT = ("NORMAL", "ATENÇÃO", "ALERTA", "CRÍTICO")
ACTIONS_PT = (
    "manter o monitoramento de rotina",
    "programar uma inspeção na próxima janela de manutenção",
    "planejar a manutenção e reservar peças e equipe",
    "providenciar intervenção imediata",
)
TREND_WINDOW = 10


def build_facts(history: pd.DataFrame, policy: CriticalityPolicy, strategy: str = "interval+anomaly", coverage_pct: float = 90.0) -> dict:
    """Structured, causal facts for one asset at its latest cycle (only past/present rows used)."""
    h = history.sort_values("cycle")
    last = h.iloc[-1]
    window = h.tail(TREND_WINDOW)
    slope = float(np.polyfit(window["cycle"], window["RUL_pred"], 1)[0]) if len(window) >= 3 else 0.0
    anomaly_rows = h.loc[h["anomaly_active"]]
    since = int(last["cycle"] - anomaly_rows["cycle"].min()) if not anomaly_rows.empty else None
    level = int(last[f"level_{strategy}"])
    return {
        "unit_id": int(last["unit_id"]),
        "cycle": int(last["cycle"]),
        "rul_estimate": round(float(last["RUL_pred"])),
        "rul_lower": round(float(last["RUL_lower"])),
        "rul_upper": round(float(last["RUL_upper"])),
        "interval_coverage_pct": round(float(coverage_pct)),
        "trend_cycles_per_cycle": round(slope, 1),
        "trend_window_cycles": TREND_WINDOW,
        "anomaly_active": bool(last["anomaly_active"]),
        "cycles_since_anomaly": since,
        "level_index": level,
        "level_name": LEVEL_NAMES[level],
        "level_label_pt": LEVEL_LABELS_PT[level],
        "recommended_action_pt": ACTIONS_PT[level],
        "thresholds": {"attention": policy.attention, "warning": policy.warning, "critical": policy.critical},
    }


def render_report(facts: dict) -> str:
    """Deterministic message for e-mail / portal; all numbers are copied from ``facts``."""
    anomaly = (
        f"Há anomalia ativa há {facts['cycles_since_anomaly']} ciclos."
        if facts["anomaly_active"] and facts["cycles_since_anomaly"] is not None
        else "Não há anomalia ativa no monitoramento estatístico."
    )
    return (
        f"Ativo {facts['unit_id']}, ciclo {facts['cycle']}. "
        f"Nível de criticidade: {facts['level_label_pt']}. "
        f"RUL estimada de {facts['rul_estimate']} ciclos (intervalo de {facts['interval_coverage_pct']}%: "
        f"{facts['rul_lower']} a {facts['rul_upper']} ciclos). "
        f"Tendência da RUL estimada: {facts['trend_cycles_per_cycle']} ciclos por ciclo nos últimos {facts['trend_window_cycles']} ciclos. "
        f"{anomaly} Ação recomendada: {facts['recommended_action_pt']}."
    )
