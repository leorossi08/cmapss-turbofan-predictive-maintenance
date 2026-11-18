"""Construtores de dados sintéticos compartilhados entre os arquivos de teste."""

import numpy as np
import pandas as pd
from pdm_cmaps.domain.schema import COLUMNS, SENSOR_COLUMNS


def sample_frame(engine_count: int = 10) -> pd.DataFrame:
    rows = []
    for unit_id in range(1, engine_count + 1):
        for cycle in range(1, 6):
            row = {column: 0.0 for column in COLUMNS}
            row.update({"unit_id": unit_id, "cycle": cycle, "setting_1": unit_id % 3, "setting_2": 0.2, "setting_3": 100.0})
            for index, sensor in enumerate(SENSOR_COLUMNS):
                row[sensor] = float(unit_id + cycle * (index + 1))
            rows.append(row)
    return pd.DataFrame(rows)

def make_synthetic_cmapss(
    n_engines: int = 12,
    min_cycles: int = 60,
    max_cycles: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic C-MAPSS dataset with diverse EOLs and 6 flight regimes."""
    rng = np.random.default_rng(seed)
    rows = []
    regime_settings = [
        (0.0, 0.0, 100.0),
        (10.0, 0.25, 100.0),
        (20.0, 0.70, 100.0),
        (25.0, 0.62, 60.0),
        (35.0, 0.84, 100.0),
        (42.0, 0.84, 100.0),
    ]
    for unit_id in range(1, n_engines + 1):
        eol = rng.integers(min_cycles, max_cycles + 1)
        for cycle in range(1, eol + 1):
            r_idx = rng.integers(0, len(regime_settings))
            s1, s2, s3 = regime_settings[r_idx]
            deg_factor = (cycle / eol) ** 2
            row = {
                "unit_id": unit_id,
                "cycle": cycle,
                "setting_1": s1 + rng.normal(0, 0.01),
                "setting_2": s2 + rng.normal(0, 0.005),
                "setting_3": s3,
            }
            # Physical sensors: T2, P2, P15, epr are boundary/inlet conditions
            row["T2"] = 518.67 - 3.5 * s1 + rng.normal(0, 0.2)
            row["P2"] = 14.696 * (1.0 - 0.02 * s1) + rng.normal(0, 0.05)
            row["P15"] = 14.0 + 0.1 * s2 + rng.normal(0, 0.02)
            row["epr"] = 1.0 + 0.3 * s2 + rng.normal(0, 0.01)
            row["Nf_dmd"] = 2388.0
            row["PCNfR_dmd"] = 100.0
            # Internal degraded sensors: T50, T24, P30 drift with deg_factor
            row["T24"] = 640.0 + 5.0 * s1 + 15.0 * deg_factor + rng.normal(0, 0.5)
            row["T30"] = 1580.0 + 10.0 * s1 + 25.0 * deg_factor + rng.normal(0, 1.0)
            row["T50"] = 1400.0 + 8.0 * s1 + 35.0 * deg_factor + rng.normal(0, 0.8)
            row["P30"] = 550.0 - 2.0 * s1 - 10.0 * deg_factor + rng.normal(0, 0.5)
            # Remaining sensor columns
            for sensor in SENSOR_COLUMNS:
                if sensor not in row:
                    row[sensor] = 100.0 + rng.normal(0, 1.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

def _toy_timeline(n_cycles: int = 80, anomaly_from: int | None = 40):
    import numpy as np
    import pandas as pd
    from pdm_cmaps.monitoring.alerts import CriticalityPolicy, build_alert_timeline

    cycles = np.arange(1, n_cycles + 1)
    rul_true = np.clip(n_cycles - cycles, 0, 125).astype(float)
    rng = np.random.default_rng(0)
    pred = np.clip(rul_true + rng.normal(0, 4, n_cycles), 0, 125)
    frame = pd.DataFrame({
        "unit_id": 7, "cycle": cycles, "RUL_true": rul_true, "RUL_pred": pred,
        "RUL_lower": np.clip(pred - 10, 0, 125), "RUL_upper": np.clip(pred + 10, 0, 125),
    })
    policy = CriticalityPolicy()
    return build_alert_timeline(frame, policy, {7: anomaly_from}), policy

