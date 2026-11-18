"""Schema of the C-MAPSS files, sensor metadata and project-wide defaults."""

from __future__ import annotations





SENSOR_COLUMNS = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]

ID_COLUMNS = ["unit_id", "cycle"]

COLUMNS = [*ID_COLUMNS, *SETTING_COLUMNS, *SENSOR_COLUMNS]

# Sensor metadata. [ASSUMPTION] The drift-direction tuples below are descriptive only: they are
# NOT used to orient the health index (that is done empirically from the data, see
# HealthIndexFuser.fit) because drift directions vary between subsets and fault modes.
POSITIVE_DRIFT_SENSORS: tuple[str, ...] = (
    "T24", "T30", "T50", "Nf", "Nc", "NRf", "BPR", "htBleed",
)

NEGATIVE_DRIFT_SENSORS: tuple[str, ...] = (
    "P30", "Ps30", "phi", "NRc", "W31", "W32",
)

EXOGENOUS_CONSTANT_SENSORS: tuple[str, ...] = (
    "T2", "P2", "P15", "epr", "farB", "Nf_dmd", "PCNfR_dmd",
)

SENSOR_UNITS: dict[str, str] = {
    "T2": "°R",
    "T24": "°R",
    "T30": "°R",
    "T50": "°R",
    "P2": "psia",
    "P15": "psia",
    "P30": "psia",
    "Nf": "rpm",
    "Nc": "rpm",
    "epr": "adim.",
    "Ps30": "psia",
    "phi": "pps/psi",
    "NRf": "rpm",
    "NRc": "rpm",
    "BPR": "adim.",
    "farB": "adim.",
    "htBleed": "°R",
    "Nf_dmd": "rpm",
    "PCNfR_dmd": "%",
    "W31": "lbm/s",
    "W32": "lbm/s",
}

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

DEFAULT_RUL_CAP = 125

DEFAULT_HEALTHY_CYCLE_LIMIT = 50

DEFAULT_RANDOM_STATE = 42
