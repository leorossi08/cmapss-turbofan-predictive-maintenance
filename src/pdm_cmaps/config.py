"""Project paths: the single place that knows where data and outputs live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# src/pdm_cmaps/config.py -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout of the repository, injectable for tests and alternative checkouts."""

    root: Path = PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "CMaps"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def tables_dir(self) -> Path:
        """Versioned tables consumed by the monograph and dashboards."""
        return self.root / "tcc-writing" / "tables"

    @property
    def logs_dir(self) -> Path:
        return self.results_dir / "logs"

    def guide_dir(self, subset: str) -> Path:
        """Output directory of the per-subset pipeline."""
        return self.results_dir / f"maintenance_guide_{subset.lower()}"


DEFAULT_PATHS = ProjectPaths()
DEFAULT_DATA_DIR = DEFAULT_PATHS.data_dir
DEFAULT_RESULTS_DIR = DEFAULT_PATHS.results_dir
