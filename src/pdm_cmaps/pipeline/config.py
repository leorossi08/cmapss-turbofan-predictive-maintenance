"""Strongly-typed pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdm_cmaps.config import DEFAULT_PATHS
from pdm_cmaps.domain.schema import DEFAULT_HEALTHY_CYCLE_LIMIT, DEFAULT_RANDOM_STATE


@dataclass
class PipelineConfig:
    """Strongly-typed pipeline configuration parameters."""

    subset: str = "FD004"
    seed: int = DEFAULT_RANDOM_STATE
    healthy_cycle_limit: int = DEFAULT_HEALTHY_CYCLE_LIMIT
    unit_id: int = 179
    include_minirocket: bool = False
    include_lstm: bool = False
    clean_output: bool = True
    run_all: bool = False
    output_dir: Path | None = None

    @property
    def resolved_output_dir(self) -> Path:
        """Return the effective output directory path."""
        if self.output_dir is not None:
            return Path(self.output_dir)
        return DEFAULT_PATHS.guide_dir(self.subset)
