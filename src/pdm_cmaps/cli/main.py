#!/usr/bin/env python3
"""Command-line entry point: `python -m pdm_cmaps` (or the `pdm-cmaps` console script).

parse_arguments() -> PipelineConfig -> PdMPipeline.execute(); with --all, then cli.update_all runs the extra studies.
"""

from __future__ import annotations



import argparse
from pathlib import Path
import sys

from pdm_cmaps.config import DEFAULT_PATHS
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.modeling.benchmark import RULBenchmark
from pdm_cmaps.pipeline.pipeline import PdMPipeline
from pdm_cmaps.pipeline.config import PipelineConfig
from pdm_cmaps.progress import ProgressTracker


def parse_arguments(args: list[str] | None = None) -> PipelineConfig:
    """Parse command-line arguments into a strongly-typed PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="NASA C-MAPSS Turbofan Predictive Maintenance Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subset",
        choices=("FD001", "FD002", "FD003", "FD004", "ALL"),
        default="ALL",
        help="C-MAPSS dataset subset to process (or ALL for all four)",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Execute pipeline sequentially across all subsets (FD001-FD004)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility and engine unit partitioning",
    )
    parser.add_argument(
        "--healthy-cycle-limit",
        type=int,
        default=50,
        help="Healthy operating cycle threshold (H) for virtual sensor calibration",
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        default=179,
        help="Sample engine unit identifier for detailed trace visualizations",
    )
    parser.add_argument(
        "--include-minirocket",
        dest="include_minirocket",
        action="store_true",
        help="Also run the MiniRocket convolutional model (off by default; it is not part of the monograph)",
    )
    parser.add_argument(
        "--no-minirocket",
        dest="include_minirocket",
        action="store_false",
        help="Disable MiniRocket (the default)",
    )
    parser.set_defaults(include_minirocket=False)
    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help="Enable PyTorch LSTM sequential model training",
    )
    parser.add_argument(
        "--no-clean",
        dest="clean_output",
        action="store_false",
        help="Do not clean existing output directories before execution",
    )
    parser.set_defaults(clean_output=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory for results (tables and figures)",
    )

    parser.add_argument(
        "--only-pipeline",
        action="store_true",
        help="With --all: run only the per-subset pipeline, without the extra studies and table sync (update_all.py)",
    )

    parsed = parser.parse_args(args)
    return PipelineConfig(
        subset=parsed.subset,
        seed=parsed.seed,
        healthy_cycle_limit=parsed.healthy_cycle_limit,
        unit_id=parsed.unit_id,
        include_minirocket=parsed.include_minirocket,
        include_lstm=parsed.include_lstm,
        clean_output=parsed.clean_output,
        run_all=parsed.run_all,
        output_dir=parsed.output_dir,
    )


def main(args: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline for each requested subset (with progress/ETA) and return the exit code."""
    # 1. Configuration
    config = parse_arguments(args)

    # 2. Determine target subsets
    target_subsets = ["FD001", "FD002", "FD003", "FD004"] if (config.subset == "ALL" or getattr(config, "run_all", False)) else [config.subset]

    # 3. Component instantiation and execution loop
    loader = CMapssLoader()
    exit_code = 0
    multi = len(target_subsets) > 1  # a single subset needs no progress file
    progress = ProgressTracker(len(target_subsets), log_file=DEFAULT_PATHS.logs_dir / "_progresso.txt" if multi else None)

    for sub in target_subsets:
        progress.start(f"pipeline {sub}")
        n_regimes = 6 if sub in {"FD002", "FD004"} else 1
        decoupler = RegimeDecoupler(n_clusters=n_regimes, random_state=config.seed)
        benchmark = RULBenchmark(random_state=config.seed)

        # Clone config for current subset
        sub_config = PipelineConfig(
            subset=sub,
            seed=config.seed,
            healthy_cycle_limit=config.healthy_cycle_limit,
            unit_id=config.unit_id if sub in {"FD002", "FD004"} else 45,
            include_minirocket=config.include_minirocket,
            include_lstm=config.include_lstm,
            clean_output=config.clean_output,
            output_dir=config.output_dir,
        )

        pipeline = PdMPipeline(
            config=sub_config,
            loader=loader,
            decoupler=decoupler,
            benchmark=benchmark,
        )
        res = pipeline.execute()
        progress.finish(f"pipeline {sub}", res)
        if res != 0:
            exit_code = res

    # `--all` atualiza tudo de uma vez: estudos adicionais, tabelas versionadas e painéis
    argv = args if args is not None else sys.argv[1:]
    if "--all" in argv and "--only-pipeline" not in argv:
        from pdm_cmaps.cli.update_all import run_studies_and_sync

        exit_code = run_studies_and_sync() or exit_code

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
