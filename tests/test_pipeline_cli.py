"""Configuração do pipeline e linha de comando."""

from pathlib import Path

import pytest

from pdm_cmaps.cli.main import parse_arguments
from pdm_cmaps.data.loader import CMapssLoader
from pdm_cmaps.detection.regime import RegimeDecoupler
from pdm_cmaps.modeling.benchmark import RULBenchmark
from pdm_cmaps.pipeline.config import PipelineConfig
from pdm_cmaps.pipeline.pipeline import PdMPipeline


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.subset == "FD004"
    assert config.seed == 42
    assert config.healthy_cycle_limit == 50
    assert config.unit_id == 179
    assert config.include_minirocket is True
    assert config.include_lstm is False
    assert config.clean_output is True
    assert "maintenance_guide_fd004" in str(config.resolved_output_dir)


def test_pipeline_config_custom_output_dir(tmp_path: Path):
    custom_dir = tmp_path / "custom_results"
    config = PipelineConfig(subset="FD001", output_dir=custom_dir)
    assert config.resolved_output_dir == custom_dir


def test_parse_arguments_cli():
    cli_args = [
        "--subset", "FD002",
        "--seed", "123",
        "--healthy-cycle-limit", "60",
        "--unit-id", "45",
        "--no-minirocket",
        "--include-lstm",
        "--no-clean",
    ]
    config = parse_arguments(cli_args)
    assert config.subset == "FD002"
    assert config.seed == 123
    assert config.healthy_cycle_limit == 60
    assert config.unit_id == 45
    assert config.include_minirocket is False
    assert config.include_lstm is True
    assert config.clean_output is False


def test_pdm_pipeline_instantiation(tmp_path: Path):
    out_dir = tmp_path / "test_out"
    config = PipelineConfig(subset="FD001", output_dir=out_dir)

    loader = CMapssLoader()
    decoupler = RegimeDecoupler(n_clusters=1, random_state=42)
    benchmark = RULBenchmark(random_state=42)

    pipeline = PdMPipeline(
        config=config,
        loader=loader,
        decoupler=decoupler,
        benchmark=benchmark,
    )

    assert pipeline.config == config
    assert pipeline.loader is loader
    assert pipeline.decoupler is decoupler
    assert pipeline.benchmark is benchmark
    assert pipeline.output_dir == out_dir

    # Test directory preparation
    pipeline.prepare_directories()
    assert pipeline.table_dir.exists()
    assert pipeline.figure_dir.exists()


def test_progress_tracker_estimates_time_left_from_finished_steps(tmp_path):
    from pdm_cmaps.progress import ProgressTracker, format_duration

    now = [0.0]
    lines = []
    log = tmp_path / "logs" / "_progresso.txt"
    tracker = ProgressTracker(4, log_file=log, clock=lambda: now[0], emit=lines.append)

    tracker.start("a")
    assert tracker.remaining_seconds() is None and "estimado após o 1º passo" in lines[-1]
    now[0] = 600.0  # first step took 10 min
    tracker.finish("a")
    assert tracker.remaining_seconds() == 3 * 600.0  # mean 10 min x 3 steps left
    assert "[1/4] a" in lines[-1] and "restante ~30min" in lines[-1]
    tracker.start("b")
    now[0] = 1800.0  # second step took 20 min -> mean 15 min x 2 left
    tracker.finish("b", returncode=1)
    assert "FALHOU" in lines[-1] and tracker.remaining_seconds() == 2 * 900.0
    assert log.read_text().count("\n") == len(lines)  # every line is also persisted for `tail -f`
    assert [format_duration(s) for s in (20, 90, 4500)] == ["20s", "1min", "1h15min"]



def test_main_prints_progress_for_each_subset_and_persists_it(tmp_path, monkeypatch, capsys):
    import pdm_cmaps.cli.main as cli_main
    from pdm_cmaps.config import ProjectPaths

    monkeypatch.setattr(cli_main, "DEFAULT_PATHS", ProjectPaths(root=tmp_path))
    monkeypatch.setattr(PdMPipeline, "execute", lambda self: 0)  # the heavy work is not under test here

    assert cli_main.main(["--subset", "ALL"]) == 0
    out = capsys.readouterr().out
    for i, subset in enumerate(["FD001", "FD002", "FD003", "FD004"], start=1):
        assert f"[{i}/4] pipeline {subset}" in out
    assert (tmp_path / "results" / "logs" / "_progresso.txt").read_text().count("\n") == 8  # start + finish per subset


def test_study_runner_run_all_reports_progress_and_keeps_failures(tmp_path, monkeypatch, capsys):
    from pdm_cmaps.cli.update_all import StudyRunner, StudySpec
    from pdm_cmaps.config import ProjectPaths

    runner = StudyRunner(ProjectPaths(root=tmp_path))
    codes = iter([0, 1, 0])
    monkeypatch.setattr(runner, "run_study", lambda study: next(codes))
    studies = tuple(StudySpec(name, "m") for name in ("a", "b", "c"))

    assert runner.run_all(studies) == 1  # one failure is not swallowed by the later successes
    out = capsys.readouterr().out
    assert "[2/3] b ... FALHOU (código 1)" in out and "[3/3] c ... ok" in out
    assert (tmp_path / "results" / "logs" / "_progresso.txt").exists()


@pytest.mark.parametrize("healthy_limit, phases", [(50, 5), (100, 4)])
def test_pipeline_execute_reports_one_progress_line_per_phase(tmp_path, monkeypatch, capsys, healthy_limit, phases):
    for name in ("run_core_workflow", "run_auxiliary_spc_horizon", "run_additional_models_and_statistics",
                 "run_monitoring_layer", "run_figure_generation"):
        monkeypatch.setattr(PdMPipeline, name, lambda self, *a, **k: None)
    config = PipelineConfig(subset="FD001", healthy_cycle_limit=healthy_limit, output_dir=tmp_path / "out")
    pipeline = PdMPipeline(config=config, loader=CMapssLoader(), decoupler=RegimeDecoupler(n_clusters=1, random_state=42),
                           benchmark=RULBenchmark(random_state=42))

    assert pipeline.execute() == 0
    out = capsys.readouterr().out
    assert f"[{phases}/{phases}] fase" in out and f"[{phases + 1}/" not in out  # H=100 skips the auxiliary phase
