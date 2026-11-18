"""Reexecuta os estudos além do pipeline por dataset e sincroniza as tabelas da monografia (tcc-writing/tables/) e os gráficos Plotly.
O site de painéis fica em outro repositório; depois desta atualização, copie os resultados para ele com
`python fonte/atualizar_dados.py` (no repositório do dashboard).

Chamado por `python -m pdm_cmaps --all` (depois do pipeline FD001 a FD004). Também pode rodar sozinho:
    python -m pdm_cmaps.cli.update_all                    # estudos + sincronização
    python -m pdm_cmaps.cli.update_all --so-sincronizar
Cada estudo roda em um processo, um por vez, com log em results/logs/<nome>.log.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

from pdm_cmaps.config import DEFAULT_PATHS, ProjectPaths
from pdm_cmaps.progress import ProgressTracker


@dataclass(frozen=True)
class StudySpec:
    """A study = a runnable module (`python -m <module> <args>`) with a log name."""

    name: str
    module: str
    args: tuple[str, ...] = ()


E_ = "pdm_cmaps.experiments"
STUDIES: tuple[StudySpec, ...] = (
    StudySpec("monitoring", "pdm_cmaps.monitoring.layer", ("--jobs", "3")),
    StudySpec("benchmark_repeated", f"{E_}.benchmark_repeated"),
    StudySpec("healthy_window", f"{E_}.healthy_window_study", ("--jobs", "3")),
    StudySpec("fault_detection", f"{E_}.fault_detection_study", ("--jobs", "3")),
    StudySpec("fault_model_diag13", f"{E_}.fault_model_study",
              ("--H", "25", "--subsets", "FD001", "FD003", "--jobs", "3", "--output-dir", "results/injecao_modelos_diag")),
    StudySpec("fault_model_diag24", f"{E_}.fault_model_study",
              ("--H", "25", "--subsets", "FD002", "FD004", "--jobs", "3", "--output-dir", "results/injecao_modelos_diag24")),
    StudySpec("fault_hard", f"{E_}.fault_hard_study", ("--jobs", "3")),
    StudySpec("fault_injection", f"{E_}.fault_injection", ("--subset", "FD004")),
    StudySpec("rul_predicoes", f"{E_}.rul_prediction_plots", ("--jobs", "3")),
    StudySpec("envelope_cases", f"{E_}.envelope_ignored_cases", ("--results", "results")),
)

# tabela versionada -> arquivo de origem em results/
TABLE_COPIES: dict[str, str] = {
    "benchmark_repeated_summary.csv": "benchmark_repeated/benchmark_repeated_summary.csv",
    "benchmark_repeated_tests.csv": "benchmark_repeated/benchmark_repeated_tests.csv",
    "diferenca_vs_media_regime.csv": "falhas_dificeis/tabelas/diferenca_vs_media_regime.csv",
    "poder_por_forma.csv": "falhas_dificeis/tabelas/poder_por_forma.csv",
    "poder_por_magnitude.csv": "falhas_dificeis/tabelas/poder_por_magnitude.csv",
    "healthy_window_summary.csv": "healthy_window_study/tables/healthy_window_summary.csv",
    "magnitude_minima_detectavel.csv": "injecao_deteccao/tabelas/magnitude_minima_detectavel.csv",
    "pareado_far_resumo.csv": "injecao_deteccao/tabelas/pareado_far_resumo.csv",
    "regra_padrao_s1.csv": "injecao_deteccao/tabelas/regra_padrao_s1.csv",
    "thesis_hypothesis_tests.csv": "maintenance_guide_fd004/tables/thesis_hypothesis_tests.csv",
    **{f: f"monitoring_repeated/tables/{f}" for f in (
        "margin_check_point_thresholds_raw.csv", "monitoring_repeated_costs_summary.csv", "monitoring_repeated_spc_mv_summary.csv",
        "monitoring_repeated_spc_summary.csv", "monitoring_repeated_summary.csv", "monitoring_repeated_uncertainty_summary.csv",
        "monitoring_repeated_uncertainty_variants_summary.csv", "regime_variance_by_subset.csv")},
}

class StudyRunner:
    """Runs studies as subprocesses, one at a time, logging each to results/logs/<name>.log."""

    def __init__(self, paths: ProjectPaths = DEFAULT_PATHS, python: str = sys.executable) -> None:
        self.paths = paths
        self.python = python
        self._env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(paths.root / "src"), os.environ.get("PYTHONPATH", "")]))}

    def _run(self, name: str, command: list[str]) -> int:
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
        with open(self.paths.logs_dir / f"{name}.log", "w") as log:
            rc = subprocess.run(command, cwd=self.paths.root, env=self._env, stdout=log, stderr=subprocess.STDOUT).returncode
        line = f"[{time.strftime('%H:%M:%S')}] {name} exit={rc} ({(time.time() - t0) / 60:.1f} min)"
        print(line, flush=True)
        with open(self.paths.logs_dir / "_resumo.txt", "a") as f:
            f.write(line + "\n")
        return rc

    def run_study(self, study: StudySpec) -> int:
        return self._run(study.name, [self.python, "-m", study.module, *study.args])

    def run_module(self, name: str, module: str, *args: str) -> int:
        return self._run(name, [self.python, "-m", module, *args])

    def run_all(self, studies: tuple[StudySpec, ...] = STUDIES) -> int:
        rc = 0
        progress = ProgressTracker(len(studies), log_file=self.paths.logs_dir / "_progresso.txt")
        for study in studies:
            progress.start(study.name)
            study_rc = self.run_study(study)
            progress.finish(study.name, study_rc)
            rc = study_rc or rc
        return rc


@dataclass
class TableSynchronizer:
    """Copies result tables into the versioned `tcc-writing/tables/` and rebuilds dashboards that depend on them."""

    runner: StudyRunner
    paths: ProjectPaths = DEFAULT_PATHS
    copies: dict[str, str] = field(default_factory=lambda: dict(TABLE_COPIES))

    def sync(self) -> None:
        res, tab = self.paths.results_dir, self.paths.tables_dir
        tab.mkdir(parents=True, exist_ok=True)
        missing = []
        for dest, origin in self.copies.items():
            src = res / origin
            if src.exists():
                shutil.copy2(src, tab / dest)
            else:
                missing.append(origin)
        diag = [res / d / "tabelas" / "modelos_pareado_resumo.csv" for d in ("injecao_modelos_diag", "injecao_modelos_diag24")]
        if all(p.exists() for p in diag):
            pd.concat([pd.read_csv(p) for p in diag]).to_csv(tab / "decomposicao_regime_modelo.csv", index=False)
        else:
            missing.append("injecao_modelos_diag*/tabelas/modelos_pareado_resumo.csv")
        if (res / "falhas_dificeis" / "runs").exists():
            subprocess.run(
                [self.runner.python, "tcc-writing/hard_by_subset.py", "--results", "results/falhas_dificeis",
                 "--out", str(tab / "falhas_dificeis_por_subconjunto.csv")],
                cwd=self.paths.root, env=self.runner._env, check=False, stdout=subprocess.DEVNULL)
        self.runner.run_module("sync_plotly_hub", "pdm_cmaps.reporting.plotly_hub",
                               "--results", "results", "--output-dir", "results/plotly", "--repo", ".")
        if missing:
            print("Atenção: fontes ausentes (não copiadas):", *missing, sep="\n  ")
        print("Sincronização concluída. Revise os números da monografia (Capítulo 5) contra tcc-writing/tables/.")


def run_studies_and_sync(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    """Used by `--all`: every study followed by the table/dashboard synchronization."""
    runner = StudyRunner(paths)
    rc = runner.run_all()
    TableSynchronizer(runner, paths).sync()
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so-sincronizar", action="store_true")
    args = ap.parse_args(argv)
    runner = StudyRunner()
    rc = 0 if args.so_sincronizar else runner.run_all()
    TableSynchronizer(runner).sync()
    return rc


if __name__ == "__main__":
    sys.exit(main())
