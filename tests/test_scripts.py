"""O runner de planos (scripts/rodar.py): tentativas, retomada, dependências entre raias, pool e os planos reais."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rodar import Plan, Runner, Step, main, status  # noqa: E402

PY = sys.executable


def sh(code: str) -> tuple[str, ...]:
    return (PY, "-c", code)


def append_to(root: Path, name: str) -> str:
    """Código Python que registra uma execução do passo `name` (para contar quantas vezes rodou)."""
    return f"open({str(root / 'execucoes.txt')!r}, 'a').write({name!r} + chr(10))"


def runs(root: Path) -> list[str]:
    f = root / "execucoes.txt"
    return f.read_text().split() if f.exists() else []


def test_runs_steps_in_lane_order_and_marks_them_done(tmp_path):
    plan = Plan("p", "", lanes={"A": [Step("um", sh(append_to(tmp_path, "um"))), Step("dois", sh(append_to(tmp_path, "dois")))]})
    assert Runner(plan, root=tmp_path).run() == 0
    assert runs(tmp_path) == ["um", "dois"]
    assert {p.name for p in (tmp_path / "results/logs/p/done").iterdir()} == {"um", "dois"}


def test_retries_a_failing_step_then_succeeds(tmp_path):
    flaky = f"import pathlib,sys; f=pathlib.Path({str(tmp_path / 'tentativa')!r}); n=int(f.read_text() or 0) if f.exists() else 0; f.write_text(str(n+1)); sys.exit(1 if n < 1 else 0)"
    plan = Plan("p", "", lanes={"A": [Step("instavel", sh(flaky))]})
    assert Runner(plan, tries=3, root=tmp_path).run() == 0
    events = (tmp_path / "results/logs/p/progresso.tsv").read_text()
    assert "falha" in events and "\tok\t2\t" in events  # falhou na 1ª, passou na 2ª


def test_a_step_whose_result_check_fails_counts_as_failure(tmp_path):
    plan = Plan("p", "", lanes={"A": [Step("sem_resultado", sh("pass"), check=lambda ctx: False)]})
    assert Runner(plan, tries=2, root=tmp_path).run() == 1
    assert "verificação do resultado falhou" in (tmp_path / "results/logs/p/progresso.tsv").read_text()


def test_resume_skips_finished_steps_and_redo_runs_them_again(tmp_path):
    plan = Plan("p", "", lanes={"A": [Step("um", sh(append_to(tmp_path, "um")))]})
    Runner(plan, root=tmp_path).run()
    Runner(plan, root=tmp_path).run()  # retomada: nada a fazer
    assert runs(tmp_path) == ["um"]
    Runner(plan, redo=True, root=tmp_path).run()
    assert runs(tmp_path) == ["um", "um"]


def test_failure_gives_up_cancels_dependents_and_skips_final_steps(tmp_path):
    plan = Plan("p", "", lanes={
        "A": [Step("quebra", sh("raise SystemExit(3)"))],
        "B": [Step("depende", sh(append_to(tmp_path, "depende")), after=("quebra",)), Step("independente", sh(append_to(tmp_path, "independente")))]},
        final=[Step("sync", sh(append_to(tmp_path, "sync")))])
    assert Runner(plan, tries=2, root=tmp_path).run() == 1
    assert runs(tmp_path) == ["independente"]  # o dependente foi cancelado, a outra tarefa seguiu, o final não rodou
    assert "cancelada" in (tmp_path / "results/logs/p/progresso.tsv").read_text()


def test_dependency_waits_for_the_step_in_another_lane(tmp_path):
    plan = Plan("p", "", lanes={
        "A": [Step("depois", sh(append_to(tmp_path, "depois")), after=("antes",))],
        "B": [Step("antes", sh(f"import time; time.sleep(0.5); {append_to(tmp_path, 'antes')}"))]})
    assert Runner(plan, root=tmp_path).run() == 0
    assert runs(tmp_path) == ["antes", "depois"]


def test_pool_tasks_run_exactly_once_across_lanes(tmp_path):
    pool = [Step(f"t{i}", sh(append_to(tmp_path, f"t{i}"))) for i in range(6)]
    plan = Plan("p", "", lanes={"A": [], "B": []}, pool=pool)
    assert Runner(plan, root=tmp_path).run() == 0
    assert sorted(runs(tmp_path)) == [f"t{i}" for i in range(6)]


def test_clean_patterns_are_removed_before_each_attempt(tmp_path):
    old = tmp_path / "results" / "runs"
    old.mkdir(parents=True)
    (old / "antigo.csv").write_text("x")
    plan = Plan("p", "", lanes={"A": [Step("limpa", sh("pass"), clean=("results/runs",))]})
    Runner(plan, root=tmp_path).run()
    assert not old.exists()


def test_dry_run_executes_and_writes_nothing(tmp_path, capsys):
    plan = Plan("p", "descrição", lanes={"A": [Step("x", sh(append_to(tmp_path, "x")), est=90)]})
    assert Runner(plan, dry_run=True, root=tmp_path).run() == 0
    assert "1min" in capsys.readouterr().out and runs(tmp_path) == [] and not (tmp_path / "results").exists()


def test_status_reports_done_and_pending_steps(tmp_path, capsys):
    plan = Plan("p", "", lanes={"A": [Step("feito", sh("pass")), Step("quebra", sh("raise SystemExit(1)")), Step("nunca", sh("pass"), after=("quebra",))]})
    Runner(plan, tries=1, root=tmp_path).run()
    status(plan, root=tmp_path)
    out = capsys.readouterr().out
    assert "1/3 passos concluídos" in out and "DESISTIU" in out and "feito" in out


def test_progress_line_has_counter_and_estimate(tmp_path, capsys):
    plan = Plan("p", "", lanes={"A": [Step("um", sh("pass"), est=10), Step("dois", sh("pass"), est=10)]})
    Runner(plan, root=tmp_path).run()
    out = capsys.readouterr().out
    assert "[1/2] um ... ok" in out and "[2/2] dois ... ok" in out and "restante" in out


def test_every_real_plan_is_well_formed(capsys):
    from planos import PLANOS

    for name in PLANOS:
        assert main(["rodar", name, "--dry-run"]) == 0, name
    assert "pytest" in capsys.readouterr().out  # o plano `tudo` roda a suíte


def test_the_everything_plan_uses_the_same_study_list_as_update_all():
    from planos import PLANOS

    from pdm_cmaps.cli.update_all import STUDIES

    steps = {s.name for s in PLANOS["tudo"]().all_steps()}
    assert {s.name for s in STUDIES} <= steps


@pytest.mark.parametrize("cmd", [["lista"], ["status", "tudo"]])
def test_list_and_status_commands_run(cmd, capsys):
    assert main(cmd) == 0
    assert capsys.readouterr().out
