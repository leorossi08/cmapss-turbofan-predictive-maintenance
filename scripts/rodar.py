#!/usr/bin/env python3
"""Executa um *plano* de estudos longos: passos em sequência ou em raias paralelas, retomável, com tentativas e progresso.

    python scripts/rodar.py lista                      # planos disponíveis
    python scripts/rodar.py rodar tudo                 # executa (retoma de onde parou se interrompido)
    python scripts/rodar.py rodar tudo --dry-run       # só mostra os comandos, nada é executado
    python scripts/rodar.py status tudo                # o que já terminou, o que está rodando, o que falhou

Os planos são dados em `planos.py`; este arquivo é só o motor. Cada passo tem um comando, uma verificação de sucesso
(o comando pode sair com código 0 sem ter gerado o resultado) e uma duração estimada, usada para a previsão do tempo que falta.
Para deixar rodando em segundo plano:  nohup python scripts/rodar.py rodar tudo > /dev/null 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]

from pdm_cmaps.progress import format_duration  # noqa: E402

# Cada processo usa poucos threads: com várias raias e joblib, OpenMP/BLAS ficam em espera ativa e a máquina
# passa de 20 de carga (medido: mais de 7x mais lento). 2 por processo mantém as raias paralelas eficientes.
THREAD_ENV = {k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")} | {"OMP_WAIT_POLICY": "PASSIVE"}


@dataclass
class Context:
    """O que as verificações podem consultar."""

    root: Path
    started: float  # quando o plano começou pela 1ª vez (sobrevive a retomadas); serve para exigir arquivos mais novos que isso


@dataclass(frozen=True)
class Step:
    name: str
    cmd: tuple[str, ...]
    check: Callable[[Context], bool] = lambda ctx: True
    est: int = 60                      # segundos; só para a previsão de tempo
    clean: tuple[str, ...] = ()        # padrões (glob) apagados antes de cada tentativa, p.ex. runs antigos que seriam reaproveitados
    after: tuple[str, ...] = ()        # passos (de outras raias) que precisam ter terminado antes


@dataclass
class Plan:
    name: str
    description: str
    lanes: dict[str, list[Step]] = field(default_factory=dict)  # raias rodam em paralelo; os passos de cada uma, em sequência
    pool: list[Step] = field(default_factory=list)             # tarefas independentes: cada raia que acabar pega a próxima livre
    final: list[Step] = field(default_factory=list)            # só rodam se nada falhou
    env: dict[str, str] = field(default_factory=dict)

    def all_steps(self) -> list[Step]:
        return [s for lane in self.lanes.values() for s in lane] + self.pool + self.final


class Runner:
    def __init__(self, plan: Plan, tries: int = 3, dry_run: bool = False, redo: bool = False, root: Path = ROOT):
        self.plan, self.tries, self.dry_run, self.root = plan, tries, dry_run, root
        self.dir = root / "results" / "logs" / plan.name
        self.done_dir, self.events = self.dir / "done", self.dir / "progresso.tsv"
        self.lock, self.cond = threading.Lock(), threading.Condition()
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.actual: dict[str, float] = {}
        self.t0 = time.time()
        steps = plan.all_steps()
        names = [s.name for s in steps]
        assert len(set(names)) == len(names), f"passos repetidos em {plan.name}"
        for s in steps:
            assert all(a in names for a in s.after), f"{s.name}: 'after' aponta para passo inexistente"
        if redo and not dry_run:
            shutil.rmtree(self.done_dir, ignore_errors=True)
        if not dry_run:
            self.done_dir.mkdir(parents=True, exist_ok=True)
            self.done |= {p.name for p in self.done_dir.iterdir()}
            start = self.dir / "inicio"
            start.touch(exist_ok=True)
            self.ctx = Context(root, start.stat().st_mtime)
        else:
            self.ctx = Context(root, self.t0)

    # ── execução ──────────────────────────────────────────────────────────
    def run(self) -> int:
        if self.dry_run:
            return self._dry_run()
        pool = queue.Queue()
        for s in self.plan.pool:
            pool.put(s)
        lanes = [threading.Thread(target=self._lane, args=(name, steps, pool), name=name) for name, steps in self.plan.lanes.items()]
        self._event("-", "geral", "inicio" if not self.done else "retomada", "", f"{len(self.done)} passos já concluídos")
        for t in lanes:
            t.start()
        for t in lanes:
            t.join()
        if not self.failed:
            for s in self.plan.final:
                self._step("final", s)
        self._event("-", "geral", "fim_com_falhas" if self.failed else "fim_ok", "", ", ".join(sorted(self.failed)))
        return 1 if self.failed else 0

    def _lane(self, lane: str, steps: list[Step], pool: queue.Queue) -> None:
        for s in steps:
            self._step(lane, s)
        while True:
            try:
                s = pool.get_nowait()
            except queue.Empty:
                return
            self._step(lane, s)

    def _step(self, lane: str, s: Step) -> None:
        if s.name in self.done:
            self._event(lane, s.name, "pulada", "", "já concluída")
            return
        if s.after and not self._wait_for(s.after):
            self._finish(s.name, ok=False)
            self._event(lane, s.name, "cancelada", "", f"depende de {', '.join(s.after)}, que falhou")
            return
        env = os.environ | THREAD_ENV | self.plan.env | {"PYTHONPATH": os.pathsep.join(filter(None, [str(self.root / "src"), os.environ.get("PYTHONPATH", "")]))}
        log = self.root / "results" / "logs" / f"{self.plan.name}_{s.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, self.tries + 1):
            for pattern in s.clean:
                for p in glob(str(self.root / pattern)):
                    shutil.rmtree(p) if Path(p).is_dir() else Path(p).unlink()
            t = time.time()
            self._event(lane, s.name, "inicio", str(attempt), f"est={format_duration(s.est)}")
            with open(log, "w") as f:
                rc = subprocess.run(s.cmd, cwd=self.root, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=lambda: os.nice(5)).returncode
            secs = time.time() - t
            if rc == 0 and s.check(self.ctx):
                self.actual[s.name] = secs
                (self.done_dir / s.name).touch()
                self._finish(s.name, ok=True)
                self._event(lane, s.name, "ok", str(attempt), format_duration(secs))
                return
            self._event(lane, s.name, "falha", str(attempt), f"codigo={rc}{'' if rc else ', verificação do resultado falhou'} ({format_duration(secs)}); ver {log.relative_to(self.root)}")
        self._finish(s.name, ok=False)
        self._event(lane, s.name, "desistiu", str(self.tries), "")

    def _finish(self, name: str, ok: bool) -> None:
        with self.cond:
            (self.done if ok else self.failed).add(name)
            self.cond.notify_all()

    def _wait_for(self, names: tuple[str, ...]) -> bool:
        with self.cond:
            self.cond.wait_for(lambda: all(n in self.done or n in self.failed for n in names))
            return all(n in self.done for n in names)

    # ── observabilidade ───────────────────────────────────────────────────
    def _eta(self) -> str:
        steps = self.plan.all_steps()
        finished = [s for s in steps if s.name in self.actual]
        pending = sum(s.est for s in steps if s.name not in self.done and s.name not in self.failed)
        if not finished:
            return "restante: estimado após o 1º passo"
        ratio = sum(self.actual[s.name] for s in finished) / max(1, sum(s.est for s in finished))  # real / estimado, dos passos concluídos
        left = pending * ratio / max(1, len(self.plan.lanes))
        return f"restante ~{format_duration(left)}" if pending else "concluído"

    def _event(self, lane: str, step: str, event: str, attempt: str, detail: str) -> None:
        total = len(self.plan.all_steps())
        with self.lock:
            line = (f"[{time.strftime('%H:%M:%S')}] [{len(self.done)}/{total}] {step} ... {event}"
                    f"{f' (tentativa {attempt})' if attempt else ''}{f' {detail}' if detail else ''}"
                    f" | decorrido {format_duration(time.time() - self.t0)} | {self._eta()}")
            print(line, flush=True)
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.events, "a") as f:
                f.write("\t".join([time.strftime("%FT%T"), lane, step, event, attempt, detail]) + "\n")

    def _dry_run(self) -> int:
        print(f"Plano '{self.plan.name}': {self.plan.description}\n")
        for lane, steps in [*self.plan.lanes.items(), ("pool (qualquer raia livre)", self.plan.pool), ("final (só se nada falhar)", self.plan.final)]:
            if steps:
                print(f"raia {lane}:")
            for s in steps:
                dep = f"  [depois de {', '.join(s.after)}]" if s.after else ""
                clean = f"  [apaga antes: {', '.join(s.clean)}]" if s.clean else ""
                print(f"  - {s.name} (~{format_duration(s.est)}){dep}{clean}\n      {' '.join(s.cmd)}")
        total = sum(s.est for s in self.plan.all_steps())
        print(f"\n{len(self.plan.all_steps())} passos, ~{format_duration(total)} de trabalho somado (menos com raias paralelas).")
        return 0


def status(plan: Plan, root: Path = ROOT) -> int:
    """Mostra o andamento lendo só os arquivos que o Runner deixou (não cria nada)."""
    run_dir = root / "results" / "logs" / plan.name
    events, done_dir = run_dir / "progresso.tsv", run_dir / "done"
    steps = plan.all_steps()
    if not events.exists():
        print(f"Plano '{plan.name}' ainda não foi executado.")
        return 0
    lines = events.read_text().splitlines()
    last = {}
    for line in lines:
        _, _lane, step, event, *_ = line.split("\t")
        last[step] = event
    done = {p.name for p in done_dir.iterdir()} if done_dir.exists() else set()
    print(f"Plano '{plan.name}': {len(done & {s.name for s in steps})}/{len(steps)} passos concluídos")
    labels = {"inicio": "RODANDO", "falha": "falhou (vai tentar de novo)", "desistiu": "DESISTIU", "cancelada": "cancelada"}
    for s in steps:
        print(f"  {'ok' if s.name in done else labels.get(last.get(s.name, ''), 'pendente'):28} {s.name}")
    print(f"\nÚltimos eventos ({events.relative_to(root)}):")
    for line in lines[-5:]:
        print("  " + line.replace("\t", " | "))
    return 0


def main(argv: list[str] | None = None) -> int:
    from planos import PLANOS

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lista", help="lista os planos")
    for name, help_ in (("rodar", "executa um plano (retoma se interrompido)"), ("status", "mostra o andamento de um plano")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("plano", choices=sorted(PLANOS))
        if name == "rodar":
            p.add_argument("--dry-run", action="store_true", help="mostra os comandos sem executar")
            p.add_argument("--refazer", action="store_true", help="ignora os passos já concluídos")
            p.add_argument("--tentativas", type=int, default=3, help="tentativas por passo (padrão: 3)")
    a = ap.parse_args(argv)
    if a.cmd == "lista":
        for name in sorted(PLANOS):
            print(f"{name:22} {PLANOS[name]().description}")
        return 0
    plan = PLANOS[a.plano]()
    if a.cmd == "status":
        return status(plan)
    return Runner(plan, tries=a.tentativas, dry_run=a.dry_run, redo=a.refazer).run()


if __name__ == "__main__":
    sys.exit(main())
