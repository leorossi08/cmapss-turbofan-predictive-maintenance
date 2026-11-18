"""Progress with time estimate for long, multi-step runs (pipeline phases, `--all`, `update_all`).

Every step prints one line such as
    [3/13] 06_fault_detection ... iniciando | decorrido 1h12min | restante ~2h05min (média dos passos concluídos)
so whoever is watching the terminal (or `tail -f results/logs/_progresso.txt`) knows what is running and how long is left.

The estimate is the mean duration of the finished steps times the steps still to run. Steps differ a lot in cost,
so it is a rough guide, not a promise; it is hidden until one step has finished.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path


def format_duration(seconds: float) -> str:
    """90 -> '1min', 4500 -> '1h15min', 20 -> '20s'."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}min" if hours else f"{minutes}min"


class ProgressTracker:
    """Counts finished steps and estimates the time left. Inject `clock` and `emit` to test without sleeping."""

    def __init__(
        self,
        total: int,
        log_file: Path | None = None,
        clock: Callable[[], float] = time.time,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self.total = total
        self.log_file = log_file
        self._clock = clock
        self._emit = emit or (lambda line: print(line, flush=True))
        self._t0 = clock()
        self._step_start = self._t0
        self._durations: list[float] = []

    @property
    def done(self) -> int:
        return len(self._durations)

    def remaining_seconds(self) -> float | None:
        """Mean finished-step duration x steps left; None before the first step finishes."""
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations) * (self.total - self.done)

    def start(self, name: str) -> None:
        self._step_start = self._clock()
        self._write(f"[{self.done + 1}/{self.total}] {name} ... iniciando | {self._timing()}")

    def finish(self, name: str, returncode: int = 0) -> None:
        self._durations.append(self._clock() - self._step_start)
        status = "ok" if returncode == 0 else f"FALHOU (código {returncode})"
        self._write(f"[{self.done}/{self.total}] {name} ... {status} em {format_duration(self._durations[-1])} | {self._timing()}")

    def _timing(self) -> str:
        text = f"decorrido {format_duration(self._clock() - self._t0)}"
        left = self.remaining_seconds()
        if left is None:
            return text + " | restante: estimado após o 1º passo"
        return text + (f" | restante ~{format_duration(left)} (média dos passos concluídos)" if left > 0 else " | concluído")

    def _write(self, line: str) -> None:
        stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
        self._emit(stamped)
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(stamped + "\n")
