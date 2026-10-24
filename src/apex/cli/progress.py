"""Progress displays of the command line.

:func:`progress_display` picks one:

- :class:`RichProgress` on an interactive terminal: an overall bar over the algorithm's
  iteration budget, with the last data mismatch (mean ± std over members), and a bar over the
  members of the current forecast, with failures and an estimate of the time left;
- :class:`LineProgress` otherwise (``nohup``, a log file, a background case study): one
  timestamped line per finished forecast and per scored ensemble;
- :class:`~apex.core.progress.NullProgress` when switched off (``--no-progress``).

Both write to standard error, so standard output keeps the final tables.
"""

import datetime as dt
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from apex.core.progress import IterationProgress, NullProgress, ProgressReporter
from apex.domain.results import IterationRecord


def describe_record(record: IterationRecord) -> str:
    """Summarise a scored ensemble: ``data mismatch 1.21e+04 ± 3.2e+03 (accepted), 100 members``."""
    text = f"data mismatch {record.data_mismatch_mean:.4g} ± {record.data_mismatch_std:.3g}"
    if record.damping is not None:  # iterative methods accept or reject their trials
        text += " (accepted)" if record.accepted else " (rejected)"
    text += f", {record.n_members} members"
    cv = (record.metrics or {}).get("cv_normalized_data_mismatch", {}).get("all")
    if cv is not None:
        text += f"; CV mismatch {cv:.4g}"
    return text


def _budget(progress: IterationProgress | None) -> str:
    if progress is None:
        return ""
    # "3 of ≤4 iterations done": what is finished, out of the budget (the current step's
    # number, beside it, is one more).
    bound = "≤" if progress.upper_bound else ""
    total = "?" if progress.total is None else f"{bound}{progress.total}"
    done = "" if progress.current == "done" else " done"  # "done (4 of ≤4 iterations)"
    return f"{progress.completed} of {total} {progress.unit}{done}"


def _forecast_name(label: str, iteration: int | None) -> str:
    """``"ensemble forecast 3"``, or ``"prior forecast"`` for a forecast without iteration."""
    return f"{label} forecast" if iteration is None else f"{label} forecast {iteration}"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


class LineProgress:
    """Timestamped progress lines, for output that is not a terminal.

    Parameters
    ----------
    stream
        Where to write (default: standard error).
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = sys.stderr if stream is None else stream
        self._algorithm = ""
        self._position: IterationProgress | None = None
        self._forecast = ""

    def _say(self, text: str) -> None:
        stamp = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
        print(f"[{stamp}] {text}", file=self.stream, flush=True)

    def _where(self) -> str:
        if self._position is None:
            return self._algorithm
        return f"{self._algorithm} {self._position.current} ({_budget(self._position)})"

    def run_started(self, *, algorithm: str, progress: IterationProgress | None) -> None:
        """Print the algorithm and its budget."""
        self._algorithm, self._position = algorithm, progress
        self._say(f"{self._where()}: running")

    def iteration_progress(self, progress: IterationProgress) -> None:
        """Remember where the algorithm is, for the next lines."""
        self._position = progress

    def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
        """Remember the forecast, for its summary line."""
        self._forecast = _forecast_name(label, iteration)

    def member_finished(self, *, member_id: int, ok: bool) -> None:
        """Nothing per member: one line per forecast is enough for a log."""

    def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
        """Print the forecast's members and duration."""
        failed = f" ({n_failed} failed)" if n_failed else ""
        self._say(
            f"{self._where()}: {self._forecast}: {n_ok}/{n_ok + n_failed} members ok{failed} "
            f"in {_duration(seconds)}"
        )

    def record_scored(self, record: IterationRecord) -> None:
        """Print the scored ensemble's data mismatch (mean ± std)."""
        # ``iter`` as in the history table: the forecast's number in the results, which for
        # GIES with retries is not its outer iteration, so no budget position here.
        self._say(f"{self._algorithm} iter {record.iteration}: {describe_record(record)}")

    def run_finished(self, *, status: str) -> None:
        """Print how the run ended."""
        self._say(f"{self._where()}: {status}")


class RichProgress:
    """Two progress bars on a terminal: the iteration budget and the current forecast.

    Use it as a context manager: the display starts on entry, so that log messages
    written meanwhile appear above the bars, and stops on exit, even after an error.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._bars = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[info]}"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            console=Console(stderr=True) if console is None else console,
        )
        self._algorithm = ""
        self._overall: TaskID | None = None
        self._forecast: TaskID | None = None
        self._failed = 0

    def __enter__(self) -> "RichProgress":
        """Start the display."""
        self._bars.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the display, leaving the last state of the bars on the terminal."""
        self._bars.stop()

    def _show_position(self, progress: IterationProgress) -> None:
        assert self._overall is not None
        self._bars.update(
            self._overall,
            description=f"{self._algorithm} {progress.current}",
            completed=progress.completed,
            total=progress.total,
        )

    def run_started(self, *, algorithm: str, progress: IterationProgress | None) -> None:
        """Add the overall bar."""
        self._algorithm = algorithm
        self._overall = self._bars.add_task(algorithm, total=None, info="")
        if progress is not None:
            self._show_position(progress)

    def iteration_progress(self, progress: IterationProgress) -> None:
        """Move the overall bar."""
        self._show_position(progress)

    def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
        """Replace the forecast bar with one for this forecast."""
        if self._forecast is not None:
            self._bars.remove_task(self._forecast)
        self._failed = 0
        self._forecast = self._bars.add_task(
            f"└ {_forecast_name(label, iteration)}", total=n_members, info=""
        )

    def member_finished(self, *, member_id: int, ok: bool) -> None:
        """Advance the forecast bar; count failures."""
        assert self._forecast is not None
        self._failed += not ok
        info = f"{self._failed} failed" if self._failed else ""
        self._bars.update(self._forecast, advance=1, info=info)

    def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
        """Nothing more to show: the forecast bar is full."""

    def record_scored(self, record: IterationRecord) -> None:
        """Show the last data mismatch (mean ± std) on the overall bar."""
        assert self._overall is not None
        self._bars.update(self._overall, info=f"iter {record.iteration}: {describe_record(record)}")

    def run_finished(self, *, status: str) -> None:
        """Show how the run ended."""
        assert self._overall is not None
        self._bars.update(self._overall, description=f"{self._algorithm} {status}")


@contextmanager
def progress_display(
    enabled: bool = True, stream: TextIO | None = None
) -> Iterator[ProgressReporter]:
    """Yield the progress display for ``stream`` (default: standard error).

    Bars on an interactive terminal, lines otherwise, nothing if not ``enabled``.
    """
    stream = sys.stderr if stream is None else stream
    if not enabled:
        yield NullProgress()
    elif stream.isatty():
        with RichProgress(Console(file=stream)) as bars:
            yield bars
    else:
        yield LineProgress(stream)
