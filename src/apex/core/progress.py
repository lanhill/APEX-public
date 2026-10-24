"""Progress reporting of ensemble runs.

A workflow tells a :class:`ProgressReporter` what it is doing: which iteration of how
many at most, which forecast, how many of its members have finished, and each scored
record. The reporter decides how to show it (a terminal progress bar, log lines, an API
stream); the workflow never imports a display library. :class:`NullProgress`, the
default, shows nothing.

Progress never influences a run: no random draws, checkpoints or arrays depend on it.
The interface is not specific to history matching, so optimization workflows can use it.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from apex.domain.results import IterationRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IterationProgress:
    """Where an algorithm is in its iteration budget.

    Attributes
    ----------
    completed
        Iterations finished, in the units of the budget (e.g. ES-MDA steps, GIES outer
        iterations).
    total
        The budget: the most iterations the run can take (it may stop earlier, e.g. on
        convergence or by cross validation); ``None`` if unbounded.
    current
        What is being worked on now, e.g. ``"prior"``, ``"iteration 3"``,
        ``"iteration 3, trial 2 of at most 4"``, or ``"done"``.
    unit
        What ``completed`` and ``total`` count, for displays (``"ensembles"`` for a
        prediction).
    upper_bound
        Whether ``total`` is only an upper bound (an iteration budget the run may stop
        short of), or exact.
    """

    completed: int
    total: int | None
    current: str
    unit: str = "iterations"
    upper_bound: bool = True


class ReportsProgress(Protocol):
    """An algorithm that can say where it is in its budget (optional)."""

    def progress(self) -> IterationProgress:
        """Return the current position; valid after ``start`` or ``restore_state``."""
        ...


class ProgressReporter(Protocol):
    """Receives the progress of a run. Calls come from the workflow's process, in order.

    A run calls :meth:`run_started` once, then for every forecast
    :meth:`forecast_started`, :meth:`member_finished` per member, and
    :meth:`forecast_finished`; :meth:`record_scored` for each scored forecast, and
    :meth:`iteration_progress` whenever the algorithm moves on; finally
    :meth:`run_finished` (not called if the run raises).
    """

    def run_started(self, *, algorithm: str, progress: IterationProgress | None) -> None:
        """Report that a run starts or resumes (``progress`` ``None``: the algorithm cannot say)."""
        ...

    def iteration_progress(self, progress: IterationProgress) -> None:
        """Report where the algorithm is, after each of its steps."""
        ...

    def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
        """Report that a forecast of ``n_members`` starts (``label``: e.g. ``"mean model"``).

        ``iteration`` is the forecast's iteration, or ``None`` when it has none (a
        prediction's forecasts).
        """
        ...

    def member_finished(self, *, member_id: int, ok: bool) -> None:
        """Report that one member of the current forecast finished, successfully or not."""
        ...

    def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
        """Report that the current forecast finished."""
        ...

    def record_scored(self, record: IterationRecord) -> None:
        """Report a scored forecast ensemble (data mismatch mean and std, metrics)."""
        ...

    def run_finished(self, *, status: str) -> None:
        """Report the end of the run, with its ``status`` (e.g. ``"converged"``)."""
        ...


class NullProgress:
    """A reporter that shows nothing (the default)."""

    def run_started(self, *, algorithm: str, progress: IterationProgress | None) -> None:
        """Ignore."""

    def iteration_progress(self, progress: IterationProgress) -> None:
        """Ignore."""

    def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
        """Ignore."""

    def member_finished(self, *, member_id: int, ok: bool) -> None:
        """Ignore."""

    def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
        """Ignore."""

    def record_scored(self, record: IterationRecord) -> None:
        """Ignore."""

    def run_finished(self, *, status: str) -> None:
        """Ignore."""


class GuardedProgress:
    """Wraps a reporter so that its failures never stop a run.

    A display may fail for reasons unrelated to the run (a closed terminal, a broken
    pipe on standard error). The first failure is logged as ``progress_failed`` and the
    reporter is then dropped; the run goes on without progress.
    """

    def __init__(self, reporter: ProgressReporter) -> None:
        self._reporter: ProgressReporter = reporter

    def _call(self, report: Callable[[ProgressReporter], None]) -> None:
        try:
            report(self._reporter)
        except Exception as error:  # noqa: BLE001  (any display fault: see the class docstring)
            logger.warning(
                "progress display failed; continuing without it: %s",
                error,
                extra={"event": "progress_failed", "error": repr(error)},
            )
            self._reporter = NullProgress()

    def run_started(self, *, algorithm: str, progress: IterationProgress | None) -> None:
        """Forward, guarded."""
        self._call(lambda r: r.run_started(algorithm=algorithm, progress=progress))

    def iteration_progress(self, progress: IterationProgress) -> None:
        """Forward, guarded."""
        self._call(lambda r: r.iteration_progress(progress))

    def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
        """Forward, guarded."""
        self._call(
            lambda r: r.forecast_started(iteration=iteration, label=label, n_members=n_members)
        )

    def member_finished(self, *, member_id: int, ok: bool) -> None:
        """Forward, guarded."""
        self._call(lambda r: r.member_finished(member_id=member_id, ok=ok))

    def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
        """Forward, guarded."""
        self._call(lambda r: r.forecast_finished(n_ok=n_ok, n_failed=n_failed, seconds=seconds))

    def record_scored(self, record: IterationRecord) -> None:
        """Forward, guarded."""
        self._call(lambda r: r.record_scored(record))

    def run_finished(self, *, status: str) -> None:
        """Forward, guarded."""
        self._call(lambda r: r.run_finished(status=status))
