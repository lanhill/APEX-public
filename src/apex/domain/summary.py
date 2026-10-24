"""Summaries of finished (or stopped) runs, for reports across case studies.

A :class:`RunSummary` is workflow-neutral: a sequence of steps (history matching: the
accepted iterates; optimization, later: the accepted iterates of the controls) and, per
step, statistics over the ensemble of every performance metric. Summarizers per workflow
build them (``apex.core.summaries``); ``apex.reporting.run_summary`` writes them.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StepSummary:
    """One step of a run.

    Attributes
    ----------
    iteration
        The step's number in the run's history (``iter_<k>``); 0 is the prior.
    n_members
        Members scored at this step.
    rejected_before
        Trials rejected since the previous step (iterative methods).
    control
        The method's control parameter the step ran with (damping; GN-EnRML's step
        length), if it has one.
    label
        A name for the step where a number says little (a prediction's ensemble, e.g.
        ``"posterior"``); ``iteration`` then only orders the steps.
    """

    iteration: int
    n_members: int
    rejected_before: int = 0
    control: float | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class MetricStatistics:
    """Statistics over the members of one metric and label at one step.

    ``p10`` and ``p90`` are the 10th and 90th percentiles; ``std`` uses ``ddof = 1``. A
    value that is not a statistic over members (a forecast score) has ``mean`` only, the
    others NaN.
    """

    metric: str
    label: str
    iteration: int
    mean: float
    std: float
    median: float
    p10: float
    p90: float


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What a run did and how well, step by step.

    Attributes
    ----------
    workflow
        The summarizer that built it (``"history_matching"``, ...).
    name, path
        The run's name and directory.
    method
        The method, with its strategy and localization, e.g. ``"gies (rlmmac)"``.
    status
        ``"converged"``, ``"complete"``, ``"stopped by cross validation"``,
        ``"stopped"`` (resumable) or ``"running or interrupted"``.
    steps
        The steps summarised, in order.
    metrics
        Statistics per metric, label and step.
    selected_iteration
        The step taken as the result (the posterior: the last step, or the minimum-CV
        iterate).
    forecasts
        Ensemble forecasts the run took, rejected trials included.
    notes
        Remarks, e.g. metrics computed after the run from its checkpoints.
    """

    workflow: str
    name: str
    path: Path
    method: str
    status: str
    steps: tuple[StepSummary, ...]
    metrics: tuple[MetricStatistics, ...]
    selected_iteration: int | None = None
    forecasts: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def step_of(self, iteration: int | None) -> int | None:
        """Return the step number of history record ``iteration``: its place among the steps.

        Steps are numbered 0, 1, 2, ... (0: the prior), rejected trials not counted, so
        a run of 10 iterations ends at step 10 however many trials it rejected; the
        record numbers (``iter_<k>``) count every trial.
        """
        for number, step in enumerate(self.steps):
            if step.iteration == iteration:
                return number
        return None

    def labels(self) -> dict[str, list[str]]:
        """``{metric: [label, ...]}`` in the order first seen."""
        found: dict[str, list[str]] = {}
        for row in self.metrics:
            labels = found.setdefault(row.metric, [])
            if row.label not in labels:
                labels.append(row.label)
        return found

    def value(self, metric: str, label: str, iteration: int) -> MetricStatistics | None:
        """Return the statistics of ``metric``/``label`` at ``iteration``, if recorded."""
        for row in self.metrics:
            if (row.metric, row.label, row.iteration) == (metric, label, iteration):
                return row
        return None
