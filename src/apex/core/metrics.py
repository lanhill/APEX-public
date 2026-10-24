"""Performance metrics of an ensemble, evaluated at every forecast of a run.

A :class:`PerformanceMetric` maps a :class:`MetricContext` (the ensemble, its forecast,
the observations and, in synthetic case studies, the true model) to one value per member
for each of its labels, e.g. ``{"permx": (n,), "all": (n,)}``. Metrics are plugins in
:data:`apex.core.plugins.metrics`; adding one means writing and registering a class.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from apex.domain.observations import ObservationSet


@dataclass(frozen=True, slots=True)
class MetricContext:
    """What a metric may look at for one forecast of the ensemble.

    Attributes
    ----------
    state
        The ensemble, ``{variable: (size, n_members)}``.
    predicted
        Its forecast in the analysis data space, ``(n_data, n_members)``.
    observations
        The observations in the same data space, with their variances.
    truth
        The true model ``{variable: (size,)}`` of a synthetic case study; ``None`` in a
        real one.
    grids
        Field shape of each state variable that is a grid field (``[[state.variables]]
        grid``), ``{variable: (nx, ny, nz)}``.
    previous_state
        The ensemble the scored one was updated from, on the same members and in the same
        order, ``{variable: (size, n_members)}``: the last accepted iterate before it (for
        a rejected trial, the iterate it was proposed from). ``None`` for the prior, and
        unless a metric declares ``requires_previous``.
    categorical
        Variables that are codes, not quantities (a parameterization's ``facies``, ADR
        0037): metrics of quantities (RMSE, SSIM, model change) leave them out unless named.
    auxiliary
        Variables that are not reservoir properties, e.g. a model-error model's parameters:
        the quantity metrics leave them out unless named, as ``categorical``.
    """

    state: Mapping[str, NDArray[np.float64]]
    predicted: NDArray[np.float64]
    observations: ObservationSet
    truth: Mapping[str, NDArray[np.float64]] | None = None
    grids: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    previous_state: Mapping[str, NDArray[np.float64]] | None = None
    categorical: frozenset[str] = frozenset()
    auxiliary: frozenset[str] = frozenset()


class PerformanceMetric(Protocol):
    """One performance metric.

    ``requires_truth`` metrics (model-space errors such as RMSE or SSIM against the true
    model) can only run in synthetic case studies; the composition root refuses them in
    real ones. A metric may also set ``requires_previous = True`` to be given the previous
    ensemble (``MetricContext.previous_state``); it is then scored from the first update
    on, and returns ``{}`` for the prior, which has none.
    """

    requires_truth: bool

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{label: (n_members,)}``, or ``{}`` when it is not defined (see above).

        Raises
        ------
        ValueError
            If the context does not fit the metric (e.g. no truth, wrong sizes).
        """
        ...
