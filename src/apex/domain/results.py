"""Records produced by a history-matching run."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One forecast-and-score step of a run.

    Attributes
    ----------
    iteration
        0 for the prior forecast, then 1, 2, ... for each analysis.
    data_mismatch_mean, data_mismatch_std
        Data mismatch ``(d_j - y_j)^T C_d^-1 (d_j - y_j)`` over members, against the
        un-inflated perturbed observations of the run.
    n_members
        Members that were simulated successfully.
    accepted
        Whether the iterate was kept (always true for fixed schedules).
    damping
        Damping used to produce the iterate, for iterative methods; ``None`` otherwise.
    data_mismatch_by_type
        Each data type's share of ``data_mismatch_mean`` (the shares add up to it).
    metrics
        Member means of the performance metrics, ``{metric: {label: mean}}``,
        e.g. ``{"normalized_data_mismatch": {"all": 1.2}, "rmse": {"permx": 0.4, "all": 0.4}}``.
        ``None`` in runs saved before metrics were recorded.
    """

    iteration: int
    data_mismatch_mean: float
    data_mismatch_std: float
    n_members: int
    accepted: bool = True
    damping: float | None = None
    data_mismatch_by_type: dict[str, float] | None = None
    metrics: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True, slots=True)
class HistoryMatchResult:
    """Outcome of a history-matching run.

    Attributes
    ----------
    posterior
        Final state ensemble, ``(n_state, n_members)``.
    member_ids
        Original member id of each posterior column (members may have been dropped).
    history
        One record per forecast, prior first.
    converged
        For iterative methods, whether the convergence test stopped the run; ``None`` for
        fixed schedules.
    complete
        Whether the run finished: its schedule or iteration budget, or a stop by cross
        validation (``False`` after a stop requested with ``max_steps``).
    selected_iteration
        With cross validation, the iterate with the lowest CV normalized data mismatch,
        which is the posterior; ``None`` otherwise.
    stopped_by_cv
        Whether the cross-validation stopping rule ended the run.
    """

    posterior: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    history: tuple[IterationRecord, ...]
    converged: bool | None
    complete: bool
    selected_iteration: int | None = None
    stopped_by_cv: bool = False
