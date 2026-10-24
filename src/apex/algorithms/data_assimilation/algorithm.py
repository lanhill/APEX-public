"""The generic ensemble-algorithm protocol and the types it exchanges with the driver.

An :class:`EnsembleAlgorithm` owns its loop and all of its loop state. The workflow
(``workflows.history_matching``) owns what every algorithm shares: forecasting, the
failure policy, scoring and metrics, saving results, and checkpoints. The two talk
through :class:`Step` objects:

.. code-block:: text

    step = algorithm.start(setup)                  # new run
    step = algorithm.restore_state(setup, state)   # resumed run (the pending step)
    while step.requests:
        forecasts = <forecast every request>
        step = algorithm.advance(forecasts)        # records to save, next requests

A new algorithm (LM-EnRML, EnKF, a line search, a multilevel method) is one class that
implements this protocol; it needs no change to the workflow. All arrays keep the
ensemble axis last: states ``(n_state, n)``, data ``(n_data, n)``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.core.diagnostics import Diagnostics, NoDiagnostics
from apex.domain.observations import ObservationSet

#: Member id of a forecast of the ensemble-mean model (not an ensemble member).
MEAN_MODEL_ID = -1

#: How far a plugin has been validated: PET parity, an independent reference
#: (analytic solution, benchmark), or not yet.
Status = Literal["parity", "validated", "experimental"]


class StreamSource(Protocol):
    """Named, checkpointed random streams of a run (``RandomStreams``)."""

    def __getitem__(self, name: str) -> np.random.Generator:
        """Return the stream called ``name`` (e.g. ``"perturbations"``, ``"localization"``)."""
        ...


@dataclass(frozen=True)
class RunSetup:
    """What a run fixes, handed to :meth:`EnsembleAlgorithm.start` and ``restore_state``.

    Attributes
    ----------
    prior
        Prior ensemble ``(n_state, N)``; empty ``(n_state, 0)`` when resuming.
    observations
        Observations in the analysis data space (after data transforms).
    data_error
        Their error covariance ``C_d``.
    streams
        The run's random streams; draw only from these, never from global state.
    diagnostics
        Where to offer optional diagnostics (nothing is saved unless requested).
    """

    prior: NDArray[np.float64]
    observations: ObservationSet
    data_error: DataErrorCovariance
    streams: StreamSource
    diagnostics: Diagnostics = field(default_factory=NoDiagnostics)


@dataclass(frozen=True)
class ForecastRequest:
    """An ensemble to forecast.

    Attributes
    ----------
    name
        Key of its result in the :class:`Forecasts` passed to ``advance``.
    iteration
        Iteration it belongs to (names simulator work directories).
    state
        The ensemble ``(n_state, n)``.
    member_ids
        Member id of each column (:data:`MEAN_MODEL_ID` for the mean model).
    required
        Every member must succeed (e.g. the mean model); a failure aborts the run.
        Otherwise failed members are reported and the run's failure policy applies.
    with_sensitivities
        Also return the member-mean sensitivity ``dg/dm``; the simulator must
        provide it (``SensitivitySimulator``).
    """

    name: str
    iteration: int
    state: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    required: bool = False
    with_sensitivities: bool = False


@dataclass(frozen=True)
class Forecast:
    """The forecast of one :class:`ForecastRequest`.

    Attributes
    ----------
    ok
        Which requested members succeeded, ``(n,)``.
    predicted
        The survivors' predictions in the analysis data space, ``(n_data, n_ok)``.
    raw
        Their untransformed predictions, when the run saves them; else ``None``.
    sensitivity
        With ``with_sensitivities``: the survivors' mean sensitivity of the predictions to
        the state, ``(n_data, n_state)`` (analysis data rows); else ``None``.
    """

    ok: NDArray[np.bool_]
    predicted: NDArray[np.float64]
    raw: NDArray[np.float64] | None = None
    sensitivity: NDArray[np.float64] | None = None


#: The forecasts of a step's requests, by request name.
Forecasts = Mapping[str, Forecast]


@dataclass(frozen=True)
class Record:
    """A forecast ensemble the workflow scores and saves.

    ``kind = "iterate"`` is saved as ``iter_<iteration>.npz`` (a kept ensemble: every step
    of a fixed schedule, every accepted iterate of an iterative method); ``"trial"`` (a
    rejected trial) as ``trial_<iteration>.npz`` when the run keeps trials. Both are
    scored (data mismatch against ``scoring``, the metrics) into
    ``data_mismatch_<iteration>.npz`` and
    the history.

    Attributes
    ----------
    iteration, kind
        As above.
    state, predicted, member_ids
        The ensemble ``(n_state, n)``, its forecast ``(n_data, n)``, its member ids.
    scoring
        Perturbed observations its data mismatch is scored against, ``(n_data, n)``.
    accepted
        Whether it was kept (false for a rejected trial).
    damping
        Damping that produced it (iterative methods).
    raw
        Its untransformed forecast, if the run saves raw forecasts.
    extras
        Further arrays saved with it, e.g. ``perturbed_observations`` (ES-MDA) or
        ``reference_prediction`` (GIES).
    """

    iteration: int
    kind: Literal["iterate", "trial"]
    state: NDArray[np.float64]
    predicted: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    scoring: NDArray[np.float64]
    accepted: bool = True
    damping: float | None = None
    raw: NDArray[np.float64] | None = None
    extras: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    """What an algorithm returns from ``start``, ``advance`` and ``restore_state``.

    Attributes
    ----------
    requests
        Ensembles to forecast next; empty when the algorithm has finished.
    iteration
        Iteration the requests belong to (``max_steps`` counts completed iterations).
    records
        Forecasts to score and save now.
    artifacts
        Result files to save once, ``{name: {array_name: array}}`` (e.g. the perturbed
        observations an iterative method draws once).
    checkpoint
        Whether the run may be checkpointed (and stopped) after this step. Steps inside
        an iteration (e.g. an extra mean-model forecast) are not checkpoints: a crash
        there redoes the iteration's pending forecasts on resume.
    converged
        For iterative methods, whether the convergence test stopped the run.
    """

    requests: Sequence[ForecastRequest]
    iteration: int
    records: Sequence[Record] = ()
    artifacts: Mapping[str, Mapping[str, NDArray[Any]]] = field(default_factory=dict)
    checkpoint: bool = True
    converged: bool | None = None


@dataclass(frozen=True)
class AlgorithmState:
    """An algorithm's complete loop state, for checkpoints.

    ``arrays`` are saved in an ``.npz`` (C-contiguous, as the loop holds them), ``values``
    in ``state.json`` (numbers, strings, booleans, lists and dicts of them).
    """

    arrays: Mapping[str, NDArray[Any]]
    values: Mapping[str, Any]


@dataclass(frozen=True)
class Estimate:
    """The algorithm's current estimate: its kept ensemble and that ensemble's forecast."""

    state: NDArray[np.float64]
    predicted: NDArray[np.float64]
    member_ids: NDArray[np.int64]


class EnsembleAlgorithm(Protocol):
    """An ensemble history-matching algorithm driven by the generic workflow.

    It keeps its loop state itself; :meth:`export_state` must capture all of it, so that
    ``restore_state(export_state())`` continues exactly (bit-identically) where it left
    off. Arrays it holds must be C-contiguous (see ``workflows.history_matching._keep``).

    Optionally, it defines ``progress() -> IterationProgress`` (``core.progress``): its
    position in its iteration budget, shown by progress displays.
    """

    status: Status
    #: Whether the number of iterations is decided as the run goes (GIES, and other
    #: iterative smoothers), so cross validation may stop it; ``False`` for a
    #: fixed schedule such as ES-MDA's.
    iterative: bool

    def start(self, setup: RunSetup) -> Step:
        """Begin a run from ``setup.prior``: usually request its forecast."""
        ...

    def advance(self, forecasts: Forecasts) -> Step:
        """Take the forecasts of the last step's requests; return the next step."""
        ...

    def export_state(self) -> AlgorithmState:
        """Everything needed to continue, at a step with ``checkpoint = True``."""
        ...

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Continue from :meth:`export_state`'s output; return the pending step."""
        ...

    def estimate(self) -> Estimate:
        """Return the current (last kept) ensemble, its forecast and member ids."""
        ...


def keep(array: NDArray[Any], ok: NDArray[np.bool_] | None = None) -> NDArray[Any]:
    """Return the members ``ok`` of an ensemble array (all without ``ok``), C-contiguous.

    Every ensemble array an algorithm or the workflow holds goes through here, so that
    uninterrupted and resumed runs compute on the same memory layout: ``array[:, ok]``
    returns Fortran order, and ``np.savez`` keeps it. The values are the same, but on
    AVX-512 CPUs NumPy and OpenBLAS sum the two layouts in different orders, which made a
    resumed run drift from an uninterrupted one by rounding (2e-16, seen on CI).
    """
    selected: NDArray[Any] = np.ascontiguousarray(array if ok is None else array[:, ok])
    return selected


def mean_model_request(name: str, state: NDArray[np.float64], iteration: int) -> ForecastRequest:
    """Request the forecast of the ensemble-mean model of ``state`` (must succeed)."""
    return ForecastRequest(
        name=name,
        iteration=iteration,
        state=keep(state.mean(axis=-1, keepdims=True)),
        member_ids=np.array([MEAN_MODEL_ID]),
        required=True,
    )
