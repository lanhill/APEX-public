"""Interfaces of ensemble optimization: objectives, random draws and the problem.

Shapes follow a set of control vectors is ``(n_x, n)``, with the ensemble (or the
set of points) last, and an objective returns one value per column, ``(n,)``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from apex.domain.observations import DataKey

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

#: How far a plugin is validated: against PET (``parity``), against an independent reference
#: (``validated``), or not yet (``experimental``). Shown in logs and reports.
Status = Literal["parity", "validated", "experimental"]

#: What an optimizer's state holds: arrays, and JSON values (the checkpoints).
type StateValue = FloatArray | IntArray | float | int | bool | str | None
type OptimizerState = dict[str, StateValue]


@runtime_checkable
class Objective(Protocol):
    """A function to minimise, evaluated column by column.

    The controls are physical (never the unit cube). ``realizations`` says which model
    realization each column is evaluated on; robust optimization averages over them, and an
    objective with one model ignores it.
    """

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the objective of each column.

        Parameters
        ----------
        controls
            Control vectors ``(n_x, n)``.
        realizations
            Model realization of each column ``(n,)``.

        Returns
        -------
        FloatArray
            Objective values ``(n,)``.
        """
        ...


@runtime_checkable
class VectorObjective(Protocol):
    """Several objectives to minimise together, evaluated column by column.

    The multi-objective optimizers (``nsga2``) take one; the others refuse it. A quantity to
    maximise is negated (or scaled by a negative ``obj_scaling``), as for one objective.
    """

    #: The objectives' names, in the order of the rows returned.
    objective_names: tuple[str, ...]

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the objectives of each column ``(n_obj, n)`` (controls ``(n_x, n)``)."""
        ...


@runtime_checkable
class ParetoFront(Protocol):
    """A vector objective whose true Pareto front is known (the analytic test problems)."""

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return ``n_points`` points spread over the true front ``(n_obj, n_points)``."""
        ...


@runtime_checkable
class DifferentiableObjective(Objective, Protocol):
    """An objective with its exact gradient and Hessian (for line search and trust region)."""

    def gradient(self, x: FloatArray) -> FloatArray:
        """Return the gradient ``(n_x,)`` at ``x`` ``(n_x,)``."""
        ...

    def hessian(self, x: FloatArray) -> FloatArray:
        """Return the Hessian ``(n_x, n_x)`` at ``x`` ``(n_x,)``."""
        ...


@runtime_checkable
class ConstrainedObjective(Objective, Protocol):
    """An objective with constraints, enforced by the external penalty (PET's EPF)."""

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the equality and inequality constraints of each column.

        Parameters
        ----------
        controls
            Control vectors ``(n_x, n)``.

        Returns
        -------
        tuple of FloatArray
            ``c_eq`` ``(n_eq, n)``, zero when satisfied, and ``c_iq`` ``(n_iq, n)``,
            non-negative when satisfied.
        """
        ...


@runtime_checkable
class ResponseObjective(Protocol):
    """An objective of a simulator's responses, such as NPV.

    The workflow writes each column of controls into a simulation (the members' state), asks
    the simulator for :meth:`response_keys`, and passes the responses here. Minimised, as every
    objective: an NPV to maximise is scaled by a negative factor.
    """

    def response_keys(self) -> Sequence[DataKey]:
        """Return the responses the objective needs, in the order :meth:`evaluate` takes."""
        ...

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return the objective of each simulation.

        Parameters
        ----------
        responses
            The simulated responses ``(n_keys, n)`` at :meth:`response_keys`.

        Returns
        -------
        FloatArray
            Objective values ``(n,)``.
        """
        ...


class Draws(Protocol):
    """Where an estimator's random draws come from.

    The default wraps a ``numpy.random.Generator`` (:class:`GeneratorDraws`). The parity tests
    replay the draws PET made, so whole optimizations can be compared with PET's.
    Both methods return NumPy's layout (one draw per row); the estimators transpose at once.
    """

    def multivariate_normal(self, mean: FloatArray, cov: FloatArray, size: int) -> FloatArray:
        """Return ``size`` draws ``(size, n_x)`` from ``N(mean, cov)``."""
        ...

    def choice(self, n: int, size: int, p: FloatArray) -> IntArray:
        """Return ``size`` indices in ``[0, n)`` drawn with replacement with probabilities ``p``."""
        ...

    def export_state(self) -> dict[str, object]:
        """Return the state of the stream (JSON values), for checkpoints."""
        ...

    def restore_state(self, state: Mapping[str, object]) -> None:
        """Restore a state returned by :meth:`export_state`."""
        ...


class GeneratorDraws:
    """:class:`Draws` from a ``numpy.random.Generator``."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def multivariate_normal(self, mean: FloatArray, cov: FloatArray, size: int) -> FloatArray:
        """Return ``size`` draws ``(size, n_x)`` from ``N(mean, cov)``."""
        return np.asarray(self.rng.multivariate_normal(mean, cov, size), dtype=np.float64)

    def choice(self, n: int, size: int, p: FloatArray) -> IntArray:
        """Return ``size`` indices in ``[0, n)`` drawn with replacement with probabilities ``p``."""
        return np.asarray(self.rng.choice(n, size, replace=True, p=p), dtype=np.int64)

    def export_state(self) -> dict[str, object]:
        """Return the bit generator's state."""
        return dict(self.rng.bit_generator.state)

    def restore_state(self, state: Mapping[str, object]) -> None:
        """Restore the bit generator's state."""
        self.rng.bit_generator.state = dict(state)


@dataclass(frozen=True)
class Problem:
    """What an optimizer minimises.

    Parameters
    ----------
    objective
        The function to minimise.
    x0
        Initial controls ``(n_x,)``.
    lower, upper
        Bounds ``(n_x,)``, or ``None`` for none. ``transform`` needs both, finite.
    n_realizations
        Model realizations for robust optimization (PET's ``num_models``): a single point is
        evaluated on each of them, and the ensemble's members cycle through them in blocks.
    names
        Optional name of each control, for reports.
    """

    objective: Objective
    x0: FloatArray
    lower: FloatArray | None = None
    upper: FloatArray | None = None
    n_realizations: int = 1
    names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        """Check ``x0`` and broadcast the bounds to its shape."""
        x0 = np.asarray(self.x0, dtype=np.float64)
        if x0.ndim != 1 or x0.size == 0:
            raise ValueError(f"x0 must be a non-empty vector, got shape {x0.shape}")
        object.__setattr__(self, "x0", x0)
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                array = np.broadcast_to(np.asarray(value, dtype=np.float64), x0.shape).copy()
                object.__setattr__(self, name, array)
        if self.n_realizations < 1:
            raise ValueError(f"n_realizations must be positive, got {self.n_realizations}")
