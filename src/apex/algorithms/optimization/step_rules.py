"""Step rules: how an ensemble optimizer turns a gradient into a step.

Ported from PET
---------------
PET module : src/popt/optimization_methods/subroutines/optimizers.py (GradientDescent, Adam,
             AdaMax, Steihaug)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Kingma, D. P. & Ba, J. (2014). Adam: a method for stochastic optimization.
             arXiv:1412.6980 (Adam, AdaMax). Nocedal, J. & Wright, S. J. (2006). Numerical
             Optimization (2nd ed.), algorithm 7.2 (CG-Steihaug). Springer.
Differences: CG-Steihaug's negative-curvature exit goes to the trust-region boundary from
             the iterate ``p``; PET computes the boundary step from the residual ``r``
             (a PET bug, docs/migration/pet_parity.md). Otherwise none. The step rules
             keep PET's backtracking contract: ``apply_backtracking``
             shrinks the step for the next trial, ``restore_parameters`` commits the
             accepted trial's moments (or velocity) and restores the step size.

Every rule minimises: ``x_new = x - step`` for a gradient ``g`` of the objective.
"""

from collections.abc import Mapping
from typing import Protocol, cast

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray, OptimizerState, StateValue
from apex.algorithms.optimization.registries import step_rules


class StepRule(Protocol):
    """A step rule with backtracking (see the module docstring)."""

    def apply_update(
        self, x: FloatArray, gradient: FloatArray, *, iteration: int, hessian: FloatArray | None
    ) -> tuple[FloatArray, FloatArray]:
        """Return the trial point and the step that reached it (both ``(n_x,)``)."""
        ...

    def apply_backtracking(self, shrink: float = 0.5) -> None:
        """Shrink the step of the next trial by ``shrink``."""
        ...

    def restore_parameters(self) -> None:
        """Commit the last trial's moments and restore the base step size."""
        ...

    def current_step_size(self) -> float:
        """Return the step size (or trust radius) the next trial uses."""
        ...

    def halve_base_step_size(self) -> None:
        """Halve the base step size (PET's rule for a single control; no-op for Steihaug)."""
        ...

    def export_state(self) -> OptimizerState:
        """Return the rule's state for a checkpoint."""
        ...

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Restore a state returned by :meth:`export_state`."""
        ...


def _array(value: StateValue) -> FloatArray | float:
    # A velocity or moment starts as the scalar 0 and becomes an array after a step.
    return (
        np.asarray(value, dtype=float)
        if isinstance(value, np.ndarray)
        else float(cast(float, value))
    )


@step_rules.register("gd")
class GradientDescent:
    r"""Gradient descent with momentum: ``v = beta v_old - alpha g``, ``x_new = x + v``.

    Parameters
    ----------
    step_size
        ``alpha``.
    momentum
        ``beta``; backtracking shrinks it with the step size.
    """

    def __init__(self, step_size: float = 0.1, momentum: float = 0.0) -> None:
        self.step_size = step_size
        self.momentum = momentum
        self.velocity: FloatArray | float = 0.0
        self.temp_velocity: FloatArray | float = 0.0
        self._step_size = step_size
        self._momentum = momentum

    def apply_update(
        self,
        x: FloatArray,
        gradient: FloatArray,
        *,
        iteration: int,
        hessian: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``x + v`` and ``v`` with ``v = beta v_old - alpha g``."""
        self.temp_velocity = self._momentum * self.velocity - self._step_size * gradient
        step = np.asarray(self.temp_velocity, dtype=float)
        return x + step, step

    def apply_smc_update(self, x: FloatArray, target: FloatArray) -> FloatArray:
        """SmcOpt's step: ``(1 - alpha) x + alpha target`` (``target``: the weighted mean)."""
        return np.asarray((1 - self._step_size) * x + self._step_size * target)

    def apply_backtracking(self, shrink: float = 0.5) -> None:
        """Shrink the step size and the momentum for the next trial."""
        self._step_size = shrink * self._step_size
        self._momentum = shrink * self._momentum

    def restore_parameters(self) -> None:
        """Keep the last trial's velocity; restore the step size and momentum."""
        self.velocity = self.temp_velocity
        self._step_size = self.step_size
        self._momentum = self.momentum

    def current_step_size(self) -> float:
        """Return the step size of the next trial."""
        return self._step_size

    def halve_base_step_size(self) -> None:
        """Halve the base step size."""
        self.step_size /= 2

    def export_state(self) -> OptimizerState:
        """Return the rule's state."""
        return {
            "step_size": self.step_size,
            "momentum": self.momentum,
            "velocity": self.velocity,
            "temp_velocity": self.temp_velocity,
            "current_step_size": self._step_size,
            "current_momentum": self._momentum,
        }

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Restore a state returned by :meth:`export_state`."""
        self.step_size = float(cast(float, state["step_size"]))
        self.momentum = float(cast(float, state["momentum"]))
        self.velocity = _array(state["velocity"])
        self.temp_velocity = _array(state["temp_velocity"])
        self._step_size = float(cast(float, state["current_step_size"]))
        self._momentum = float(cast(float, state["current_momentum"]))


@step_rules.register("adam")
class Adam:
    r"""Adam: bias-corrected first and second moments of the gradient.

    ``m = b1 m_old + (1 - b1) g``, ``v = b2 v_old + (1 - b2) g^2``,
    ``x_new = x - alpha (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)``, ``eps = 1e-7``.

    Parameters
    ----------
    step_size
        ``alpha``.
    momentum
        ``b1``. PET's EnOpt and GenOpt pass their momentum ``beta`` here (default 0).
    beta2
        ``b2``.
    """

    eps = 1e-7

    def __init__(self, step_size: float = 0.1, momentum: float = 0.9, beta2: float = 0.999) -> None:
        self.step_size = step_size
        self.beta1 = momentum
        self.beta2 = beta2
        self.vel1: FloatArray | float = 0.0
        self.vel2: FloatArray | float = 0.0
        self.temp_vel1: FloatArray | float = 0.0
        self.temp_vel2: FloatArray | float = 0.0
        self._step_size = step_size

    def apply_update(
        self,
        x: FloatArray,
        gradient: FloatArray,
        *,
        iteration: int,
        hessian: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``x - step`` and the Adam ``step`` at iteration ``iteration`` (from 1)."""
        self.temp_vel1 = self.beta1 * self.vel1 + (1 - self.beta1) * gradient
        self.temp_vel2 = self.beta2 * self.vel2 + (1 - self.beta2) * gradient**2
        vel1_hat = self.temp_vel1 / (1 - self.beta1**iteration)
        vel2_hat = self.temp_vel2 / (1 - self.beta2**iteration)
        step = np.asarray(self._step_size * vel1_hat / (np.sqrt(vel2_hat) + self.eps), dtype=float)
        return x - step, step

    def apply_backtracking(self, shrink: float = 0.5) -> None:
        """Shrink the step size for the next trial."""
        self._step_size = shrink * self._step_size

    def restore_parameters(self) -> None:
        """Keep the last trial's moments; restore the step size."""
        self.vel1 = self.temp_vel1
        self.vel2 = self.temp_vel2
        self._step_size = self.step_size

    def current_step_size(self) -> float:
        """Return the step size of the next trial."""
        return self._step_size

    def halve_base_step_size(self) -> None:
        """Halve the base step size."""
        self.step_size /= 2

    def export_state(self) -> OptimizerState:
        """Return the rule's state."""
        return {
            "step_size": self.step_size,
            "vel1": self.vel1,
            "vel2": self.vel2,
            "temp_vel1": self.temp_vel1,
            "temp_vel2": self.temp_vel2,
            "current_step_size": self._step_size,
        }

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Restore a state returned by :meth:`export_state`."""
        self.step_size = float(cast(float, state["step_size"]))
        self.vel1 = _array(state["vel1"])
        self.vel2 = _array(state["vel2"])
        self.temp_vel1 = _array(state["temp_vel1"])
        self.temp_vel2 = _array(state["temp_vel2"])
        self._step_size = float(cast(float, state["current_step_size"]))


@step_rules.register("adamax")
class AdaMax(Adam):
    r"""AdaMax: Adam with the infinity norm for the second moment.

    ``u = max(b2 u_old, |g|)``, ``x_new = x - alpha / (1 - b1^t) m / u``.
    """

    def apply_update(
        self,
        x: FloatArray,
        gradient: FloatArray,
        *,
        iteration: int,
        hessian: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``x - step`` and the AdaMax ``step``."""
        self.temp_vel1 = self.beta1 * self.vel1 + (1 - self.beta1) * gradient
        self.temp_vel2 = np.maximum(self.beta2 * self.vel2, np.abs(gradient))
        step = np.asarray(
            self._step_size / (1 - self.beta1**iteration) * self.temp_vel1 / self.temp_vel2,
            dtype=float,
        )
        return x - step, step


@step_rules.register("steihaug")
class Steihaug:
    """CG-Steihaug: the truncated conjugate-gradient trust-region step.

    Minimises ``g^T p + p^T B p / 2`` over ``||p|| <= delta`` by conjugate gradients, stopping
    at negative curvature or at the boundary (Nocedal & Wright, algorithm 7.2). Backtracking
    shrinks the trust radius ``delta``.

    Parameters
    ----------
    step_size
        Initial trust radius ``delta0``. PET's EnOpt builds it with 3.0, whatever its
        ``alpha``.
    momentum
        Unused (the common factory signature of the step rules).
    maxiter, epsilon, delta_max
        CG iterations, relative residual tolerance, largest radius.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        momentum: float = 0.0,
        maxiter: float = 1e6,
        epsilon: float = 1e-8,
        delta_max: float = 1e5,
    ) -> None:
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.delta_max = delta_max
        self.delta0 = step_size
        self.delta = step_size

    def apply_update(
        self,
        x: FloatArray,
        gradient: FloatArray,
        *,
        iteration: int,
        hessian: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``x + p`` and the CG-Steihaug step ``p``.

        Raises
        ------
        ValueError
            Without a Hessian.
        RuntimeError
            If CG does not converge within ``maxiter`` iterations.
        """
        if hessian is None:
            raise ValueError("the Steihaug step needs a Hessian (EnOpt: hessian = true)")
        p = np.zeros(len(x), np.float64)
        r = gradient * 1.0
        d = -gradient * 1.0
        b = hessian * 1.0
        length_r0 = np.sqrt(np.dot(r, r))
        length_test = self.epsilon * length_r0
        if length_r0 < self.epsilon:
            return x, p
        j = 0
        while True:
            curvature = np.dot(d, np.dot(b, d))
            if curvature <= 0.0:
                p_new = self._negative_curvature_step(p, r, d)
                return x + p_new, p_new
            a = np.dot(r, r) / curvature
            p_new = p + a * d
            if np.sqrt(np.dot(p_new, p_new)) >= self.delta:
                p_new = p + self._tau(p, d) * d
                return x + p_new, p_new
            r_new = r + a * np.dot(b, d)
            if np.sqrt(np.dot(r_new, r_new)) < length_test:
                return x + p_new, p_new
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            d = -r_new + beta * d
            p, r = p_new * 1.0, r_new * 1.0
            if j > self.maxiter:
                raise RuntimeError(f"Steihaug CG did not converge within {self.maxiter} iterations")
            j += 1

    def _negative_curvature_step(self, p: FloatArray, r: FloatArray, d: FloatArray) -> FloatArray:
        """Return the step to the boundary along ``d`` from the iterate ``p``.

        PET passes the residual ``r``, not ``p``, to its ``get_tau`` here, which puts the step
        off the boundary (docs/migration/pet_parity.md).
        """
        return np.asarray(p + self._tau(p, d) * d)

    def _tau(self, p: FloatArray, d: FloatArray) -> float:
        # The positive root of ||p + tau d||^2 = delta^2.
        pd = np.dot(p, d)
        dd = np.dot(d, d)
        return float((-pd + np.sqrt(pd**2 - dd * (np.dot(p, p) - self.delta**2))) / dd)

    def apply_backtracking(self, shrink: float = 0.5) -> None:
        """Shrink the trust radius for the next trial."""
        self.delta = shrink * self.delta

    def restore_parameters(self) -> None:
        """Restore the initial trust radius."""
        self.delta = self.delta0

    def current_step_size(self) -> float:
        """Return the trust radius of the next trial."""
        return self.delta

    def halve_base_step_size(self) -> None:
        """Do nothing: PET halves only rules with a ``step_size``."""

    def export_state(self) -> OptimizerState:
        """Return the rule's state."""
        return {"delta0": self.delta0, "delta": self.delta}

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Restore a state returned by :meth:`export_state`."""
        self.delta0 = float(cast(float, state["delta0"]))
        self.delta = float(cast(float, state["delta"]))
