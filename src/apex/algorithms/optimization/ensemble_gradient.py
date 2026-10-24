"""Gaussian control perturbations: the ensemble gradient, Hessian and SMC weights.

Ported from PET
---------------
PET module : src/popt/ensembles/ensemble_gaussian.py (GaussianEnsemble.gradient, .hessian,
             .calc_ensemble_weights), src/popt/ensembles/ensemble_base.py (_aux_input)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y., Oliver, D. S. & Zhang, D. (2009). Efficient ensemble-based closed-loop
             production optimization. SPE Journal, 14(4), 634-645 (EnOpt).
             Zhang, Y., Stordal, A. S. & Lorentzen, R. J. (2023). A natural Hessian
             approximation for ensemble based optimization. Computational Geosciences, 27,
             355-364. Stordal, A. S., Szklarz, S. P. & Leeuwenburgh, O. (2016). A theoretical
             look at ensemble-based optimization in reservoir management. Mathematical
             Geosciences, 48, 399-417 (the natural gradient).
Differences: the baseline ``f`` subtracted from the members' objectives is the caller's
             ``f(x_k)``; PET subtracts the last single-point evaluation, which after a
             rejected trial is that trial's value. This matters for the Hessian always, and
             for the gradient when members are clipped or realizations differ
             (docs/migration/pet_parity.md). Under Nesterov the ensemble is centred at the
             extrapolated point while the baseline stays ``f(x_k)``, as in PET.
             Multilevel ensembles are not ported.

Notation: ``x`` ``(n_x,)`` the controls, ``C`` ``(n_x, n_x)`` the perturbation covariance,
``X`` ``(n_x, N)`` the perturbed controls, ``F`` ``(N,)`` their objectives and ``f`` the
baseline. With ``dX = X - x`` and ``dF = F - f``:

- natural gradient ``g = dX dF / N`` (an estimate of ``C grad f``);
- natural Hessian ``H = (dX * dF) dX^T / N - C mean(dF)`` (an estimate of ``C (hess f) C``);
- without ``natural``, both are preconditioned by ``C^-1`` (on each side for ``H``).
"""

from collections.abc import Callable, Mapping
from typing import cast

import numpy as np

from apex.algorithms.optimization.protocols import (
    Draws,
    FloatArray,
    IntArray,
    OptimizerState,
    StateValue,
)

#: Evaluates the objective of a set of physical controls ``(n_x, n)`` on the given model
#: realizations ``(n,)`` and returns ``(n,)`` values.
type EnsembleEvaluator = Callable[[FloatArray, IntArray], FloatArray]


def member_realizations(n_members: int, n_samples: int, n_realizations: int) -> IntArray:
    """Return the model realization of each member ``(n_members,)``.

    Members cycle through the realizations in blocks of ``n_samples / n_realizations``
    (PET's ``aux_input``).

    Raises
    ------
    ValueError
        If ``n_samples`` is not a multiple of ``n_realizations``.
    """
    if n_samples % n_realizations:
        raise ValueError(
            f"the ensemble size {n_samples} must be a multiple of the number of model "
            f"realizations {n_realizations}"
        )
    block = n_samples // n_realizations
    return np.asarray((np.arange(n_members) // block) % n_realizations, dtype=np.int64)


def gaussian_perturbations(
    x: FloatArray,
    cov: FloatArray,
    n: int,
    draws: Draws,
    lower: FloatArray | None = None,
    upper: FloatArray | None = None,
) -> FloatArray:
    """Draw ``n`` perturbed controls around ``x``, re-centred on ``x`` and clipped.

    Parameters
    ----------
    x
        Controls ``(n_x,)``.
    cov
        Perturbation covariance ``(n_x, n_x)``.
    n
        Ensemble size.
    draws
        Source of the normal draws.
    lower, upper
        Bounds ``(n_x,)`` to clip to, or ``None``.

    Returns
    -------
    FloatArray
        ``(n_x, n)``: the ensemble mean is exactly ``x`` before clipping, so the perturbations
        carry no first-order bias into the gradient.
    """
    _check_cov(x, cov)
    ensemble = draws.multivariate_normal(x, cov, n).T
    ensemble = ensemble - ensemble.mean(axis=1, keepdims=True) + x[:, None]
    return _clip(ensemble, lower, upper)


def ensemble_gradient(
    x: FloatArray,
    perturbed: FloatArray,
    values: FloatArray,
    baseline: FloatArray,
    cov: FloatArray,
    *,
    natural: bool = True,
) -> FloatArray:
    """Return the ensemble gradient ``(n_x,)`` (see the module docstring).

    Parameters
    ----------
    x
        Controls ``(n_x,)``.
    perturbed
        Perturbed controls ``(n_x, N)``.
    values
        Their objectives ``(N,)``.
    baseline
        The objective at ``x`` on each member's realization ``(N,)``.
    cov
        Perturbation covariance ``(n_x, n_x)``.
    natural
        Return the natural gradient ``C grad f``; otherwise ``grad f``.
    """
    d_x, d_f = _anomalies(x, perturbed, values, baseline)
    # (n_x,) for any n_x: PET's np.squeeze makes one control's gradient 0-d, which its own
    # normalisation (norm with ord=inf) cannot take.
    gradient = np.reshape(d_x @ d_f, (x.size,)) / d_f.size
    if not natural:
        gradient = np.linalg.solve(cov, gradient)
    return np.asarray(gradient, dtype=float)


def ensemble_hessian(
    x: FloatArray,
    perturbed: FloatArray,
    values: FloatArray,
    baseline: FloatArray,
    cov: FloatArray,
    *,
    natural: bool = True,
) -> FloatArray:
    """Return the ensemble Hessian ``(n_x, n_x)`` of Zhang et al. (2023).

    Parameters are those of :func:`ensemble_gradient`. The natural Hessian estimates
    ``C (hess f) C``; otherwise ``C^-1`` is applied on each side.
    """
    d_x, d_f = _anomalies(x, perturbed, values, baseline)
    hessian = (d_x * d_f) @ d_x.T / d_f.size - cov * np.mean(d_f)
    if not natural:
        hessian = np.linalg.solve(cov, np.linalg.solve(cov, hessian).T).T
    return np.asarray(hessian, dtype=float)


def smc_weights(values: FloatArray, inflation: float) -> FloatArray:
    """Return SmcOpt's normalised importance weights ``(N,)``.

    ``w_i ∝ exp(min(-(F_i - min F) * inflation, 10)) + 1e-6``: a tempered Boltzmann weight
    (the best member gets ``exp(0)``), floored so that no member's weight vanishes.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.exp(np.clip(-(values - np.min(values)) * inflation, None, 10))
    weights = weights + 1e-6
    return np.asarray(weights / np.sum(weights), dtype=float)


class GaussianEnsemble:
    """The ensemble gradient estimator of EnOpt and the particle system of SmcOpt.

    Parameters
    ----------
    evaluate
        Evaluates a set of physical controls (see :data:`EnsembleEvaluator`).
    n_samples
        Ensemble size ``N`` (PET's ``ne``).
    draws
        Source of the random draws.
    lower, upper
        Physical bounds ``(n_x,)`` the members are clipped to, or ``None``.
    n_realizations
        Model realizations of robust optimization (PET's ``num_models``).
    natural_gradient
        Return the natural gradient and Hessian (PET's default).
    """

    def __init__(
        self,
        evaluate: EnsembleEvaluator,
        n_samples: int,
        draws: Draws,
        lower: FloatArray | None = None,
        upper: FloatArray | None = None,
        n_realizations: int = 1,
        *,
        natural_gradient: bool = True,
    ) -> None:
        if n_samples < 2:
            raise ValueError(f"the ensemble needs at least two members, got {n_samples}")
        member_realizations(n_samples, n_samples, n_realizations)  # validates the blocks
        self.evaluate = evaluate
        self.n_samples = n_samples
        self.draws = draws
        self.lower = lower
        self.upper = upper
        self.n_realizations = n_realizations
        self.natural_gradient = natural_gradient
        # The last ensemble, for the Hessian of the same draw.
        self.x: FloatArray | None = None
        self.cov: FloatArray | None = None
        self.perturbed: FloatArray | None = None
        self.values: FloatArray | None = None
        self.baseline: FloatArray | None = None
        # SmcOpt's particles and the members they resample.
        self.particles: FloatArray | None = None
        self.particle_values: FloatArray | None = None
        self.resample_index: IntArray | None = None

    def realizations(self, n_members: int) -> IntArray:
        """Return the model realization of each of ``n_members`` members."""
        return member_realizations(n_members, self.n_samples, self.n_realizations)

    def gradient(self, x: FloatArray, cov: FloatArray, f_x: FloatArray) -> FloatArray:
        """Draw and evaluate an ensemble around ``x``; return the ensemble gradient ``(n_x,)``.

        Parameters
        ----------
        x
            Physical controls ``(n_x,)``.
        cov
            Perturbation covariance ``(n_x, n_x)``.
        f_x
            The objective at the iterate on each realization ``(n_realizations,)``.
        """
        perturbed = gaussian_perturbations(
            x, cov, self.n_samples, self.draws, self.lower, self.upper
        )
        realizations = self.realizations(self.n_samples)
        values = np.asarray(self.evaluate(perturbed, realizations), dtype=float)
        baseline = np.asarray(f_x, dtype=float).reshape(-1)[realizations]
        self.x, self.cov, self.perturbed, self.values, self.baseline = (
            x,
            cov,
            perturbed,
            values,
            baseline,
        )
        return ensemble_gradient(x, perturbed, values, baseline, cov, natural=self.natural_gradient)

    def hessian(self) -> FloatArray:
        """Return the ensemble Hessian ``(n_x, n_x)`` of the last :meth:`gradient`'s ensemble.

        Raises
        ------
        RuntimeError
            Before any gradient.
        """
        if (
            self.x is None
            or self.cov is None
            or self.perturbed is None
            or self.values is None
            or self.baseline is None
        ):
            raise RuntimeError("the Hessian needs the ensemble of a gradient first")
        return ensemble_hessian(
            self.x,
            self.perturbed,
            self.values,
            self.baseline,
            self.cov,
            natural=self.natural_gradient,
        )

    def smc_sensitivity(
        self, x: FloatArray, inflation: float, cov: FloatArray, survival: float
    ) -> tuple[FloatArray, FloatArray, float]:
        """Draw new particles, weight all of them and resample the survivors (SmcOpt).

        The first call draws ``N`` particles. Later calls keep ``N - round(N survival)``
        particles resampled by the last weights and draw ``round(N survival)`` new ones.

        Parameters
        ----------
        x
            Physical controls ``(n_x,)`` the new particles are drawn around (not re-centred).
        inflation
            Tempering factor of :func:`smc_weights`.
        cov
            Covariance of the new particles ``(n_x, n_x)``.
        survival
            Fraction of particles drawn anew.

        Returns
        -------
        tuple
            The weighted particle mean ``(n_x,)`` (SmcOpt's target), the best particle
            ``(n_x,)`` and its objective.
        """
        _check_cov(x, cov)
        n = self.n_samples
        n_new = int(np.round(n * survival))
        n_drawn = n if self.resample_index is None else n_new
        drawn = _clip(self.draws.multivariate_normal(x, cov, n_drawn).T, self.lower, self.upper)
        values = np.asarray(self.evaluate(drawn, self.realizations(n_drawn)), dtype=float)
        n_survivors = n - n_new
        if self.resample_index is None or self.particles is None or self.particle_values is None:
            self.particles = drawn.copy()
            self.particle_values = values.copy()
        else:
            self.particles[:, :n_survivors] = self.particles[:, self.resample_index]
            self.particles[:, n_survivors:] = drawn
            self.particle_values[:n_survivors] = self.particle_values[self.resample_index]
            self.particle_values[n_survivors:] = values
        weights = smc_weights(self.particle_values, inflation)
        target = self.particles @ weights
        best = int(np.argmin(self.particle_values))
        best_x = self.particles[:, best].copy()
        best_f = float(self.particle_values[best])
        self.resample_index = self.draws.choice(n, n_survivors, weights)
        return np.asarray(target, dtype=float), best_x, best_f

    def export_state(self) -> OptimizerState:
        """Return SmcOpt's particle state (the gradient's ensemble is redrawn every step)."""
        return {
            "particles": self.particles,
            "particle_values": self.particle_values,
            "resample_index": self.resample_index,
        }

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Restore a state returned by :meth:`export_state`."""
        particles = state.get("particles")
        values = state.get("particle_values")
        index = state.get("resample_index")
        self.particles = None if particles is None else np.array(particles, dtype=float)
        self.particle_values = None if values is None else np.array(values, dtype=float)
        self.resample_index = (
            None if index is None else np.array(cast(IntArray, index), dtype=np.int64)
        )


def _anomalies(
    x: FloatArray, perturbed: FloatArray, values: FloatArray, baseline: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if perturbed.ndim != 2 or perturbed.shape[0] != x.size:
        raise ValueError(f"perturbed controls must be ({x.size}, N), got {perturbed.shape}")
    n = perturbed.shape[1]
    if values.shape != (n,) or baseline.shape != (n,):
        raise ValueError(
            f"values and baseline must be ({n},), got {values.shape} and {baseline.shape}"
        )
    return perturbed - x[:, None], values - baseline


def _check_cov(x: FloatArray, cov: FloatArray) -> None:
    if cov.shape != (x.size, x.size):
        raise ValueError(f"the covariance must be ({x.size}, {x.size}), got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("the covariance has non-finite entries")


def _clip(ensemble: FloatArray, lower: FloatArray | None, upper: FloatArray | None) -> FloatArray:
    if lower is None and upper is None:
        return ensemble
    low = -np.inf if lower is None else lower[:, None]
    high = np.inf if upper is None else upper[:, None]
    return np.asarray(np.clip(ensemble, low, high), dtype=float)
