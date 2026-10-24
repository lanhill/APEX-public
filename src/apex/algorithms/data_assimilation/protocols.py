"""Interfaces for ensemble data-assimilation plugins.

All arrays keep the ensemble axis last: states ``(n_state, n_ensemble)``, data
``(n_data, n_ensemble)``.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance


class UpdateStrategy(Protocol):
    """Computes the state increment of one ensemble analysis.

    The meaning of ``damping`` is specific to each strategy (e.g. absolute ``1 + lambda``
    for ``ApproxUpdate``, relative ``lambda * mean(s^2)`` for ``RLMMACUpdate``).
    """

    #: Whether ``step`` needs ``reference_prediction``, the forecast at the mean state.
    requires_reference_prediction: bool
    #: Whether ``damping`` scales the spectrum (RLM-MAC's ``gamma = lambda * mean(s^2)``),
    #: so that ``damping = 0`` leaves the step unregularised (refused by GN-EnRML).
    relative_damping: bool

    def step(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        *,
        damping: float = 0.0,
        reference_prediction: NDArray[np.float64] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Return the increment to add to ``state``, shape ``(n_state, n_ensemble)``.

        Parameters
        ----------
        state
            Current state ensemble, ``(n_state, n_ensemble)``.
        predicted
            Predicted data for each member, ``(n_data, n_ensemble)``.
        perturbed_observations
            Perturbed observations, ``(n_data, n_ensemble)``.
        data_error
            Observation-error covariance the update uses (already inflated, if the method
            inflates it).
        damping
            Levenberg-Marquardt parameter ``lambda``; 0 for ES/ES-MDA. Each strategy
            documents how it enters the regularisation.
        reference_prediction
            Predicted data at the ensemble-mean state, ``(n_data,)``. Strategies that centre
            data anomalies on it (RLM-MAC) require it; strategies centring on the ensemble
            mean reject it rather than silently ignoring it.
        rng
            Random stream for strategies whose localization draws random numbers (the
            run's checkpointed ``localization`` stream); unused otherwise.
        """
        ...


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Outcome of one assimilation step.

    Attributes
    ----------
    posterior
        Updated state ensemble, ``(n_state, n_ensemble)``.
    perturbed_observations
        Perturbed observations the update used, ``(n_data, n_ensemble)``.
    """

    posterior: NDArray[np.float64]
    perturbed_observations: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DampingDecision:
    """Outcome of comparing a trial iterate's data mismatch with the current one.

    Attributes
    ----------
    accepted
        Whether the trial iterate becomes the current one (its mean data mismatch is lower).
    converged
        Whether the relative data mismatch change fell below the tolerance; the iteration stops.
    damping
        Damping parameter for the next analysis.
    stalled
        A rejected trial whose relative change was within the tolerance: the update barely
        moved. GIES stops after ``max_stalled_rejections`` of them in a row.
    tentative
        An improving trial within the tolerance taken with the damping above its minimum:
        accepted, but not yet convergence, since the damping may be what kept it small. The
        loop stops after two of them in a row (``confirm_convergence``).
    """

    accepted: bool
    converged: bool
    damping: float
    stalled: bool = False
    tentative: bool = False


class DampingScheme(Protocol):
    """How an iterative method adapts its damping ``lambda`` between trials.

    A scheme judges each trial and chooses the next ``lambda``; how ``lambda`` enters the
    step stays the update strategy's (RLM-MAC's relative ``lambda * mean(s^2)``, the approx
    update's absolute ``1 + lambda``). Registered in ``damping_schemes``: ``rlmmac``
    (GIES's rule, :class:`~apex.algorithms.data_assimilation.gies.DampingSchedule`) and
    ``lm`` (PET's LM-EnRML rule,
    :class:`~apex.algorithms.data_assimilation.lm_enrml.LMDampingSchedule`).
    """

    @property
    def initial(self) -> float | Literal["auto"]:
        """``lambda`` of the first analysis (``"auto"``: from the prior's data mismatch)."""
        ...

    @property
    def factor(self) -> float:
        """The factor ``lambda`` grows by after a rejected trial."""
        ...

    @property
    def data_mismatch_tolerance(self) -> float:
        """Stop when the relative change of the mean data mismatch drops below this."""
        ...

    @property
    def minimum(self) -> float | None:
        """Do not reduce ``lambda`` at or below this; ``None``: no floor."""
        ...

    def decide(
        self,
        damping: float,
        previous_data_mismatch: float,
        data_mismatch: float,
        *,
        previous_spread: float,
        spread: float,
    ) -> DampingDecision:
        """Judge a trial: accept or reject it, and give the next ``lambda``.

        ``damping`` is the ``lambda`` the trial ran with; the data mismatches are member
        means, the spreads their standard deviations (a scheme may ignore them).
        """
        ...


class HyperparameterHook(Protocol):
    """What an iterative method tells a hyper-parameter estimator (CHOP).

    GIES (and the methods on its loop: LM-EnRML, GN-EnRML) calls it, when set as its
    ``hyperparameter_hook``; without one they are unchanged. Shapes are ensemble last.
    """

    def before_update(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        reference: NDArray[np.float64] | None,
    ) -> None:
        """Take the accepted ``state`` and forecast, before each trial is formed."""
        ...

    def after_accept(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        reference: NDArray[np.float64] | None,
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        damping: float,
    ) -> None:
        """Take an accepted iterate, once its forecasts are in.

        ``reference`` is the forecast of its ensemble mean, ``damping`` the ``lambda`` the
        accepted trial ran with.
        """
        ...

    def drop_members(self, ok: NDArray[np.bool_]) -> None:
        """Drop the members that failed (``ok`` marks the survivors)."""
        ...
