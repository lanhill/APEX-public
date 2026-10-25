"""Levenberg-Marquardt ensemble randomized maximum likelihood (LM-EnRML).

Ported from PET
---------------
PET module : src/pipt/update_schemes/enrml.py (IterativeEnRML.update_step and
             score_and_commit, LMEnRML), src/pipt/update_schemes/analysis/approx.py
             (approx_update), src/pipt/update_schemes/core/scheme_base.py (score,
             propose_state)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y. & Oliver, D. S. (2013). Levenberg-Marquardt forms of the iterative
             ensemble smoother for efficient history matching and uncertainty
             quantification. Computational Geosciences, 17(4), 689-703.
Differences: none in the step or the damping rules with ``converge_on_rejection = True``
             (the parity tests pin both); by default (since 2026-10-04) a
             worse trial within the tolerance is rejected, not converged, and (since
             2026-10-19, ``confirm_convergence``) an improving one within it stops the run
             only at ``minimum`` or after a second such trial in a row (with no rejection
             between); such a trial relaxes ``lambda`` whatever the spread did, and also at
             ``maximum``, where PET stops. One PET
             bug is not reproduced: after a rejected trial, PET's retry updates the last
             accepted state with the *rejected* trial's forecast (``calc_analysis`` reads
             ``pred_data``, which the rejected forecast overwrote); APEX retries from the
             accepted iterate and its own forecast, as Chen & Oliver's algorithm does, so
             runs diverge after the first rejection (docs/migration/pet_parity.md). The loop
             is GIES's: dropping members whose trial fails, and bit-identical
             resume. Localization, PET's ``localanalysis``, ``emp_cov`` and adjoint
             branches are not ported (the ``full``, ``subspace``, ``subspace2`` and
             ``margis`` analyses are separate strategies).

The approximate update (``approx`` analysis) is, per member ``j`` and with the fixed
perturbed observations ``d_j`` drawn once from ``C_d``,

    ``m_j <- m_j + C_md ((1 + lambda) C_d + C_dd)^-1 (d_j - g(m_j))``,

in APEX's :class:`~apex.algorithms.data_assimilation.update_strategies.ApproxUpdate` at
``damping = lambda``: with whitened anomalies ``L^-1 dY = U S V^T`` (truncated SVD) the
step is ``dX V S ((1 + lambda) I + S^2)^-1 U^T L^-1 (D - Y)``. This form omits the prior
increment term of the full Gauss-Newton RML step (Chen & Oliver's approximation): the
prior acts through the ensemble covariance in the gain, not as an explicit mismatch term.

Each trial is accepted if its mean data mismatch (against the perturbed observations,
PET's ``calc_objectivefun``) falls; ``lambda`` is then divided by ``factor`` if the data mismatch
spread fell too, and not below ``minimum``. A rejected trial multiplies ``lambda`` by
``factor`` and is retried from the same iterate. The run stops when the relative data mismatch
change falls below ``data_mismatch_tolerance`` on an improving trial at ``minimum``, or on
two improving trials in a row (``confirm_convergence``; PET stops on the first: a heavily
damped step is small without being near a minimum), on any trial with
``converge_on_rejection`` (PET's rule; the trial is then kept only if it improved), when
``lambda`` reaches ``maximum``, after ``max_stalled_rejections`` rejected trials within the
tolerance in a row, after ``max_inner_iterations`` rejected trials in a row, or after
``max_iterations`` accepted iterations.
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from numpy.typing import ArrayLike

from apex.algorithms.data_assimilation.algorithm import Status
from apex.algorithms.data_assimilation.gies import GIES
from apex.algorithms.data_assimilation.protocols import (
    DampingDecision,
    DampingScheme,
    UpdateStrategy,
)
from apex.algorithms.data_assimilation.registries import damping_schemes, methods
from apex.algorithms.data_assimilation.update_strategies import ApproxUpdate

if TYPE_CHECKING:
    from apex.algorithms.data_assimilation.gies import _GiesRun

logger = logging.getLogger(__name__)


@damping_schemes.register("lm")
@dataclass(frozen=True, slots=True)
class LMDampingSchedule:
    """LM-EnRML's damping scheme, ``damping_schemes["lm"]`` (PET ``[dataassim.iteration]``).

    The default of LM-EnRML; GIES may use it too. ``lambda`` is relaxed only
    when the data mismatch spread falls as well.

    Parameters
    ----------
    initial
        The damping ``lambda`` of the first analysis (positive, default 100), or ``"auto"``
        for half the prior's mean data mismatch per datum.
    factor
        Divide ``lambda`` by this after an accepted trial whose data mismatch spread also
        fell, and multiply by it after a rejected one (greater than 1, default 5).
    minimum
        ``lambda`` is not reduced once it is at or below this (default 0.01).
    maximum
        Stop once a trial's ``lambda`` reaches this (above ``minimum``, default 1e10).
    data_mismatch_tolerance
        The run has converged when an improving trial changes the mean data mismatch by a
        relative amount below this (positive, default 0.01).
    converge_on_rejection
        Also stop on a worse trial within the tolerance (default false: it is rejected and
        ``lambda`` raised).
    max_stalled_rejections
        Stop after this many consecutive rejected trials within the tolerance (at least 1,
        default 3).
    confirm_convergence
        A small improvement stops the run only when ``lambda`` is at its ``minimum`` or the
        previous accepted trial was small too (default true; otherwise the trial is
        accepted and ``lambda`` relaxed). False stops at the first small improvement.

    Notes
    -----
    The names in PET (``[dataassim.iteration]``) are ``lambda``, ``lambda_factor``,
    ``lambda_min``, ``lambda_max`` and ``data_misfit_tol``. ``"auto"`` is ``0.5 *
    mean(Phi_prior) / n_data``. ``converge_on_rejection`` is PET's ``score_and_commit``
    rule; by default a worse trial is rejected, as in GIES's ``rlmmac``.
    ``confirm_convergence`` is true because a heavily damped step can be small
    without the data mismatch being near a minimum (the subspace update's step is ``(1 +
    lambda)^-1`` of a Gauss-Newton step, 1 % of it at the default ``lambda`` of 100); false
    is PET's rule. GIES's ``rlmmac`` scheme has the option too, false by default (its
    docstring says why).
    """

    initial: float | Literal["auto"] = 100.0
    factor: float = 5.0
    minimum: float = 0.01
    maximum: float = 1e10
    data_mismatch_tolerance: float = 0.01
    converge_on_rejection: bool = False
    max_stalled_rejections: int = 3
    confirm_convergence: bool = True

    def __post_init__(self) -> None:
        """Validate the parameters."""
        if self.max_stalled_rejections < 1:
            raise ValueError(
                f"max_stalled_rejections must be at least 1, got {self.max_stalled_rejections}"
            )
        if self.initial != "auto" and not (
            isinstance(self.initial, int | float)
            and math.isfinite(self.initial)
            and self.initial > 0
        ):
            raise ValueError(f"initial lambda must be positive or 'auto', got {self.initial!r}")
        if not math.isfinite(self.factor) or self.factor <= 1:
            raise ValueError(f"lambda factor must be greater than 1, got {self.factor}")
        if not 0 <= self.minimum < self.maximum:
            raise ValueError(f"need 0 <= minimum < maximum, got {self.minimum}, {self.maximum}")
        if not self.data_mismatch_tolerance > 0:
            raise ValueError(
                f"data_mismatch_tolerance must be positive, got {self.data_mismatch_tolerance}"
            )

    def decide(
        self,
        damping: float,
        previous_data_mismatch: float,
        data_mismatch: float,
        *,
        previous_spread: float,
        spread: float,
    ) -> DampingDecision:
        """Judge a trial as PET's ``IterativeEnRML.score_and_commit`` with LM's controls.

        ``damping`` is the ``lambda`` the trial ran with; the data mismatches are member means and
        the spreads their standard deviations.

        Raises
        ------
        ValueError
            If a data mismatch is not finite or ``previous_data_mismatch`` is not positive.
        """
        if not (math.isfinite(previous_data_mismatch) and math.isfinite(data_mismatch)):
            raise ValueError(
                f"data mismatches must be finite, got {previous_data_mismatch} and {data_mismatch}"
            )
        if previous_data_mismatch <= 0:
            raise ValueError(
                f"previous_data_mismatch must be positive, got {previous_data_mismatch}"
            )
        improved = data_mismatch < previous_data_mismatch
        small = abs(1.0 - data_mismatch / previous_data_mismatch) < self.data_mismatch_tolerance
        if small and improved and self.confirm_convergence and damping > self.minimum:
            # Small, but perhaps only because lambda damped it: accept, relax lambda, and let
            # the loop judge the next, less damped trial (two such in a row stop the run).
            return DampingDecision(accepted=True, converged=False,
                                   damping=max(damping / self.factor, self.minimum),
                                   tentative=True)  # fmt: skip
        # Converged (an improving trial within the tolerance; a worse one only with
        # converge_on_rejection, PET's rule), or lambda at its maximum: stop.
        if (small and (improved or self.converge_on_rejection)) or damping >= self.maximum:
            return DampingDecision(accepted=improved, converged=True, damping=damping)
        if improved:
            # PET relaxes lambda only if the data mismatch spread fell as well (_on_improved).
            relax = spread < previous_spread and damping > self.minimum
            return DampingDecision(
                accepted=True, converged=False, damping=damping / self.factor if relax else damping
            )
        return DampingDecision(
            accepted=False, converged=False, damping=damping * self.factor, stalled=small
        )


@methods.register("lm_enrml")
class LMEnRML(GIES):
    """LM-EnRML: an iterative smoother with Levenberg-Marquardt damping of its steps.

    Parameters
    ----------
    strategy
        Update strategy; ``ApproxUpdate(energy=0.95)`` by default (PET ``analysis =
        "approx"``, ``iteration.energy``). A ``[method.strategy] name = "approx"`` without
        ``energy`` gets ``ApproxUpdate``'s own default, 0.98. Localization is refused:
        upstream PET's LM-EnRML localizes differently (its ``approx`` ``rank-r``
        projection), and that is not ported.
    schedule
        Damping scheme; :class:`LMDampingSchedule` (``lm``, PET's defaults) by
        default, or any other, e.g. GIES's ``rlmmac``.
    lower_bounds, upper_bounds
        Optional per-state bounds ``(n_state,)`` applied to each trial iterate.
    max_iterations
        Accepted iterations at most (``max_iter``).
    max_inner_iterations
        Rejected trials in a row after which the run stops (``max_inner_iter``, default
        10); with the ``lm`` scheme ``maximum`` usually stops it first (``rlmmac`` has none:
        then the tolerance, the iteration budget or these rejections do).
    """

    status: Status = "parity"
    schedule_type: ClassVar[type] = LMDampingSchedule
    #: The adaptive localization this method takes (agents propose it).
    default_localization: ClassVar[str] = "adaptive_projected"
    accepts_local_analysis: ClassVar[bool] = True

    def __init__(
        self,
        strategy: UpdateStrategy | None = None,
        schedule: DampingScheme | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        max_iterations: int = 10,
        max_inner_iterations: int = 10,
    ) -> None:
        # Local analysis (a selection) and upstream PET's localizations of the approx update
        # (adaptive_projected, distance_entries: they declare enrml_localization) are
        # ported; PET-fork's tapers are not.
        localization = getattr(strategy, "localization", None)
        if (
            localization is not None
            and getattr(strategy, "localization_role", "taper") != "selection"
            and not getattr(localization, "enrml_localization", False)
        ):
            raise ValueError(
                "LM-EnRML and GN-EnRML localize as upstream PET does: use [method.localization] "
                "name = 'adaptive_projected' (in the projected-data space) or "
                "'distance_entries' (or local analysis)"
            )
        super().__init__(
            ApproxUpdate(energy=0.95) if strategy is None else strategy,
            # PET's LM rule by default (``lm``); any damping scheme.
            LMDampingSchedule() if schedule is None else schedule,
            lower_bounds,
            upper_bounds,
            max_iterations,
            # GIES's inner retries count accepted iterations; exhausting them stops the run.
            max_trials_per_iteration=max_inner_iterations,
        )
        # RLM-MAC's damping scales the spectrum (gamma = lambda * mean(s^2)): with no floor
        # lambda may fall towards 0 near convergence, an unregularised step that only a
        # rejection would undo. A scheme without a floor (rlmmac's
        # default) counts as 0.
        if getattr(strategy, "relative_damping", False) and not self.schedule.minimum:
            raise ValueError(
                f"LM-EnRML with the {type(strategy).__name__} analysis needs a positive "
                "damping minimum (lambda_min): at lambda = 0 RLM-MAC is unregularised; "
                "keep the lm scheme's default 0.01 or give another positive value"
            )

    @property
    def lm_schedule(self) -> DampingScheme:
        """The damping scheme (``lm`` by default); the name."""
        return self.schedule

    def decide(
        self, damping: float, previous_data_mismatch: float, data_mismatch: float
    ) -> DampingDecision:
        """Not LM-EnRML's rule: its decisions need the data mismatch spreads (``lm_schedule``).

        Raises
        ------
        TypeError
            Always; use ``lm_schedule.decide``.
        """
        raise TypeError("LM-EnRML decides with lm_schedule.decide, which takes the spreads")

    def _count_trial(self, run: "_GiesRun", *, accepted: bool) -> None:
        assert self.max_trials_per_iteration is not None
        run.trials += 1
        if accepted:
            run.outer, run.trials = run.outer + 1, 0
        elif run.trials >= self.max_trials_per_iteration:
            # PET: no improving step left to try in this iteration, so the run stops.
            run.converged = True
            logger.info(
                "no improving step after %d attempts (%s = %.3g)",
                run.trials,
                self.control_name,
                run.damping,
                extra={
                    "event": "trials_exhausted",
                    "trial": run.iteration,
                    "control": self.control_name,
                    "value": run.damping,
                },
            )
