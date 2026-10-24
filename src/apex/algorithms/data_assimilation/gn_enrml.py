"""Gauss-Newton ensemble randomized maximum likelihood (GN-EnRML).

Ported from PET
---------------
PET module : src/pipt/update_schemes/enrml.py (IterativeEnRML.update_step and
             score_and_commit, GNEnRML, gn_enrml), src/pipt/update_schemes/core/
             scheme_base.py (propose_state: ``step_scale``)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y. & Oliver, D. S. (2013). Levenberg-Marquardt forms of the iterative
             ensemble smoother for efficient history matching and uncertainty
             quantification. Computational Geosciences, 17(4), 689-703. Raanes, P. N.,
             Stordal, A. S. & Evensen, G. (2019). Revising the stochastic iterative
             ensemble smoother. Nonlinear Processes in Geophysics, 26(3), 325-338.
             Evensen, G., Raanes, P. N., Stordal, A. S. & Hove, J. (2019). Efficient
             implementation of an iterative ensemble smoother for data assimilation and
             reservoir history matching. Frontiers in Applied Mathematics and Statistics,
             5, 47.
Differences: none in the step-length rules with ``converge_on_rejection = True`` (the
             parity test replays them); by default (since 2026-10-04) a
             worse trial within the tolerance is rejected, not converged. As for
             LM-EnRML, a retry after a rejected trial starts from the accepted iterate and
             its own forecast (PET reuses the rejected trial's forecast; see
             docs/migration/pet_parity.md), and the loop is GIES's (member drops,
             bit-identical resume). PET's ``localanalysis``, ``emp_cov`` and adjoints are
             not ported.

GN-EnRML solves the same RML problem as LM-EnRML but takes *undamped* Gauss-Newton steps
(the strategy runs at ``damping = 0``) scaled by a step length ``gamma``:

    ``m_j <- m_j + gamma * step_j``  (``approx``/``full``: the state step at lambda = 0),
    ``W <- W + gamma * dW``          (``subspace``: the step in the ensemble weights).

Trials are accepted on the mean data mismatch (against the fixed perturbed observations),
as in LM-EnRML. After an accepted trial whose data mismatch spread also fell, ``gamma`` relaxes
towards ``maximum``:

    ``gamma <- gamma + (maximum - gamma) * 2^(-(k + 1) / (factor - 1))``,

with ``k`` the number of accepted iterations before this one; after a rejected trial
``gamma <- gamma / factor`` and the trial is retried from the same iterate. With
``factor <= 1`` (allowed by PET) ``gamma`` never changes. The run stops when the relative
data mismatch change falls below ``data_mismatch_tolerance`` on an improving trial (on any
trial with ``converge_on_rejection``, PET's rule; the trial is kept only if it improved),
after ``max_stalled_rejections`` rejected trials within the tolerance in a row (at once with
``factor <= 1``, whose retry would be the same trial), after ``max_inner_iterations``
rejected trials in a row (there is no ``gamma`` floor), or after ``max_iterations``
accepted iterations.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from numpy.typing import ArrayLike

from apex.algorithms.data_assimilation.algorithm import Status
from apex.algorithms.data_assimilation.lm_enrml import LMEnRML
from apex.algorithms.data_assimilation.protocols import (
    DampingDecision,
    DampingScheme,
    UpdateStrategy,
)
from apex.algorithms.data_assimilation.registries import methods
from apex.algorithms.data_assimilation.update_strategies import ApproxUpdate

if TYPE_CHECKING:
    from apex.algorithms.data_assimilation.gies import _GiesRun

#: PET's ``gamma = 'auto'``: a fixed step length (it needs nothing from the prior).
AUTO_GAMMA = 0.1


@dataclass(frozen=True, slots=True)
class GNStepSchedule:
    """GN-EnRML's step-length control (PET ``[dataassim.iteration]`` keywords).

    Attributes
    ----------
    initial
        Step length of the first trial (``gamma``, default 0.2); ``"auto"`` is 0.1.
    maximum
        Value the step length relaxes towards after an improving trial (``gamma_max``,
        0.5).
    factor
        Divide the step length by this after a rejected trial; it also sets how fast it
        relaxes (``gamma_factor``, 2.5). At most 1, the step length is fixed.
    data_mismatch_tolerance
        Stop when ``|1 - data_mismatch / previous_data_mismatch|`` drops below this
        (``data_misfit_tol``,
        0.01).
    converge_on_rejection
        Also stop on a *worse* trial within the tolerance (PET's ``score_and_commit``;
        default False: it is rejected, as GIES's ``rlmmac``).
    max_stalled_rejections
        Stop after this many consecutive rejected trials within the tolerance (default 3).
    """

    initial: float | Literal["auto"] = 0.2
    maximum: float = 0.5
    factor: float = 2.5
    data_mismatch_tolerance: float = 0.01
    converge_on_rejection: bool = False
    max_stalled_rejections: int = 3

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
            raise ValueError(f"initial gamma must be positive or 'auto', got {self.initial!r}")
        if not (math.isfinite(self.maximum) and self.maximum > 0):
            raise ValueError(f"gamma maximum must be positive, got {self.maximum}")
        if not (math.isfinite(self.factor) and self.factor > 0):
            raise ValueError(f"gamma factor must be positive, got {self.factor}")
        if not self.data_mismatch_tolerance > 0:
            raise ValueError(
                f"data_mismatch_tolerance must be positive, got {self.data_mismatch_tolerance}"
            )

    @property
    def initial_gamma(self) -> float:
        """The first step length, with ``"auto"`` resolved."""
        return AUTO_GAMMA if self.initial == "auto" else float(self.initial)

    def decide(
        self,
        gamma: float,
        previous_data_mismatch: float,
        data_mismatch: float,
        *,
        previous_spread: float,
        spread: float,
        accepted_iterations: int,
    ) -> DampingDecision:
        """Judge a trial as PET's ``IterativeEnRML.score_and_commit`` with GN's controls.

        ``gamma`` is the step length the trial ran with, the data mismatches are member means and
        the spreads their standard deviations; ``accepted_iterations`` counts the accepted
        iterations before this trial (PET's ``iteration``). The decision's ``damping`` is
        the next step length.

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
        if small and (improved or self.converge_on_rejection):
            return DampingDecision(accepted=improved, converged=True, damping=gamma)
        if improved:
            if spread < previous_spread and self.factor > 1:
                # PET's _on_improved: relax towards the maximum, less with every iteration.
                gamma = gamma + (self.maximum - gamma) * 2.0 ** (
                    -(accepted_iterations + 1) / (self.factor - 1)
                )
            return DampingDecision(accepted=True, converged=False, damping=gamma)
        if self.factor > 1:
            gamma = gamma / self.factor
        elif small:
            # A fixed step length: the retry would repeat this very trial; stop now.
            return DampingDecision(accepted=False, converged=True, damping=gamma, stalled=True)
        return DampingDecision(accepted=False, converged=False, damping=gamma, stalled=small)


@methods.register("gn_enrml")
class GNEnRML(LMEnRML):
    """GN-EnRML: an iterative smoother with Gauss-Newton steps scaled by a step length.

    Parameters
    ----------
    strategy
        Update strategy, run at ``damping = 0``; ``ApproxUpdate(energy=0.95)`` by default
        (PET ``analysis = "approx"``). ``full``, ``subspace``, ``subspace2`` and
        ``margis`` are the other PET analyses ported (PET's legacy ``gn_enrml`` scheme is
        this with ``subspace``).
        Localization is ``adaptive_projected`` (or local analysis), as for LM-EnRML;
        strategies with relative
        damping (``rlmmac``, ``local_rlmmac``): at damping 0 they are unregularised.
    schedule
        Step-length control; :class:`GNStepSchedule` (PET's defaults) by default.
    lower_bounds, upper_bounds
        Optional per-state bounds ``(n_state,)`` applied to each trial iterate.
    max_iterations
        Accepted iterations at most (``max_iter``).
    max_inner_iterations
        Rejected trials in a row after which the run stops (``max_inner_iter``, 10).
    """

    status: Status = "parity"
    schedule_type: ClassVar[type] = GNStepSchedule
    control_name: ClassVar[str] = "step length"
    #: GN-EnRML runs a step length, not a damping: no damping scheme.
    accepts_damping: ClassVar[bool] = False

    def __init__(
        self,
        strategy: UpdateStrategy | None = None,
        schedule: GNStepSchedule | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        max_iterations: int = 10,
        max_inner_iterations: int = 10,
    ) -> None:
        # GN-EnRML runs its strategy at damping 0. RLM-MAC's damping is relative to the
        # spectrum, so that is an unregularised truncated-SVD step; on the tanh linear case
        # it stalls at 1.7-2.4x the data mismatch the other pairs reach
        # (tests/workflows/test_method_strategy_pairs.py).
        if getattr(strategy, "relative_damping", False):
            raise ValueError(
                f"GN-EnRML with the {type(strategy).__name__} analysis is not supported: it "
                "runs at damping 0, which leaves RLM-MAC unregularised (gamma = lambda * "
                "mean(s^2) = 0); use approx, full, subspace, subspace2 or margis, or RLM-MAC "
                "with gies or lm_enrml"
            )
        super().__init__(
            ApproxUpdate(energy=0.95) if strategy is None else strategy,
            None,
            lower_bounds,
            upper_bounds,
            max_iterations,
            max_inner_iterations,
        )
        self.step_schedule = GNStepSchedule() if schedule is None else schedule

    # ``schedule`` holds LM-EnRML's default scheme, which GN-EnRML never reads: its trials
    # are judged by step_schedule (``_judge``, ``_initial_damping`` and ``decide`` below).

    @property
    def lm_schedule(self) -> DampingScheme:
        """None: GN-EnRML runs a step length, not a damping scheme."""
        raise AttributeError("GN-EnRML has no damping scheme; its control is step_schedule")

    def decide(
        self, damping: float, previous_data_mismatch: float, data_mismatch: float
    ) -> DampingDecision:
        """Not GN-EnRML's rule: its decisions need the spreads (``step_schedule``).

        Raises
        ------
        TypeError
            Always; use ``step_schedule.decide``.
        """
        raise TypeError("GN-EnRML decides with step_schedule.decide, which takes the spreads")

    def _stall_limit(self) -> int | None:
        return self.step_schedule.max_stalled_rejections

    @property
    def initial_damping(self) -> float:
        """The first step length ``gamma`` (the loop's control parameter)."""
        return self.step_schedule.initial_gamma

    def _initial_damping(self, run: "_GiesRun") -> float:
        return self.step_schedule.initial_gamma

    def _analysis_damping(self, run: "_GiesRun") -> float:
        # Gauss-Newton: undamped steps (PET sets lam = 0 for the analyses).
        return 0.0

    def _step_length(self, run: "_GiesRun") -> float:
        return run.damping

    def _judge(self, run: "_GiesRun") -> DampingDecision:
        return self.step_schedule.decide(
            run.damping,
            run.data_mismatch,
            run.trial_data_mismatch,
            previous_spread=self._spread(run, run.predicted),
            spread=self._spread(run, run.trial_predicted),
            # LM's _count_trial ends the run when an iteration exhausts its trials, so the
            # completed outer iterations are the accepted ones (PET's ``iteration``).
            accepted_iterations=run.outer,
        )
