"""Generalised iterative ensemble smoother (GIES) with the RLM-MAC update.

Ported from PET
---------------
PET module : pipt/update_schemes/gies/gies_base.py (GIESMixIn.calc_analysis,
             check_convergence, _ext_iter_param), pipt/update_schemes/gies/rlmmac_update.py
             (rlmmac_update.update), ensemble/ensemble.py (mean-model forecast),
             pipt/loop/ensemble.py (_ext_obs), pipt/geostat/decomp.py (Cholesky.gen_real),
             pipt/misc_tools/analysis_tools.py (calc_objectivefun)
PET commit : 1348861b08314b8b3f3e1ccf886c7c095273e202 (not public;
             upstream PET's gies/ is dormant)
Reference  : Luo, X., Stordal, A. S., Lorentzen, R. J. & Naevdal, G. (2015). Iterative
             ensemble smoother as an approximate solution to a regularized
             minimum-average-cost problem: theory and applications. SPE Journal, 20(5),
             962-982. doi:10.2118/176023-PA (RLM-MAC). Luo, X. (2021). Novel iterative
             ensemble smoothers derived from a class of generalized cost functions.
             Computational Geosciences, 25(3), 1159-1189. doi:10.1007/s10596-021-10046-1
             (GIES).
Differences: decided with the method's author (2026-09-25):
             - data anomalies are scaled by 1/sqrt(N-1) like the state anomalies (PET-fork
               leaves them unscaled);
             - regularisation gamma = lambda * mean(s_apex^2), as in PET-fork's localised
               branch, instead of the unlocalised branch's raw lambda;
             - truncation uses APEX's truncated_svd (first rank reaching the energy, at least
               1) instead of the old rule (cumulative fraction <= energy, possibly empty).
             ``scale_data_anomalies = false`` restores PET-fork's unscaled dY, for
             reproducing its runs (the user, 2026-10-12, after the Brugge MEC replay).
             Here s_apex are the singular values of L^-1 dY with dY scaled by 1/sqrt(N-1), and
             s_pet = sqrt(N-1) s_apex those of PET's unscaled matrix. With no truncation,
             step_apex = sqrt(N-1) * step_pet exactly for lambda_apex = lambda_pet /
             mean(s_pet^2); the parity test pins this.
             Also differs (from the numerics review, approved by the author 2026-09-25):
             - on convergence, the trial is accepted only if its data mismatch improved, as in
               upstream PET's IterativeEnRML; PET-fork always keeps the last trial, even a
               worse or rejected one;
             - numerically null singular values (s <= eps * max(shape) * s_max) are dropped,
               so tiny damping cannot turn rounding noise into a pseudo-inverse step, and the
               initial damping must be positive.
             APEX addition (not in PET): ``DampingSchedule.reduction``, a separate factor for
             the damping after an accepted step, as ``lambda_reduction_factor`` of the
             author's MATLAB RLM-MAC (MATLAB ``main/Kalman/RLM_average_cost.m``: lambda
             times 0.9 after an acceptance, times ``lambda_increment_factor`` after a
             rejection). 0.9 is the default since 2026-10-13 (the user
             3); ``reduction = 0.5`` with ``factor = 2`` is PET-fork's division by the
             factor, exactly. Also from that code:
             ``GIES.max_trials_per_iteration``, its inner retries of rejected trials
             (unset, every trial is an iteration, as in PET-fork). MATLAB's alpha
             equals gamma here when the same singular values are kept; its alpha floor
             is not ported (see docs/migration/pet_parity.md, GIES notes).
             Since 2026-10-04: a worse trial within the tolerance is
             rejected rather than converged, as MATLAB's accepted-only minimum-reduction
             test (``converge_on_rejection = True`` restores PET-fork's rule), and
             ``max_stalled_rejections`` such rejections in a row stop the run.
             Only the unlocalised, analytic-covariance path is ported. Whitening uses the lower
             Cholesky factor (see ``data_error``). PET's ``step_tol`` is read but unused, and
             ``lambda_max`` only feeds a report flag, so neither is ported.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_assimilation.algorithm import (
    AlgorithmState,
    Estimate,
    ForecastRequest,
    Forecasts,
    Record,
    RunSetup,
    Status,
    Step,
    keep,
    mean_model_request,
)
from apex.algorithms.data_assimilation.bounds import StateBounds
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.localization import Localization, refuse_projected
from apex.algorithms.data_assimilation.protocols import (
    DampingDecision,
    DampingScheme,
    HyperparameterHook,
    UpdateStrategy,
)
from apex.algorithms.data_assimilation.registries import (
    damping_schemes,
    methods,
    update_strategies,
)
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import Diagnostics, EmitsDiagnostics
from apex.core.progress import IterationProgress
from apex.domain.observations import DataKey

#: The warning of ``RLMMACUpdate(scale_data_anomalies=False)`` (the user, 2026-10-13).
UNSCALED_DATA_ANOMALIES_WARNING = (
    "rlmmac with scale_data_anomalies = false: the data anomalies are not scaled by "
    "1/sqrt(N - 1) like the state anomalies, so the update does not follow the RLM-MAC "
    "theory (Luo et al. 2015); use it only to reproduce PET-fork's runs"
)


@update_strategies.register("rlmmac")
class RLMMACUpdate(EmitsDiagnostics):
    """Regularised Levenberg-Marquardt update for the minimum-average-cost problem (RLM-MAC).

    From Luo, Stordal, Lorentzen & Naevdal (2015), SPE Journal 20(5), 962-982,
    doi:10.2118/176023-PA.

    Data anomalies are centred on the forecast at the ensemble-mean state ``g(x_mean)``:

    ``dX = (X - mean(X)) / sqrt(N - 1)``, ``dY = (Y - g(x_mean)) / sqrt(N - 1)``,
    ``L^-1 dY = U S V^T`` (truncated),
    ``step = dX V diag(s / (s^2 + gamma)) U^T L^-1 (D - Y)``, ``gamma = lambda * mean(s^2)``,

    which without truncation equals ``dX dY^T (dY dY^T + gamma C_d)^-1 (D - Y)``.

    ``lambda`` is relative to the spectrum (``gamma = lambda * mean(s^2)``), unlike
    :class:`ApproxUpdate`'s absolute ``1 + lambda``, so damping values are not
    interchangeable between the two strategies.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum of ``L^-1 dY`` to retain, in ``(0, 1]``
        (PET keyword ``energy``, default 0.95 for GIES).
    scale_data_anomalies
        Scale ``dY`` by ``1/sqrt(N - 1)`` like ``dX`` (default True, as agreed with the
        method's author). False leaves ``dY`` unscaled, as PET-fork does, to reproduce
        its runs: the singular values grow by ``sqrt(N - 1)`` and ``gamma = lambda *
        mean(s^2)`` with them, so every direction's weight, hence the step, is
        ``1/sqrt(N - 1)`` times the default's at the same ``lambda``. No ``lambda`` maps one
        onto the other (the ratio depends on ``s^2 / gamma``), hence the option. False
        logs :data:`UNSCALED_DATA_ANOMALIES_WARNING`: that update does not follow the
        theory.
    """

    #: This strategy needs the forecast of the ensemble-mean state (an extra simulation).
    requires_reference_prediction = True
    #: ``lambda`` scales the spectrum (``gamma = lambda * mean(s^2)``): at 0 there is no
    #: regularisation at all, unlike ``ApproxUpdate``'s ``1 + lambda``.
    relative_damping = True
    #: Its localization gets the shared gain factor and may taper it per member (CHOP's
    #: per-member tapers).
    per_member_taper = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "singular_values": "retained singular values s of L^-1 dY (dY about g(x_mean)), (r,)",
        "gain": (
            "Kalman-gain-like dX G acting on whitened data L^-1 d, (n_state, n_data); "
            "without localization's taper (with one, the step is not gain @ L^-1 d); "
            "dense, formed only when requested"
        ),
    }

    def __init__(
        self,
        energy: float = 0.95,
        localization: Localization | None = None,
        scale_data_anomalies: bool = True,
    ) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        refuse_projected(localization, "rlmmac")
        self.energy = energy
        self.localization = localization
        self.scale_data_anomalies = scale_data_anomalies
        if not scale_data_anomalies:
            logger.warning(
                UNSCALED_DATA_ANOMALIES_WARNING, extra={"event": "rlmmac_unscaled_anomalies"}
            )

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
        """Return the state increment, shape ``(n_state, n_ensemble)``.

        ``damping`` is ``lambda``; ``reference_prediction`` ``(n_data,)`` is ``g(x_mean)``
        and is required. See
        :class:`~apex.algorithms.data_assimilation.protocols.UpdateStrategy`.

        Raises
        ------
        ValueError
            If ``reference_prediction`` is missing or malformed, the shapes are
            inconsistent, ``damping`` is negative, or every forecast equals
            ``g(x_mean)`` (no sensitivity to invert).
        """
        check_shapes(state, predicted, perturbed_observations, data_error)
        if reference_prediction is None:
            raise ValueError("RLMMACUpdate needs reference_prediction, the forecast of mean(state)")
        if reference_prediction.shape != (predicted.shape[0],):
            raise ValueError(
                f"reference_prediction must have shape ({predicted.shape[0]},), "
                f"got {reference_prediction.shape}"
            )
        if not np.all(np.isfinite(reference_prediction)):
            raise ValueError("reference_prediction contains non-finite values")
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")

        n_ensemble = state.shape[-1]
        state_anomalies = anomalies(state)  # dX, (n_state, n_ensemble)
        # dY centred on g(x_mean) rather than mean(Y): the RLM-MAC sensitivity estimate.
        data_anomalies = predicted - reference_prediction[:, None]
        if self.scale_data_anomalies:
            data_anomalies = data_anomalies / math.sqrt(n_ensemble - 1)
        whitened_anomalies = data_error.whiten(data_anomalies)  # L^-1 dY, (n_data, n_ensemble)
        whitened_innovations = data_error.whiten(perturbed_observations - predicted)

        u, s, vt = truncated_svd(whitened_anomalies, self.energy)  # (n_data, r), (r,), (r, n_ens)
        if s[0] == 0.0:
            raise ValueError(
                "data anomalies are all zero: every forecast equals the mean-model forecast"
            )
        # Drop numerically null directions (numpy's matrix_rank tolerance): with small
        # damping, s / (s^2 + gamma) ~ 1/s would amplify rounding noise into the step.
        keep = s > np.finfo(np.float64).eps * max(whitened_anomalies.shape) * s[0]
        u, s, vt = u[:, keep], s[keep], vt[keep, :]
        gamma = damping * float(np.mean(s**2))  # relative to the kept spectrum
        self._diagnostics.add("strategy", "singular_values", s)
        if self._diagnostics.wants("strategy", "gain"):
            self._diagnostics.add(
                "strategy",
                "gain",
                lambda: state_anomalies @ ((vt.T * (s / (s**2 + gamma))) @ u.T),
                nbytes=state.shape[0] * u.shape[0] * 8,
            )
        if self.localization is not None:
            # Gain factor X = V diag(s / (s^2 + gamma)) U^T acting on whitened data.
            gain_factor = (vt.T * (s / (s**2 + gamma))) @ u.T  # (n_ens, n_data)
            return self.localization.localized_step(
                state_anomalies, gain_factor, whitened_innovations, rng
            )
        # diag(s / (s^2 + gamma)) U^T L^-1 (D - Y), as a row scaling.
        projected = (u.T @ whitened_innovations) * (s / (s**2 + gamma))[:, None]  # (r, n_ens)
        step: NDArray[np.float64] = state_anomalies @ (vt.T @ projected)
        return step


@damping_schemes.register("rlmmac")
@dataclass(frozen=True, slots=True)
class DampingSchedule:
    """GIES's (RLM-MAC's) damping scheme, ``damping_schemes["rlmmac"]`` (PET ``iteration``).

    RLM-MAC: Luo, Stordal, Lorentzen & Naevdal (2015), SPE Journal 20(5), 962-982,
    doi:10.2118/176023-PA.

    The default of GIES; LM-EnRML may use it too. It ignores the data mismatch
    spreads.

    Attributes
    ----------
    initial
        Damping for the first analysis (``lambda``, default 1); must be positive, since the
        schedule only ever multiplies and divides it.
    factor
        Multiply the damping by this after a rejected step (``lambda_factor``, default
        2); with ``reduction = None``, also divide by it after an accepted one.
    reduction
        Multiply the damping by this after an accepted step, in ``(0, 1]``: the MATLAB
        RLM-MAC's ``lambda_reduction_factor`` (``RLM_average_cost.m``, with ``factor`` as
        its ``lambda_increment_factor``), default 0.9 since 2026-10-13 (the user).
        ``None`` divides by ``factor``, PET-fork's rule and the default
        before; runs started before then resume with it. ``0.5`` with ``factor = 2`` is
        the same rule (a power of two, so bit for bit), the way a configuration file
        gives it.
    minimum
        Do not reduce the damping once it is at or below this (``lambda_min``; unset by
        default).
    data_mismatch_tolerance
        Stop when ``|1 - data_mismatch / previous_data_mismatch|`` drops below this
        (``data_misfit_tol``, default 0.01) on an improving trial.
    converge_on_rejection
        Also stop on a *worse* trial within the tolerance (PET-fork's
        ``check_convergence``; default False). A worse trial is a rejection: stopping there
        ended runs after one rejected trial that barely moved (a starved localized update,
        5Spots_snesim 2026-10-04), whereas the MATLAB RLM-MAC tests the reduction on
        accepted steps only and raises the damping.
    max_stalled_rejections
        Stop after this many *consecutive* rejected trials within the tolerance (default
        3): the damping only shortens a step that barely moves, so more trials would spend
        forecasts without changing the iterate (the numerics review of 2026-10-04).
    confirm_convergence
        An improving trial within the tolerance stops the run only when the damping is at
        its ``minimum`` or the previous accepted trial was within the tolerance too, with no
        rejection between (default False). Otherwise it is accepted and the damping reduced
        as after any accepted trial, and the loop judges the next, less damped trial.

        Why False here and True in the ``lm`` scheme (the user, 2026-10-20): the ``lm``
        scheme starts at PET's ``lambda = 100`` on an absolute scale, where a step can be
        ``1/(1 + lambda)``, about 1 %, of a Gauss-Newton step, so its first trial improves
        by less than the 1 % tolerance however far the minimum is (release-tester finding
        34: LM-EnRML stopped at its prior). Here the damping starts at 1 and an accepted
        trial lowers it. With RLM-MAC's relative damping (``gamma = lambda * mean(s^2)``)
        the leading directions, which carry most of the data mismatch reduction, then
        take a large fraction of a Gauss-Newton step; with an absolute strategy
        (``approx``, ``1 + lambda + s^2``) lambda = 1 still leaves at least about half of
        it. So a small improvement mostly does mean a flat objective; the risk is left to
        runs whose damping rejections have raised. Keeping False also keeps GIES
        bit for bit with PET-fork and the MATLAB RLM-MAC, and its existing runs and
        studies unchanged; set True for a run that seems to stop too early.
    """

    initial: float = 1.0
    factor: float = 2.0
    minimum: float | None = None
    data_mismatch_tolerance: float = 0.01
    reduction: float | None = 0.9
    converge_on_rejection: bool = False
    max_stalled_rejections: int = 3
    confirm_convergence: bool = False

    def __post_init__(self) -> None:
        """Validate the parameters."""
        if not isinstance(self.initial, int | float) or isinstance(self.initial, bool):
            raise ValueError(
                f"initial damping must be a number, got {self.initial!r}; an 'auto' start is "
                "the lm scheme's (damping = {name = 'lm', options = {initial = 'auto'}})"
            )
        if not math.isfinite(self.initial) or self.initial <= 0:
            raise ValueError(f"initial damping must be finite and positive, got {self.initial}")
        if not math.isfinite(self.factor) or self.factor <= 1:
            raise ValueError(f"damping factor must be greater than 1, got {self.factor}")
        if self.reduction is not None and not (
            math.isfinite(self.reduction) and 0 < self.reduction <= 1
        ):
            raise ValueError(f"damping reduction must be in (0, 1], got {self.reduction}")
        if self.minimum is not None and self.minimum < 0:
            raise ValueError(f"minimum damping must be non-negative, got {self.minimum}")
        if not self.data_mismatch_tolerance > 0:
            raise ValueError(
                f"data_mismatch_tolerance must be positive, got {self.data_mismatch_tolerance}"
            )
        if self.max_stalled_rejections < 1:
            raise ValueError(
                f"max_stalled_rejections must be at least 1, got {self.max_stalled_rejections}"
            )

    def decide(
        self,
        damping: float,
        previous_data_mismatch: float,
        data_mismatch: float,
        *,
        previous_spread: float | None = None,
        spread: float | None = None,
    ) -> DampingDecision:
        """Compare a trial iterate's mean data mismatch with the current one's.

        Follows PET-fork's ``check_convergence`` damping rule, except that by default only
        an improving trial can converge (``converge_on_rejection``): a relative change below
        the tolerance stops the iteration; otherwise an improvement is accepted and the damping
        reduced (multiplied by ``reduction``, or divided by ``factor`` without one; not below
        ``minimum``), and a deterioration is rejected and the damping multiplied by
        ``factor``. On convergence the trial is accepted only if it improved, as in upstream
        PET; PET-fork keeps it regardless (see the module docstring).

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
        at_floor = self.minimum is not None and damping <= self.minimum
        # PET-fork divides by the rejection factor; the MATLAB RLM-MAC multiplies by
        # its own reduction factor (e.g. 0.9), which lowers the damping more gently.
        reduced = damping / self.factor if self.reduction is None else damping * self.reduction
        if small and improved and self.confirm_convergence and not at_floor:
            # Small, but perhaps only because the damping shortened it: accept, reduce the
            # damping, and let the loop judge the next trial (two such in a row stop the run).
            return DampingDecision(accepted=True, converged=False, damping=reduced, tentative=True)
        if small and (improved or self.converge_on_rejection):
            return DampingDecision(accepted=improved, converged=True, damping=damping)
        if improved:
            return DampingDecision(
                accepted=True,
                converged=False,
                damping=damping if at_floor else reduced,
            )
        return DampingDecision(
            accepted=False, converged=False, damping=damping * self.factor, stalled=small
        )


@methods.register("gies")
class GIES(EmitsDiagnostics):
    """GIES iterative smoother: fixed perturbed observations, damped strategy updates.

    Parameters
    ----------
    strategy
        Update strategy; ``RLMMACUpdate()`` by default (PET ``daalg = ["gies", "rlmmac"]``).
    schedule
        Damping control; ``DampingSchedule()`` by default (PET-fork's, but for ``reduction``).
    lower_bounds, upper_bounds
        Optional per-state bounds ``(n_state,)`` applied to each trial iterate.
    max_iterations
        Iteration budget of a run. Without ``max_trials_per_iteration`` it
        counts trial iterates, accepted or not (PET-fork); with it, outer iterations.
    max_trials_per_iteration
        Retry a rejected trial within the same iteration, from the same accepted iterate
        with the increased damping, up to this many trials. An accepted trial ends the
        iteration. When every trial is rejected, the damping is multiplied by ``factor``
        once more, raised to at least a numeric ``schedule.initial`` (no floor for an
        ``"auto"`` start, which only the ``lm`` scheme has), and the iteration ends
        without an update. These are the inner retries of the author's MATLAB RLM-MAC
        (MATLAB ``RLM_average_cost.m``: ``maxInnerIter = 5`` gives at most 4 trials).
        Unset (the default), every trial is an iteration, as in PET-fork. Every trial
        keeps its own number in the results (``iter_<k>``/``trial_<k>``) either way. The
        convergence test is the schedule's in both modes: a rejected trial whose data mismatch
        changed by less than ``data_mismatch_tolerance`` stops the run only with
        ``converge_on_rejection`` (PET-fork); by default it is rejected and retried with
        more damping, as the MATLAB RLM-MAC, which tests the minimum reduction on accepted
        steps only. An exhausted
        iteration adds no record; its extra damping shows in the next trial's record.
    """

    #: PET-fork parity of the analysis step (with the agreed differences above).
    status: Status = "parity"
    #: The ``[method.schedule]`` table's type (the composition builds it).
    schedule_type: ClassVar[type] = DampingSchedule
    #: Iterates until converged: cross validation may stop it earlier.
    iterative: bool = True
    #: What ``run.damping`` and the records' ``damping`` hold (GN-EnRML: the step length).
    control_name: ClassVar[str] = "damping"
    #: Whether the local-analysis strategy may run in this loop (LM-/GN-EnRML: yes).
    accepts_local_analysis: ClassVar[bool] = False
    #: Whether ``[method.damping]`` may choose its damping scheme (GN-EnRML: no).
    accepts_damping: ClassVar[bool] = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "step": "state increment of the trial, before state bounds are applied, (n_state, N)"
    }

    def __init__(
        self,
        strategy: UpdateStrategy | None = None,
        schedule: DampingScheme | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        max_iterations: int = 10,
        max_trials_per_iteration: int | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
        if max_trials_per_iteration is not None and (
            isinstance(max_trials_per_iteration, bool)
            or not isinstance(max_trials_per_iteration, int)
            or max_trials_per_iteration < 1
        ):
            raise ValueError(
                "max_trials_per_iteration must be a positive integer, "
                f"got {max_trials_per_iteration!r}"
            )
        if getattr(strategy, "requires_levels", False):
            raise ValueError("the hybrid update needs fidelity levels: use multilevel_esmda")
        if self.accepts_damping and not getattr(strategy, "supports_damping", True):
            # ERT's smoothers take no damping: refused here, not at the first
            # damped trial.
            raise ValueError(
                f"{type(self).__name__} damps its updates, which the {type(strategy).__name__} "
                "analysis does not support: use it with es, esmda or gn_enrml"
            )
        if (
            getattr(strategy, "localization_role", None) == "selection"
            and not self.accepts_local_analysis
            and not getattr(strategy, "requires_reference_prediction", False)
        ):
            raise ValueError(
                "GIES runs local analysis with RLM-MAC: use strategy local_rlmmac (the approx "
                "local_analysis and adaptive_rows run with es, esmda, lm_enrml and gn_enrml; "
                "adaptive_rows has no RLM-MAC variant)"
            )
        self.strategy: UpdateStrategy = RLMMACUpdate() if strategy is None else strategy
        # GIES's (RLM-MAC's) rule by default (``rlmmac``); any damping scheme.
        self.schedule: DampingScheme = DampingSchedule() if schedule is None else schedule
        self.bounds = StateBounds(lower_bounds, upper_bounds)
        self.max_iterations = max_iterations
        self.max_trials_per_iteration = max_trials_per_iteration
        #: A hyper-parameter estimator told of every update, acceptance and dropped member
        #: (CHOP); ``None``: none.
        self.hyperparameter_hook: HyperparameterHook | None = None
        self._run: _GiesRun | None = None

    @property
    def initial_damping(self) -> float:
        """Damping for the first analysis: ``nan`` for ``"auto"`` (resolved from the prior)."""
        initial = self.schedule.initial
        return math.nan if initial == "auto" else float(initial)

    @property
    def requires_reference_prediction(self) -> bool:
        """Whether each update needs the forecast of the ensemble-mean state."""
        return bool(getattr(self.strategy, "requires_reference_prediction", False))

    @property
    def uses_sensitivities(self) -> bool:
        """Whether the strategy takes the simulator's sensitivities."""
        return bool(getattr(self.strategy, "uses_sensitivities", False))

    @property
    def requires_prior_state(self) -> bool:
        """Whether each update needs every member's prior state (e.g. the full update)."""
        return bool(getattr(self.strategy, "requires_prior_state", False))

    def perturb_observations(
        self,
        observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        rng: np.random.Generator,
        n_ensemble: int,
    ) -> NDArray[np.float64]:
        """Draw ``d_j = d_obs + L z_j`` once, ``(n_data, n_ensemble)``; they stay fixed."""
        if observations.shape != (data_error.n_data,):
            raise ValueError(
                f"observations must have shape ({data_error.n_data},), got {observations.shape}"
            )
        return observations[:, None] + data_error.perturbations(rng, n_ensemble)

    def update(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        *,
        damping: float,
        reference_prediction: NDArray[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        prior_state: NDArray[np.float64] | None = None,
        step_length: float = 1.0,
        sensitivity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return the bounded trial iterate from the current ``state`` and its forecast.

        ``reference_prediction`` is the forecast of ``mean(state)``, which RLM-MAC needs;
        ``prior_state`` each member's prior, which the full and subspace updates need.
        ``step_length`` scales the step (GN-EnRML's ``gamma``; 1 otherwise). A strategy
        that declares ``applies_step_length`` (the subspace update, whose step is in the
        ensemble weights) takes it itself; any other strategy's step is multiplied by it.
        """
        extra: dict[str, Any] = {"prior_state": prior_state} if self.requires_prior_state else {}
        applies = bool(getattr(self.strategy, "applies_step_length", False))
        if applies:
            extra["step_length"] = step_length
        if self.uses_sensitivities:
            extra["sensitivity"] = sensitivity
        step = self.strategy.step(
            state,
            predicted,
            perturbed_observations,
            data_error,
            damping=damping,
            reference_prediction=reference_prediction,
            rng=rng,
            **extra,
        )
        if step_length != 1.0 and not applies:
            step = step_length * step
        self._diagnostics.add("method", "step", step)
        return self.bounds.apply(state + step)

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Set up the strategy's localization, and a prior-based strategy, from the prior."""
        localization = self._localization()
        if localization is not None:
            localization.prepare(prior_state, prior_predicted, keys, rng)
        if self.requires_prior_state:
            self.strategy.prepare(prior_state)  # type: ignore[attr-defined]

    def export_prepared(self) -> dict[str, NDArray[np.float64]]:
        """Return the localization's and the strategy's prepared arrays (may be empty)."""
        localization = self._localization()
        arrays = {} if localization is None else dict(localization.export_state())
        if self.requires_prior_state:
            prepared = self.strategy.export_state()  # type: ignore[attr-defined]
            arrays.update({f"strategy:{k}": v for k, v in prepared.items()})
        return arrays

    def restore_prepared(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore the localization's and the strategy's prepared arrays on resume."""
        own = {k[len("strategy:") :]: v for k, v in arrays.items() if k.startswith("strategy:")}
        localization = self._localization()
        if localization is not None:
            localization.restore_state(
                {k: v for k, v in arrays.items() if not k.startswith("strategy:")}
            )
        if self.requires_prior_state:
            self.strategy.restore_state(own)  # type: ignore[attr-defined]

    def attach_diagnostics(self, sink: Diagnostics) -> None:
        """Use ``sink`` here and in the strategy and localization this method owns."""
        self._diagnostics = sink
        for part in (self.strategy, self._localization()):
            attach = getattr(part, "attach_diagnostics", None)
            if attach is not None:
                attach(sink)

    def _localization(self) -> Localization | None:
        localization: Localization | None = getattr(self.strategy, "localization", None)
        return localization

    def mean_data_mismatch(
        self,
        predicted: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
    ) -> float:
        """Mean over members of ``(d_j - y_j)^T C_d^-1 (d_j - y_j)`` (PET ``calc_objectivefun``)."""
        if predicted.shape != perturbed_observations.shape:
            raise ValueError(
                f"predicted shape {predicted.shape} must equal perturbed_observations shape "
                f"{perturbed_observations.shape}"
            )
        return float(np.mean(data_error.data_mismatch(perturbed_observations - predicted)))

    def _checked_data_mismatch(self, run: "_GiesRun", stored: float) -> float:
        """Return the checkpoint's accepted data mismatch, after recomputing it from its forecast.

        The next accept/reject decision compares a trial with the stored value, so it is
        what is used. A value that no longer matches the saved forecast and perturbed
        observations (a changed data mismatch formula, corrupted arrays) is refused. Differences
        within :data:`RESUME_DATA_MISMATCH_RTOL` are rounding (another CPU, another NumPy or
        OpenBLAS) and are only logged.

        Raises
        ------
        RuntimeError
            If the recomputed data mismatch differs from ``stored`` by more than
            ``RESUME_DATA_MISMATCH_RTOL`` relative.
        """
        recomputed = self.mean_data_mismatch(run.predicted, run.perturbed, run.setup.data_error)
        if recomputed == stored:
            return stored
        relative = abs(recomputed - stored) / abs(stored) if stored else math.inf
        if relative > RESUME_DATA_MISMATCH_RTOL:
            raise RuntimeError(
                f"the checkpoint's forecast, perturbed observations and accepted data mismatch are "
                "inconsistent: the data mismatch recomputed on resume "
                f"({recomputed!r}) differs from "
                f"the stored one ({stored!r}) by {relative:.3g} relative (the data mismatch "
                f"arithmetic or the saved arrays changed); rerun, or resume with the software "
                f"versions that wrote the checkpoint"
            )
        logger.warning(
            "accepted data mismatch recomputed on resume differs by rounding; the stored value "
            "is used",
            extra={
                "event": "data_mismatch_recomputed_differs",
                "stored": stored,
                "recomputed": recomputed,
                "relative": relative,
            },
        )
        return stored

    def decide(
        self, damping: float, previous_data_mismatch: float, data_mismatch: float
    ) -> DampingDecision:
        """Accept or reject a trial iterate and choose the next damping (see the schedule).

        Only for the ``rlmmac`` scheme, which ignores the spreads; the loop passes them.

        Raises
        ------
        TypeError
            For another scheme (``lm`` needs the spreads): use ``schedule.decide``.
        """
        if not isinstance(self.schedule, DampingSchedule):
            raise TypeError(
                f"{type(self.schedule).__name__} decides with the data mismatch spreads: use "
                "schedule.decide(..., previous_spread=..., spread=...)"
            )
        return self.schedule.decide(damping, previous_data_mismatch, data_mismatch)

    # --------------------------------------------------------- EnsembleAlgorithm
    #
    # The loop, as a state machine over the workflow's forecasts: perturb the observations
    # once and forecast the prior (and, for RLM-MAC, its mean model); then per iteration
    # propose a trial from the last *accepted* iterate, forecast it, and accept or reject
    # it. Members whose trial fails are dropped from the run even when the trial is
    # rejected, so the accepted and trial data mismatches compare the same members; the mean
    # model is then re-forecast. Mean-model forecasts inside an iteration are not
    # checkpoints: a crash there redoes that iteration's forecasts on resume.

    def _use_setup(self, setup: RunSetup) -> None:
        """Attach the diagnostics sink, and give the strategy the data keys if it wants them.

        A strategy that treats data types differently (margis) declares
        ``use_data_keys``; the history-matching data rows are fixed for the run.
        """
        self.attach_diagnostics(setup.diagnostics)
        use_keys = getattr(self.strategy, "use_data_keys", None)
        if use_keys is not None:
            use_keys(setup.observations.keys)

    def start(self, setup: RunSetup) -> Step:
        """Perturb the observations once and request the prior forecast."""
        self._use_setup(setup)
        prior = keep(setup.prior)
        member_ids = np.arange(prior.shape[-1])
        perturbed = keep(
            self.perturb_observations(
                setup.observations.values,
                setup.data_error,
                setup.streams["perturbations"],
                prior.shape[-1],
            )
        )
        self._run = _GiesRun(setup, perturbed=perturbed, trial=prior, member_ids=member_ids)
        if self.requires_prior_state:
            self._run.prior = prior  # each member's prior realisation (the full update)
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble", 0, prior, member_ids, with_sensitivities=self.uses_sensitivities
                )
            ],
            iteration=0,
            artifacts={
                "perturbed_observations": {
                    "perturbed_observations": perturbed,
                    "member_ids": member_ids,
                }
            },
            checkpoint=False,
        )

    def advance(self, forecasts: Forecasts) -> Step:
        """Take the requested forecasts and move the loop on."""
        run = self._require_run()
        if run.phase == "prior":
            forecast = forecasts["ensemble"]
            if self.hyperparameter_hook is not None and not forecast.ok.all():
                self.hyperparameter_hook.drop_members(forecast.ok)
            run.state = keep(run.trial, forecast.ok)
            if run.prior is not None:
                run.prior = keep(run.prior, forecast.ok)
            run.member_ids = run.member_ids[forecast.ok]
            run.perturbed = keep(run.perturbed, forecast.ok)
            run.predicted, run.trial_raw = forecast.predicted, forecast.raw
            run.sensitivity = forecast.sensitivity
            if self.requires_reference_prediction:
                run.phase = "prior_mean"
                return self._mean_step(run, 0)
            return self._finish_prior(run)
        if run.phase == "prior_mean":
            run.reference = forecasts["mean"].predicted[:, 0]
            return self._finish_prior(run)
        if run.phase == "trial":
            forecast = forecasts["ensemble"]
            if not forecast.ok.all():
                # A strategy with per-member loop state (the subspace update's weights)
                # reduces it too, or refuses, before the run's state changes.
                drop = getattr(self.strategy, "drop_members", None)
                if drop is not None:
                    drop(forecast.ok)
                if self.hyperparameter_hook is not None:
                    self.hyperparameter_hook.drop_members(forecast.ok)
            run.ok = forecast.ok
            run.trial_predicted, run.trial_raw = forecast.predicted, forecast.raw
            run.trial_sensitivity = forecast.sensitivity
            if not forecast.ok.all():
                # Compare like with like: score the accepted iterate on the survivors too.
                run.state, run.predicted = (
                    keep(run.state, forecast.ok),
                    keep(run.predicted, forecast.ok),
                )
                run.member_ids = run.member_ids[forecast.ok]
                if run.prior is not None:
                    run.prior = keep(run.prior, forecast.ok)
                run.perturbed = keep(run.perturbed, forecast.ok)
                run.data_mismatch = self.mean_data_mismatch(
                    run.predicted, run.perturbed, run.setup.data_error
                )
                if run.reference is not None:
                    run.phase = "reduced_mean"
                    return self._mean_step(run, run.iteration)
            return self._decide(run)
        if run.phase == "reduced_mean":
            run.reference = forecasts["mean"].predicted[:, 0]
            return self._decide(run)
        if run.phase == "accepted_mean":
            run.reference = forecasts["mean"].predicted[:, 0]
            return self._finish_iteration(run)
        raise RuntimeError(f"GIES cannot advance from phase {run.phase!r}")

    def _mean_step(self, run: "_GiesRun", iteration: int) -> Step:
        return Step(
            requests=[mean_model_request("mean", run.state, iteration)],
            iteration=iteration,
            checkpoint=False,
        )

    def _finish_prior(self, run: "_GiesRun") -> Step:
        run.data_mismatch = self.mean_data_mismatch(
            run.predicted, run.perturbed, run.setup.data_error
        )
        run.damping, run.iteration, run.converged = self._initial_damping(run), 0, False
        record = self._iterate_record(run, 0, run.state, run.predicted, run.trial_raw)
        # One-off set-up from the prior (e.g. a localization taper).
        self.prepare(
            run.state, run.predicted, run.setup.observations.keys, run.setup.streams["localization"]
        )
        return self._propose(run, [record])

    def _initial_damping(self, run: "_GiesRun") -> float:
        """Return the damping of the first analysis (a hook; GN-EnRML's is its gamma)."""
        if self.schedule.initial != "auto":
            return float(self.schedule.initial)
        # PET's lambda = 'auto': half the prior's mean data mismatch per datum.
        return float(0.5 * run.data_mismatch / run.perturbed.shape[0])

    def _spread(self, run: "_GiesRun", predicted: NDArray[np.float64]) -> float:
        """Return the population std over members of the data mismatch (PET ``np.std``)."""
        return float(np.std(run.setup.data_error.data_mismatch(run.perturbed - predicted)))

    def _analysis_damping(self, run: "_GiesRun") -> float:
        """Return the strategy's ``damping`` for the next trial (a hook; GN-EnRML's is 0).

        ``run.damping`` holds the method's control parameter: the damping here and in
        LM-EnRML, the step length in GN-EnRML.
        """
        return run.damping

    def _step_length(self, run: "_GiesRun") -> float:
        """Return the step length of the next trial (a hook: GN-EnRML's ``gamma``)."""
        return 1.0

    def _judge(self, run: "_GiesRun") -> DampingDecision:
        """Accept or reject the scored trial by the damping scheme (a hook; GN-EnRML's own).

        The spreads are recomputed from the forecasts (the accepted one on the current
        members), so no further loop state is needed for resume; ``rlmmac`` ignores them.
        """
        return self.schedule.decide(
            run.damping,
            run.data_mismatch,
            run.trial_data_mismatch,
            previous_spread=self._spread(run, run.predicted),
            spread=self._spread(run, run.trial_predicted),
        )

    def _decide(self, run: "_GiesRun") -> Step:
        run.trial_data_mismatch = self.mean_data_mismatch(
            run.trial_predicted, run.perturbed, run.setup.data_error
        )
        run.decision = self._judge(run)
        if run.decision.accepted:
            # A strategy that carries state between iterations (the subspace update's
            # weights) commits the trial's.
            accept = getattr(self.strategy, "accept_trial", None)
            if accept is not None:
                accept()
            run.state = keep(run.trial, run.ok)
            run.predicted, run.data_mismatch = run.trial_predicted, run.trial_data_mismatch
            run.sensitivity = run.trial_sensitivity
            if self.requires_reference_prediction:
                run.phase = "accepted_mean"
                return self._mean_step(run, run.iteration)
        return self._finish_iteration(run)

    def _finish_iteration(self, run: "_GiesRun") -> Step:
        decision = run.decision
        assert decision is not None
        used_damping = run.damping
        run.damping, run.converged = decision.damping, decision.converged
        # Consecutive rejections that barely moved: the update has stalled, and raising the
        # damping only shortens it further; stop rather than spend forecasts on it.
        run.stalls = run.stalls + 1 if decision.stalled else 0
        # Small improving trials while the damping was above its minimum: two in a row are
        # convergence (a scheme's confirm_convergence); one may only have been damped. A
        # rejection in between raises the damping again, so the count starts over: the
        # confirming trial must be a less damped one.
        run.tentatives = run.tentatives + 1 if decision.accepted and decision.tentative else 0
        if run.tentatives >= 2:
            run.converged = True
            logger.info(
                "two small improvements in a row: converged",
                extra={"event": "converged", "iteration": run.iteration},
            )
        limit = self._stall_limit()
        if limit is not None and run.stalls >= limit:
            run.converged = True
            logger.info(
                "the update stalled: stopping",
                extra={"event": "stalled", "iteration": run.iteration, "stalls": run.stalls},
            )
        if self.max_trials_per_iteration is not None:
            self._count_trial(run, accepted=decision.accepted)
        logger.info(
            "trial scored",
            extra={
                "event": "trial",
                "iteration": run.iteration,
                "accepted": decision.accepted,
                "data_mismatch": run.trial_data_mismatch,
                "damping": used_damping,
            },
        )
        trial_state = keep(run.trial, run.ok)
        if decision.accepted and self.hyperparameter_hook is not None:
            # The accepted iterate and, with RLM-MAC, its mean model's forecast are in.
            self.hyperparameter_hook.after_accept(
                run.state, run.predicted, run.reference, run.perturbed, run.setup.data_error,
                used_damping,
            )  # fmt: skip
        if decision.accepted:
            record = self._iterate_record(
                run, run.iteration, trial_state, run.trial_predicted, run.trial_raw, used_damping
            )
        else:
            record = Record(
                run.iteration,
                "trial",
                trial_state,
                run.trial_predicted,
                run.member_ids,
                run.perturbed,
                accepted=False,
                damping=used_damping,
                raw=run.trial_raw,
            )
        return self._propose(run, [record])

    def _stall_limit(self) -> int | None:
        """Return the scheme's ``max_stalled_rejections`` (``None``: no stall stop)."""
        limit = getattr(self.schedule, "max_stalled_rejections", None)
        return None if limit is None else int(limit)

    def _count_trial(self, run: "_GiesRun", *, accepted: bool) -> None:
        """Count a trial of the current outer iteration (``max_trials_per_iteration``)."""
        assert self.max_trials_per_iteration is not None
        run.trials += 1
        if accepted:
            run.outer, run.trials = run.outer + 1, 0
        elif run.trials >= self.max_trials_per_iteration:
            # No better iterate found: as the MATLAB RLM-MAC, increase the damping once
            # more, no lower than the initial one, and end the iteration without an update.
            initial = self.schedule.initial  # no floor for "auto": not kept as loop state
            floor = float(initial) if initial != "auto" else 0.0
            run.damping = max(run.damping * self.schedule.factor, floor)
            run.outer, run.trials = run.outer + 1, 0
            logger.info(
                "no accepted trial in this iteration",
                extra={
                    "event": "trials_exhausted",
                    "trial": run.iteration,
                    "outer": run.outer,
                    "damping": run.damping,
                },
            )

    def _budget_spent(self, run: "_GiesRun") -> bool:
        # Without retries every trial is an iteration; with them, outer iterations count.
        if self.max_trials_per_iteration is None:
            return run.iteration >= self.max_iterations
        return run.outer >= self.max_iterations

    def _iterate_record(
        self,
        run: "_GiesRun",
        iteration: int,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        raw: NDArray[np.float64] | None,
        damping: float | None = None,
    ) -> Record:
        extras = {} if run.reference is None else {"reference_prediction": run.reference}
        return Record(
            iteration,
            "iterate",
            state,
            predicted,
            run.member_ids,
            run.perturbed,
            damping=damping,
            raw=raw,
            extras=extras,
        )

    def _propose(self, run: "_GiesRun", records: list[Record]) -> Step:
        """Propose the next trial from the accepted iterate, or finish."""
        if run.converged or self._budget_spent(run):
            run.phase = "done"
            return Step(
                requests=[], iteration=run.iteration, records=records, converged=run.converged
            )
        run.iteration += 1
        if self.hyperparameter_hook is not None:
            self.hyperparameter_hook.before_update(run.state, run.predicted, run.reference)
        run.trial = keep(
            self.update(
                run.state,
                run.predicted,
                run.perturbed,
                run.setup.data_error,
                damping=self._analysis_damping(run),
                reference_prediction=run.reference,
                rng=run.setup.streams["localization"],
                prior_state=run.prior,
                step_length=self._step_length(run),
                sensitivity=run.sensitivity,
            )
        )
        run.phase = "trial"
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble",
                    run.iteration,
                    run.trial,
                    run.member_ids,
                    with_sensitivities=self.uses_sensitivities,
                )
            ],
            iteration=run.iteration,
            records=records,
            converged=run.converged,
        )

    def export_state(self) -> AlgorithmState:
        """Return the accepted iterate, the pending trial and the damping state."""
        run = self._require_run()
        if run.phase not in ("trial", "done"):
            raise RuntimeError(f"GIES is not at a checkpoint (phase {run.phase!r})")
        arrays: dict[str, NDArray[Any]] = {
            "state": run.state,
            "predicted": run.predicted,
            "member_ids": run.member_ids,
            "perturbed": run.perturbed,
            "trial": run.trial,
            **({} if run.prior is None else {"prior": run.prior}),
            # The accepted iterate's mean sensitivity: the next trial steps from it.
            **({} if run.sensitivity is None else {"sensitivity": run.sensitivity}),
        }
        if run.reference is not None:
            arrays["reference_prediction"] = run.reference
        arrays.update({f"method_state:{k}": v for k, v in self.export_prepared().items()})
        values = {
            "phase": run.phase,
            "iteration": run.iteration,
            "damping": run.damping,
            "data_mismatch": run.data_mismatch,
            "converged": run.converged,
            "outer": run.outer,
            "trials": run.trials,
            "stalls": run.stalls,
            "tentatives": run.tentatives,
        }
        return AlgorithmState(arrays=arrays, values=values)

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Continue from :meth:`export_state`: request the pending trial's forecast."""
        self._use_setup(setup)
        a, v = state.arrays, state.values
        run = _GiesRun(
            setup, perturbed=a["perturbed"], trial=a["trial"], member_ids=a["member_ids"]
        )
        run.state, run.predicted = a["state"], a["predicted"]
        run.reference = a.get("reference_prediction")
        run.prior = a.get("prior")
        run.sensitivity = a.get("sensitivity")
        if self.uses_sensitivities and run.sensitivity is None:
            raise ValueError("the checkpoint holds no sensitivity, which this strategy needs")
        if self.requires_prior_state and run.prior is None:
            raise ValueError("the checkpoint holds no prior ensemble, which this strategy needs")
        run.phase, run.iteration = str(v["phase"]), int(v["iteration"])
        run.damping = float(v["damping"])
        # Checkpoints before 2026-09-30 name it "misfit".
        stored = v["data_mismatch"] if "data_mismatch" in v else v["misfit"]
        run.data_mismatch = self._checked_data_mismatch(run, float(stored))
        run.converged = bool(v["converged"])
        # Checkpoints written before inner retries existed have neither (both 0 there). A
        # run's options are fixed by its config.json, so such a run never switches retries
        # on; if it did, its outer count would start again at 0.
        run.outer, run.trials = int(v.get("outer", 0)), int(v.get("trials", 0))
        run.stalls = int(v.get("stalls", 0))  # checkpoints before 2026-10-04 have none
        run.tentatives = int(v.get("tentatives", 0))  # checkpoints before 2026-10-19 have none
        prefix = "method_state:"
        self.restore_prepared({k[len(prefix) :]: x for k, x in a.items() if k.startswith(prefix)})
        self._run = run
        if run.phase == "done":
            return Step(requests=[], iteration=run.iteration, converged=run.converged)
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble",
                    run.iteration,
                    run.trial,
                    run.member_ids,
                    with_sensitivities=self.uses_sensitivities,
                )
            ],
            iteration=run.iteration,
            converged=run.converged,
        )

    def restore_legacy(self, setup: RunSetup, legacy: Mapping[str, Any]) -> Step:
        """Continue a run of the old iterative driver.

        It checkpointed the accepted iterate after iteration ``iteration``; the next trial
        had not been proposed, so it is proposed now, as the old driver would.
        """
        self._use_setup(setup)
        run = _GiesRun(
            setup,
            perturbed=legacy["perturbed"],
            trial=legacy["state"],
            member_ids=legacy["member_ids"],
        )
        run.state, run.predicted = legacy["state"], legacy["predicted"]
        run.reference = legacy["reference_prediction"]
        if self.requires_prior_state or self.uses_sensitivities:
            raise ValueError(
                "runs of the old iterative driver cannot resume with a prior-based or "
                "sensitivity-based strategy"
            )
        run.iteration, run.damping = int(legacy["iteration"]), float(legacy["damping"])
        stored = legacy["data_mismatch"] if "data_mismatch" in legacy else legacy["misfit"]
        run.data_mismatch = self._checked_data_mismatch(run, float(stored))
        run.converged = bool(legacy["converged"])
        self.restore_prepared(legacy["method_state"])
        self._run = run
        return self._propose(run, [])

    def max_forecasts(self) -> tuple[int, int]:
        """Return the most forecasts a run can take: ``(ensemble, single-member)``.

        The prior, then one trial per iteration, or up to ``max_trials_per_iteration`` per
        outer iteration. RLM-MAC also forecasts the mean model after the prior and, per
        trial, at most twice (on failed members, and after an acceptance).
        """
        trials = self.max_iterations * (self.max_trials_per_iteration or 1)
        single = 1 + 2 * trials if self.requires_reference_prediction else 0
        return 1 + trials, single

    def progress(self) -> IterationProgress:
        """Return the position in the iteration budget.

        Counts what ``max_iterations`` counts: trial iterates without retries, outer
        iterations with them (then the current trial is named too).
        """
        run = self._run
        total = self.max_iterations
        if run is None or run.phase in ("prior", "prior_mean"):
            return IterationProgress(0, total, "prior")
        if self.max_trials_per_iteration is None:
            if run.phase == "done":
                return IterationProgress(run.iteration, total, "done")
            return IterationProgress(run.iteration - 1, total, f"iteration {run.iteration}")
        if run.phase == "done":
            return IterationProgress(run.outer, total, "done")
        return IterationProgress(
            run.outer,
            total,
            f"iteration {run.outer + 1}, trial {run.trials + 1} of at most "
            f"{self.max_trials_per_iteration}",
        )

    def estimate(self) -> Estimate:
        """Return the last accepted iterate."""
        run = self._require_run()
        return Estimate(run.state, run.predicted, run.member_ids)

    def _require_run(self) -> "_GiesRun":
        if self._run is None:
            raise RuntimeError("GIES has not started: call start() or restore_state() first")
        return self._run


@dataclass
class _GiesRun:
    """GIES's loop state (see :meth:`GIES.export_state`)."""

    setup: RunSetup
    perturbed: NDArray[np.float64]
    trial: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    phase: str = "prior"
    state: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    predicted: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    reference: NDArray[np.float64] | None = None
    iteration: int = 0
    damping: float = 0.0
    data_mismatch: float = math.inf
    converged: bool = False
    #: With max_trials_per_iteration: completed outer iterations, and trials in this one.
    outer: int = 0
    trials: int = 0
    #: Consecutive rejected trials within the tolerance (``max_stalled_rejections``).
    stalls: int = 0
    #: Consecutive accepted trials within the tolerance taken with damping above its minimum.
    tentatives: int = 0
    ok: NDArray[np.bool_] = field(default_factory=lambda: np.empty(0, dtype=bool))
    trial_predicted: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    trial_raw: NDArray[np.float64] | None = None
    trial_data_mismatch: float = math.inf
    decision: DampingDecision | None = None
    #: Each member's prior state, for strategies that need it (``requires_prior_state``).
    prior: NDArray[np.float64] | None = None
    #: The accepted iterate's and the trial's member-mean sensitivity. After a
    #: member drop the accepted one keeps its mean over the members it was formed from.
    sensitivity: NDArray[np.float64] | None = None
    trial_sensitivity: NDArray[np.float64] | None = None


logger = logging.getLogger(__name__)

#: Relative difference between a checkpoint's stored accepted data mismatch and the one recomputed
#: on resume above which the checkpoint is refused. Rounding across CPUs and NumPy/OpenBLAS
#: versions is ~1e-16 to 1e-14 relative; a changed formula or corrupted arrays is far above
#: 1e-10.
RESUME_DATA_MISMATCH_RTOL = 1e-10
