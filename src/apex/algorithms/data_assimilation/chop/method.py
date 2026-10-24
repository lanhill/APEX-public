"""CHOP: continuous hyper-parameter optimization of an iterative smoother.

``methods["chop"]`` wraps an iterative method on GIES's loop (``gies``, ``lm_enrml``; not
``gn_enrml``, which refuses RLM-MAC) with an RLM-MAC strategy, and estimates its
components' hyper-parameters as the run goes. Each member ``j`` carries its own vector
``theta_j``:

- **the model update** uses member ``j``'s hyper-parameters through the components'
  Schur factors on the shared gain (:class:`~.taper.ChopTaper`, set as the strategy's
  localization);
- **the hyper-parameter update**, after each accepted trial, is an RLM-MAC step on the
  hyper-parameter block from the accepted iterate's forecast:

  ``theta_j <- theta_j + (T_theta(theta_j) o K_theta) L^-1 (d_j - g(m_j^{i+1}))``,
  ``K_theta = S_theta V diag(s / (s^2 + gamma')) U^T``, ``L^-1 S_g^{i+1} = U S V^T``,

  with ``S_theta`` the hyper-parameters' anomalies, ``S_g^{i+1}`` the new forecasts'
  anomalies about the new mean model's forecast, ``gamma' = lambda * mean(s^2)`` with the
  accepted trial's ``lambda``, and the run's perturbed observations ``d_j``. It is
  :class:`~apex.algorithms.data_assimilation.gies.RLMMACUpdate` on the hyper-parameters,
  with the inner strategy's truncation energy and ``hyper_localization``'s taper.

The published MATLAB code updates the hyper-parameters in every trial and discards the
update with a rejected one; updating after an acceptance only is the same, and uses the
mean-model forecast GIES requests after an acceptance (no extra simulation).

Reference
---------
Luo, X., Cruz, W. C., Zhang, X.-L. & Xiao, H. (2023). Hyper-parameter optimization for
improving the performance of localization in an iterative ensemble smoother. Geoenergy
Science and Engineering, 231, 212404. doi:10.1016/j.geoen.2023.212404.
Equation labels of the user's manuscript ``P_AutoAdaLoc_R1``: (eq:rlm_mac_chop_2),
(eq:rlm_mac_KG_loc_chop_final)
and Algorithm 2; reference code ``RLM_MAC_CHOP.m``, ``update_hyper.m``,
``update_hyper_no_loc.m`` (5Spots and Brugge).
Differences from the code:

- correlations of the initial ensemble by default (``correlation = "current"`` is the
  code's); the taper factor is the code's 2 by default (the paper's formula is 1);
- the correlations are not shrunk by ``(N - 1) / N`` (``getCorrMtx``), and rows or columns
  without spread get the floored std of ``localization.correlation`` (the code zeroes
  them);
- no ``alpha >= 1e-3`` floor; the truncation keeps the first rank whose share *reaches*
  the energy (the code's is strictly greater) and drops numerically null directions;
- the scale is ``factor * max(l, 1e-12)`` (the code's ``factor * |l + 1e-6|``);
- the hyper-parameters' anomalies are about the members' mean (the code's, about its
  stored mean column, differ once the floor binds), and the mean model is the mean of the
  clipped members (the code's is the clipped mean).
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.algorithm import (
    AlgorithmState,
    Estimate,
    Forecasts,
    RunSetup,
    Step,
)
from apex.algorithms.data_assimilation.chop.components import (
    ChopComponent,
    ComponentContext,
    stack_sizes,
)
from apex.algorithms.data_assimilation.chop.taper import ChopTaper, HyperTaper
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.gies import RLMMACUpdate
from apex.algorithms.data_assimilation.localization import correlation
from apex.algorithms.data_assimilation.registries import (
    chop_components,
    localizations,
    methods,
)
from apex.core.options import TEXT

#: The paper's P-AutoAdaLoc-0: one Gaspari-Cohn length scale per datum and member.
DEFAULT_COMPONENTS: tuple[Mapping[str, Any], ...] = ({"name": "length_scale", "mode": "data"},)
#: The key of the hyper-parameters in iterate records and checkpoints.
HYPERPARAMETERS = "hyperparameters"
_STATE_KEY = "chop:theta"


def build_components(entries: Sequence[Mapping[str, Any]]) -> list[ChopComponent]:
    """Return the components of ``components`` entries (``{"name": ..., **options}``)."""
    if not entries:
        raise ValueError("CHOP needs at least one component in components")
    built = []
    for entry in entries:
        options = dict(entry)
        name = options.pop("name", None)
        if not isinstance(name, str):
            raise ValueError(f"each CHOP component needs a name, got {entry!r}")
        built.append(chop_components.create(name, **options))
    return built


@methods.register("chop")
class Chop:
    """Estimate a wrapped method's localization and gain hyper-parameters per member as it runs.

    CHOP (P-AutoAdaLoc; Luo et al. 2023, Geoenergy Sci. Eng. 231, 212404) wraps a method on
    GIES's loop with an RLM-MAC strategy (``[method.options] inner``, default ``gies``).

    Parameters
    ----------
    inner_method
        The wrapped method (built by the composition from ``inner``).
    components
        The estimated components: ``[{name = "length_scale", mode = "data", ...}, ...]``
        (registry ``chop_components``); default the paper's, one length scale per datum.
    hyper_localization
        How the hyper-parameter update is localized: ``"same"`` (each component's own
        taper, the default), ``"none"``, or a registered localization's name (or
        ``{name = ..., options}``) applied to the hyper-parameter block.
    block_sizes
        The state variables' sizes (supplied by the composition from ``[state]``).
    max_batch_mb
        The memory of one row block of the model update's per-member tapers.
    """

    iterative = True

    def __init__(
        self,
        inner_method: Any,  # noqa: ANN401 (an EnsembleAlgorithm on GIES's loop)
        components: Annotated[Sequence[Mapping[str, Any]], TEXT] = DEFAULT_COMPONENTS,
        hyper_localization: Annotated[str | Mapping[str, Any], TEXT] = "same",
        block_sizes: Sequence[int] = (),
        max_batch_mb: float = 256.0,
    ) -> None:
        if not hasattr(inner_method, "hyperparameter_hook"):
            raise ValueError(
                f"CHOP wraps the methods on GIES's loop with RLM-MAC (gies, lm_enrml), not "
                f"{type(inner_method).__name__}"
            )
        strategy = inner_method.strategy
        if not getattr(strategy, "per_member_taper", False):
            raise ValueError(
                "CHOP needs an RLM-MAC strategy (rlmmac), whose gain it tapers per member and "
                f"whose mean-model forecast its hyper-parameter update uses; got "
                f"{type(strategy).__name__}"
            )
        if not getattr(strategy, "scale_data_anomalies", True):
            # The option reproduces PET-fork's GIES runs; CHOP follows the author's MATLAB,
            # whose step is the default's, and its hyper-parameter update shares the damping.
            raise ValueError(
                "CHOP needs rlmmac's default scale_data_anomalies = true (false only reproduces "
                "PET-fork's GIES runs): remove the option"
            )
        if strategy.localization is not None:
            raise ValueError(
                "CHOP's components localize the update (length_scale): remove [method.localization]"
            )
        self.inner = inner_method
        self.components = build_components(components)
        self.block_sizes = tuple(int(b) for b in block_sizes)
        self.taper = ChopTaper(self.components, max_batch_mb)
        strategy.localization = self.taper
        self.hyper_localization = hyper_localization
        self._hyper_taper: Any = self._hyper_localizer(hyper_localization)
        self._hyper_prepared = False
        self.hyper_strategy = RLMMACUpdate(
            energy=float(getattr(strategy, "energy", 0.95)), localization=self._hyper_taper
        )
        self.context: ComponentContext | None = None
        self.slices: list[slice] = []
        #: The members' hyper-parameters in the update's (transformed) space.
        self.theta_u: NDArray[np.float64] | None = None
        self._setup: RunSetup | None = None
        inner_method.hyperparameter_hook = self

    def _hyper_localizer(self, spec: str | Mapping[str, Any]) -> Any:  # noqa: ANN401
        if spec == "same":
            return HyperTaper(self.components)
        if spec == "none":
            return None
        if isinstance(spec, str):
            return spec  # a registered localization, created at start (sizes needed)
        if isinstance(spec, Mapping) and isinstance(spec.get("name"), str):
            return dict(spec)
        raise ValueError(f"hyper_localization must be same, none or a localization, got {spec!r}")

    # ------------------------------------------------------------------ delegation

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 (the inner method's attributes)
        """Return the inner method's attribute (status, progress(), max_iterations, ...)."""
        # Only reached for attributes CHOP lacks.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    # ------------------------------------------------------------------ the run

    def _configure(self, setup: RunSetup, n_ensemble: int) -> None:
        n_state = setup.prior.shape[0]
        keys = setup.observations.keys
        blocks = self.block_sizes or (n_state,)
        if sum(blocks) != n_state:
            raise ValueError(f"the state's blocks {blocks} do not add up to {n_state} rows")
        self.context = ComponentContext(
            n_state=n_state, n_data=len(keys), block_sizes=blocks,
            datatypes=tuple(k.datatype for k in keys), n_ensemble=n_ensemble,
        )  # fmt: skip
        self.slices = stack_sizes(self.components, self.context)
        self.taper.configure(self.context, self.slices)
        if isinstance(self._hyper_taper, HyperTaper):
            self._hyper_taper.configure(self.context, self.slices)
        elif isinstance(self._hyper_taper, (str, dict)):
            spec = self._hyper_taper
            name, options = (
                (spec, {})
                if isinstance(spec, str)
                else (spec["name"], dict(spec.get("options", {})))
            )
            sizes = [part.stop - part.start for part in self.slices]
            self._hyper_taper = localizations.create(name, block_sizes=sizes, **options)
            self.hyper_strategy.localization = self._hyper_taper
        self._setup = setup

    def natural(self) -> NDArray[np.float64]:
        """Return the members' natural hyper-parameters, ``(n_theta, n_ensemble)``."""
        if self.theta_u is None:
            raise RuntimeError("CHOP has not started")
        return np.ascontiguousarray(np.vstack([
            component.from_u(self.theta_u[part])
            for component, part in zip(self.components, self.slices, strict=True)
        ]))  # fmt: skip

    def start(self, setup: RunSetup) -> Step:
        """Draw the hyper-parameters' prior, then start the inner method."""
        self._configure(setup, setup.prior.shape[-1])
        assert self.context is not None
        rng = setup.streams["chop"]
        natural = [component.prior(rng, self.context) for component in self.components]
        self.theta_u = np.ascontiguousarray(np.vstack([
            component.to_u(block) for component, block in zip(self.components, natural,
                                                              strict=True)
        ]))  # fmt: skip
        self.taper.theta = self.natural()
        return self._with_hyperparameters(self.inner.start(setup))

    def advance(self, forecasts: Forecasts) -> Step:
        """Move the inner method on (its hooks update the hyper-parameters)."""
        return self._with_hyperparameters(self.inner.advance(forecasts))

    def _with_hyperparameters(self, step: Step) -> Step:
        """Save the hyper-parameters with every iterate the step records."""
        if not step.records:
            return step
        natural = self.natural()
        records = [
            dataclasses.replace(record, extras={**record.extras, HYPERPARAMETERS: natural})
            if record.kind == "iterate" else record
            for record in step.records
        ]  # fmt: skip
        return dataclasses.replace(step, records=records)

    def estimate(self) -> Estimate:
        """Return the inner method's estimate."""
        estimate: Estimate = self.inner.estimate()
        return estimate

    # ------------------------------------------------------------------ hooks

    def before_update(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        reference: NDArray[np.float64] | None,
    ) -> None:
        """Give the model taper the current hyper-parameters (and correlations)."""
        self.taper.set_current(state, predicted)
        self.taper.theta = self.natural()

    def after_accept(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        reference: NDArray[np.float64] | None,
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        damping: float,
    ) -> None:
        """Update the hyper-parameters from the accepted iterate's forecast."""
        if self.theta_u is None or self._setup is None:
            raise RuntimeError("CHOP has not started")
        if reference is None:
            raise RuntimeError("CHOP's hyper-parameter update needs the mean model's forecast")
        rng = self._setup.streams["chop"]
        if isinstance(self._hyper_taper, HyperTaper):
            self._hyper_taper.theta = self.natural()
            self._hyper_taper.correlation = correlation(self.theta_u, predicted)
        elif self._hyper_taper is not None and not self._hyper_prepared:
            # A registered localization: set up from the first update's ensembles.
            self._hyper_taper.prepare(self.theta_u, predicted, self._setup.observations.keys, rng)
            self._hyper_prepared = True
        step = self.hyper_strategy.step(
            self.theta_u, predicted, perturbed_observations, data_error,
            damping=damping, reference_prediction=reference, rng=rng,
        )  # fmt: skip
        updated = self.theta_u + step
        # Within each component's bounds (``none``: floored at ``floor``).
        self.theta_u = np.ascontiguousarray(np.vstack([
            component.bounded(updated[part])
            for component, part in zip(self.components, self.slices, strict=True)
        ]))  # fmt: skip

    def drop_members(self, ok: NDArray[np.bool_]) -> None:
        """Drop the failed members' hyper-parameters."""
        if self.theta_u is not None:
            self.theta_u = np.ascontiguousarray(self.theta_u[:, ok])

    # ------------------------------------------------------------------ checkpoints

    def export_state(self) -> AlgorithmState:
        """Return the inner method's state with the hyper-parameters."""
        if self.theta_u is None:
            raise RuntimeError("CHOP has not started")
        inner: AlgorithmState = self.inner.export_state()
        values = {**inner.values, "chop:hyper_prepared": self._hyper_prepared}
        arrays = {**inner.arrays, _STATE_KEY: self.theta_u}
        if self._hyper_prepared and not isinstance(self._hyper_taper, HyperTaper):
            # A registered localization's prepared arrays (its prior taper, ...).
            prepared = self._hyper_taper.export_state()
            arrays.update({f"chop:hyper_loc:{k}": v for k, v in prepared.items()})
        return AlgorithmState(arrays=arrays, values=values)

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Restore the hyper-parameters, then continue the inner method."""
        theta = np.ascontiguousarray(state.arrays[_STATE_KEY])
        self._configure(setup, theta.shape[1])
        self.theta_u = theta
        self._hyper_prepared = bool(state.values.get("chop:hyper_prepared", False))
        if (
            self._hyper_prepared
            and self._hyper_taper is not None
            and not isinstance(self._hyper_taper, HyperTaper)
        ):
            prefix = "chop:hyper_loc:"
            self._hyper_taper.restore_state(
                {k[len(prefix) :]: v for k, v in state.arrays.items() if k.startswith(prefix)}
            )
        self.taper.theta = self.natural()
        arrays = {k: v for k, v in state.arrays.items() if not k.startswith("chop:")}
        values = {k: v for k, v in state.values.items() if not k.startswith("chop:")}
        step: Step = self.inner.restore_state(setup, AlgorithmState(arrays, values))
        return step
