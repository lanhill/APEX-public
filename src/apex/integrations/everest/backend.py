"""APEX's optimizers as a ``ropt`` backend, so Everest can run them.

Adapted from ropt
-----------------
Library   : ropt 0.28 (github.com/TNO-ropt/ropt; GPL-3.0), the backend interface of
            ``ropt.backend.Backend`` and its SciPy backend's handling of masks and bounds;
            called, not copied. ropt is installed with ERT 26 (the ``ert`` extra).
Reference : Chen, Y., Oliver, D. S. & Zhang, D. (2009). Efficient ensemble-based closed-loop
            production optimization. SPE Journal, 14(4), 634-645 (EnOpt).

How it maps:

- **The objective** is ropt's: the callback's function value of each variable vector,
  which ropt has already combined over the realizations (its function estimator). APEX
  therefore sees one realization. Only the free variables (ropt's ``mask``) are optimized.
- **The ensemble optimizers** (EnOpt, GenOpt, SmcOpt) draw their own perturbations and ask
  for the members' values as one batch (``is_parallel``); their covariance is
  ``diag(perturbation_magnitudes**2)`` from ropt's variables, or the ``variance`` option.
  Line search and trust region ask ropt for its gradient (ropt's ensemble estimate);
  methods that need a Hessian (``Newton-CG``, an exact-Hessian trust region) are refused.
- **Nonlinear constraints** become APEX's external penalty (``epf``, ensemble optimizers
  only; its factor ``r`` must be given): ``c = b`` for equal bounds, ``c - lower >= 0``
  and ``upper - c >= 0`` otherwise,
  from the same callback values (a batch's are cached, so constraints cost no extra
  evaluation). Linear constraints are refused.
- **Randomness:** the draws are seeded from ropt's ``variables.seed``, spawned as
  ``apex optimize`` spawns ``[optimization] seed``, so a run through ropt follows the same
  path as the same problem run by APEX.
- **Limits:** ropt's ``max_iterations`` is the optimizer's ``maxiter`` unless the options
  give it (with EPF, per outer iteration: up to ``max_epf_iter`` times as many in all); its
  ``convergence_tolerance`` is ``tol`` for the optimizers that have it. ropt's own limits
  (``max_functions``, ``max_batches``) abort through the callback.
- **Units:** everything happens in ropt's (Everest's) optimizer domain: Everest scales
  controls to ``scaled_range`` ((0, 1) by default), and ropt transforms bounds and absolute
  perturbation magnitudes into it. Options in control units (``variance``, ``alpha``,
  ``xtol``, ``epf.r`` against scaled constraints) must be given in that domain.
- **Feasibility:** the external penalty is quadratic, so iterates approach an active
  constraint from outside and stop within ``conv_crit``; ropt keeps as "best" only points
  within its constraint tolerance. A large ``epf.r`` (and small ``conv_crit``) brings the
  last iterate closer; ropt or Everest may otherwise report an earlier feasible point, or
  none for an equality constraint.
"""

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from ropt.backend import Backend
from ropt.enums import PerturbationType
from ropt.plugins.backend import BackendPlugin

from apex.algorithms.optimization import GeneratorDraws, OptimizerBase, Problem, optimizers

if TYPE_CHECKING:
    from ropt.config import BackendConfig
    from ropt.context import EnOptContext
    from ropt.core import OptimizerCallback

type FloatArray = NDArray[np.float64]

#: The APEX optimizers offered to ropt, and the default.
METHODS = ("enopt", "genopt", "smcopt", "line_search", "trust_region")
DEFAULT_METHOD = "enopt"
#: Options of this backend itself (not passed to the APEX optimizer).
OWN_OPTIONS = ("variance",)


def _method(name: str) -> str:
    _, _, method = name.lower().rpartition("/")
    return DEFAULT_METHOD if method == "default" else method


class _RoptObjective:
    """ropt's objective (and constraints) as an APEX objective of physical controls."""

    def __init__(
        self,
        callback: "OptimizerCallback",
        context: "EnOptContext",
        constraint_bounds: tuple[FloatArray, FloatArray] | None,
    ) -> None:
        self.callback = callback
        self.context = context
        self.bounds = constraint_bounds
        self._cache: dict[bytes, FloatArray] = {}

    def _functions(self, controls: FloatArray) -> FloatArray:
        """Return ropt's function values ``(n, 1 + n_constraints)`` of ``(n_x, n)`` controls."""
        key = np.ascontiguousarray(controls).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self.callback(
            np.ascontiguousarray(controls.T), return_functions=True, return_gradients=False
        )
        if result.functions is None:
            raise RuntimeError("ropt returned no function values for a function request")
        functions = np.atleast_2d(np.asarray(result.functions, dtype=np.float64))
        if result.nonlinear_constraint_bounds is not None:
            self.bounds = result.nonlinear_constraint_bounds
        # One batch is kept: the loop asks for the constraints of what it just evaluated.
        self._cache = {key: functions}
        return functions

    def __call__(self, controls: FloatArray, realizations: NDArray[np.int64]) -> FloatArray:
        """Return the objective of each column ``(n,)``."""
        return np.asarray(self._functions(controls)[:, 0], dtype=np.float64)

    def gradient(self, x: FloatArray) -> FloatArray:
        """Return ropt's gradient of the objective ``(n_x,)``."""
        result = self.callback(x, return_functions=False, return_gradients=True)
        if result.gradients is None:
            raise RuntimeError("ropt returned no gradient for a gradient request")
        gradients: FloatArray = np.asarray(result.gradients, dtype=np.float64)
        return np.asarray(gradients[0], dtype=np.float64)

    def hessian(self, x: FloatArray) -> FloatArray:
        """Ropt gives no Hessian (methods that need one are refused)."""
        raise NotImplementedError("ropt gives no Hessian")

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return APEX's ``c_eq`` (zero when met) and ``c_iq`` (non-negative when met)."""
        values = self._functions(controls)[:, 1:].T  # (n_constraints, n)
        if self.bounds is None:
            raise ValueError("the problem has no nonlinear constraints for the penalty")
        lower, upper = (np.asarray(b, dtype=np.float64) for b in self.bounds)
        equal = lower == upper
        c_eq = values[equal] - lower[equal][:, None]
        rows = [values[i] - lower[i] for i in np.flatnonzero(~equal & np.isfinite(lower))]
        rows += [upper[i] - values[i] for i in np.flatnonzero(~equal & np.isfinite(upper))]
        n = controls.shape[1]
        c_iq = np.vstack(rows) if rows else np.zeros((0, n))
        return np.asarray(c_eq, dtype=np.float64), np.asarray(c_iq, dtype=np.float64)


class ApexBackend(Backend):
    """APEX's optimizers behind ropt's backend interface (see the module docstring)."""

    def __init__(self, backend_config: "BackendConfig") -> None:
        self._config = backend_config
        self._method = _method(backend_config.method)
        if self._method not in METHODS:
            raise NotImplementedError(f"APEX has no optimizer {self._method!r}; use {METHODS}")

    def init(self, context: "EnOptContext", optimizer_callback: "OptimizerCallback") -> None:
        """Keep the context and callback; refuse what APEX cannot honour."""
        self._context = context
        self._callback = optimizer_callback
        if context.linear_constraints is not None:
            raise ValueError("APEX's optimizers do not take linear constraints")
        self._constraints = context.nonlinear_constraints is not None
        if self._constraints and self._method in ("line_search", "trust_region"):
            raise ValueError(
                f"apex/{self._method} takes no constraints (APEX's penalty needs an ensemble "
                "optimizer: apex/enopt, apex/genopt or apex/smcopt)"
            )

    def validate_options(self) -> None:
        """Check the options by creating the optimizer from them."""
        options = self._config.options
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("APEX backend options must be a dictionary")
        try:
            self._optimizer()
        except (TypeError, ValueError) as error:
            raise ValueError(f"apex/{self._method}: {error}") from error

    @property
    def is_parallel(self) -> bool:
        """The ensemble optimizers evaluate their members as one batch."""
        return self._method in ("enopt", "genopt", "smcopt")

    def start(self, initial_values: FloatArray) -> None:
        """Run the APEX optimizer from ``initial_values`` (all variables; masked here)."""
        variables = self._context.variables
        mask = np.asarray(variables.mask, dtype=bool)
        x0 = np.asarray(initial_values, dtype=np.float64)[mask]
        lower = np.broadcast_to(variables.lower_bounds, mask.shape)[mask]
        upper = np.broadcast_to(variables.upper_bounds, mask.shape)[mask]
        bounds = None
        if self._constraints:
            assert self._context.nonlinear_constraints is not None
            bounds = (
                np.asarray(self._context.nonlinear_constraints.lower_bounds, dtype=np.float64),
                np.asarray(self._context.nonlinear_constraints.upper_bounds, dtype=np.float64),
            )
        objective = _RoptObjective(self._callback, self._context, bounds)
        problem = Problem(
            objective,
            x0,
            lower=None if np.all(np.isinf(lower)) else lower,
            upper=None if np.all(np.isinf(upper)) else upper,
        )
        optimizer = self._optimizer()
        cov = None
        if optimizer.needs_covariance:
            cov = np.diag(self._variances(mask))
        (child,) = np.random.SeedSequence(self._seed()).spawn(1)
        optimizer.minimize(problem, GeneratorDraws(np.random.default_rng(child)), cov=cov)

    # ------------------------------------------------------------------ helpers

    def _options(self) -> dict[str, Any]:
        raw = self._config.options
        options: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
        return {k: v for k, v in options.items() if k not in OWN_OPTIONS}

    def _optimizer(self) -> OptimizerBase:
        options = self._options()
        factory = optimizers.factory(self._method)
        accepted = inspect.signature(factory).parameters
        if "maxiter" not in options and self._config.max_iterations is not None:
            options["maxiter"] = self._config.max_iterations
        tolerance = self._config.convergence_tolerance
        if "tol" in accepted and "tol" not in options and tolerance is not None:
            options["tol"] = tolerance
        if self._method == "line_search" and options.get("method") == "Newton-CG":
            raise ValueError("Newton-CG needs a Hessian, which ropt does not give")
        if self._method == "trust_region" and not options.get("quasi_newton", False):
            options["quasi_newton"] = True  # ropt gives no Hessian: BFGS (said in the docs)
        constrained = getattr(self, "_constraints", None)
        if constrained is False and options.get("epf"):
            raise ValueError("epf (the external penalty) needs nonlinear constraints")
        if constrained and "r" not in (options.get("epf") or {}):
            # The penalty factor sets how hard constraints bind against the objective: a
            # property of the problem's scales that no default can know.
            raise ValueError(
                "nonlinear constraints need the external penalty's factor: options "
                "epf = {r = ...} (and optionally r_factor, tol_factor, conv_crit, max_epf_iter)"
            )
        return optimizers.create(self._method, **options)

    def _variances(self, mask: NDArray[np.bool_]) -> FloatArray:
        raw = self._config.options
        given = raw.get("variance") if isinstance(raw, Mapping) else None
        if given is not None:
            values = np.broadcast_to(np.asarray(given, dtype=np.float64), mask.shape)
            return np.asarray(values[mask], dtype=np.float64)
        variables = self._context.variables
        magnitudes = np.broadcast_to(variables.perturbation_magnitudes, mask.shape)
        # As ropt's own gradient (core/_gradient.py): a RELATIVE magnitude is a fraction of
        # the variable's range.
        relative = np.broadcast_to(variables.perturbation_types, mask.shape) == (
            PerturbationType.RELATIVE
        )
        span = np.broadcast_to(variables.upper_bounds, mask.shape) - np.broadcast_to(
            variables.lower_bounds, mask.shape
        )
        if np.any(relative & ~np.isfinite(span)):
            raise ValueError("a relative perturbation needs finite bounds")
        scaled = np.where(relative, span * magnitudes, magnitudes)
        return np.asarray(scaled[mask], dtype=np.float64) ** 2

    def _seed(self) -> list[int]:
        seed = self._context.variables.seed
        return [int(s) for s in (seed if isinstance(seed, tuple | list) else (seed,))]


class ApexBackendPlugin(BackendPlugin):
    """The ropt plugin that creates :class:`ApexBackend` (entry point ``apex``)."""

    @classmethod
    def create(cls, backend_config: "BackendConfig") -> ApexBackend:
        """Return the backend for ``backend_config``."""
        return ApexBackend(backend_config)

    @classmethod
    def is_supported(cls, method: str) -> bool:
        """Whether ``method`` (``name`` or ``default``) is one of APEX's optimizers."""
        return method.lower() in (*METHODS, "default")
