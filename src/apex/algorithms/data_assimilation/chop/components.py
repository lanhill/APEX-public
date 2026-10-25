"""CHOP's hyper-parameterized components (registry ``chop_components``).

A component owns a block of every member's hyper-parameter vector ``theta_j`` and says how
it acts on member ``j``'s update: here, always as a Schur factor on the shared gain
``K = dX G`` (``(n_state, n_data)``), so that

``dm_j = (F_j o K) r_j``, ``F_j`` the product of the components' factors for member ``j``,

with ``r_j`` member ``j``'s whitened innovation. A component also says how its rows of the
hyper-parameter update are tapered when that update is localized (``hyper_taper``).

The hyper-parameters are updated in a transformed space ``u`` (``transform``): ``none``
(natural units, then ``floor``, as the published MATLAB code), ``log`` (positive) or
``logit`` (on ``[lower, upper]``).

Reference
---------
Luo, X., Cruz, W. C., Zhang, X.-L. & Xiao, H. (2023). Hyper-parameter optimization for
improving the performance of localization in an iterative ensemble smoother.
Geoenergy Science and Engineering, 231, 212404. doi:10.1016/j.geoen.2023.212404.
Equation labels of the user's manuscript ``P_AutoAdaLoc_R1``:
(eq:GC_tapering_rule_parameterized_chop) and
(eq:rlm_mac_KG_loc_chop_final). The ``length_scale`` component is the paper's P-AutoAdaLoc;
``gain_inflation`` and ``data_weights`` apply its derivation (which holds for any
member-local hyper-parameters) to inflation factors.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from apex.algorithms.data_assimilation.localization import gaspari_cohn
from apex.algorithms.data_assimilation.registries import chop_components

Transform = Literal["none", "log", "logit"]


@dataclass(frozen=True)
class ComponentContext:
    """What a component's size and factors depend on.

    Attributes
    ----------
    n_state, n_data
        The state's and the data's sizes.
    block_sizes
        The state variables' sizes, in layout order (they sum to ``n_state``).
    datatypes
        Each datum's data type, ``(n_data,)``.
    n_ensemble
        The ensemble size.
    """

    n_state: int
    n_data: int
    block_sizes: tuple[int, ...]
    datatypes: tuple[str, ...]
    n_ensemble: int

    def variable_of_rows(self) -> NDArray[np.int64]:
        """Return each state row's variable index, ``(n_state,)``."""
        return np.repeat(np.arange(len(self.block_sizes)), self.block_sizes)

    def type_of_data(self) -> tuple[list[str], NDArray[np.int64]]:
        """Return the data types (first-seen order) and each datum's index into them."""
        names = list(dict.fromkeys(self.datatypes))
        index = {name: i for i, name in enumerate(names)}
        return names, np.array([index[t] for t in self.datatypes], dtype=np.int64)


class ChopComponent:
    """Base of the components: prior, transform and the default hyper-update taper.

    Parameters
    ----------
    low, high
        The uniform prior's range of the natural values.
    transform
        ``none`` (natural units, floored), ``log`` or ``logit``.
    floor
        The smallest natural value with ``transform = "none"`` (the code's 1e-6).
    lower, upper
        ``logit``'s bounds (default: ``low`` and ``high``, widened by their range).
    threshold
        The hyper-update taper's ``c`` in ``|corr| > c / sqrt(N)`` (components without a
        taper of their own).
    """

    #: Whether the factors use correlations of the state with the data.
    needs_correlation: ClassVar[bool] = False

    def __init__(
        self,
        low: float,
        high: float,
        transform: Transform = "none",
        floor: float = 1e-6,
        lower: float | None = None,
        upper: float | None = None,
        threshold: float = 3.0,
    ) -> None:
        if not low <= high:
            raise ValueError(f"the prior's low ({low}) must not exceed its high ({high})")
        if transform not in ("none", "log", "logit"):
            raise ValueError(f"transform must be none, log or logit, got {transform!r}")
        if transform == "log" and low <= 0:
            raise ValueError("a log transform needs a positive prior (low > 0)")
        width = high - low
        # The prior's range widened by its width, never below 0: every component's values
        # are positive quantities (a negative inflation would flip the gain's sign).
        self.lower = max(0.0, low - width) if lower is None else lower
        self.upper = high + width if upper is None else upper
        if self.lower < 0:
            raise ValueError(f"lower must not be negative, got {self.lower}")
        if transform == "logit" and not self.lower < low <= high < self.upper:
            raise ValueError(
                f"a logit transform needs lower < low <= high < upper, got lower={self.lower}, "
                f"[{low}, {high}], upper={self.upper}"
            )
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self.low, self.high = float(low), float(high)
        self.transform: Transform = transform
        self.floor = float(floor)
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ size and prior

    def size(self, context: ComponentContext) -> int:
        """Return the number of hyper-parameters per member."""
        raise NotImplementedError

    def prior(self, rng: np.random.Generator, context: ComponentContext) -> NDArray[np.float64]:
        """Return the natural values' prior draw, ``(size, n_ensemble)``."""
        shape = (self.size(context), context.n_ensemble)
        drawn: NDArray[np.float64] = rng.uniform(self.low, self.high, size=shape)
        return drawn

    # ------------------------------------------------------------------ transform

    def to_u(self, natural: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the transformed values the update works on."""
        if self.transform == "log":
            return np.log(natural)
        if self.transform == "logit":
            scaled: NDArray[np.float64] = logit((natural - self.lower) / (self.upper - self.lower))
            return scaled
        return np.array(natural, dtype=np.float64)

    def from_u(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the natural values of transformed ones (``none``: floored)."""
        if self.transform == "log":
            return np.exp(u)
        if self.transform == "logit":
            natural: NDArray[np.float64] = self.lower + (self.upper - self.lower) * expit(u)
            return natural
        return np.maximum(u, self.floor)

    def bounded(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return updated values within bounds.

        ``none`` floors them; ``log`` and ``logit`` are bounded by their maps already (a
        round trip through the natural values would overflow to infinity).
        """
        if self.transform == "none":
            return np.maximum(u, self.floor)
        if self.transform == "log":
            return np.clip(u, -700.0, 700.0)  # exp() stays finite and positive
        return u

    # ------------------------------------------------------------------ actions

    def factor(
        self,
        theta: NDArray[np.float64],
        rows: slice,
        context: ComponentContext,
        correlation: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return member's Schur factor on the gain's ``rows``, broadcastable to them.

        ``theta`` is the member's natural values ``(size,)``; ``correlation`` is the
        state rows' correlations with the data ``(rows, n_data)``, when
        :attr:`needs_correlation`.
        """
        raise NotImplementedError

    def hyper_taper(
        self,
        theta: NDArray[np.float64],
        correlation: NDArray[np.float64],
        context: ComponentContext,
        n_ensemble: int,
    ) -> NDArray[np.float64]:
        """Return the taper of this component's rows of the hyper-parameter update.

        ``correlation`` is the correlations of its hyper-parameters with the new forecast
        ``(size, n_data)``, over the current ``n_ensemble`` members; the default keeps those
        above ``threshold / sqrt(n_ensemble)``.
        """
        cutoff = min(1.0, self.threshold / math.sqrt(n_ensemble))
        return (np.abs(correlation) > cutoff).astype(np.float64)


@chop_components.register("length_scale")
class LengthScale(ChopComponent):
    """P-AutoAdaLoc: per-member Gaspari-Cohn length scales of a correlation-based taper.

    ``T_j[k, s] = f_GC((1 - |rho[k, s]|) / (factor * l_j[s]))``, ``rho`` the correlations
    of state row ``k`` with datum ``s``: of the initial ensemble (``correlation =
    "prior"``, the paper's rho^0) or of the current one (``"current"``, the published
    code's, recomputed before each update). ``mode = "data"`` has one length scale per
    datum (P-AutoAdaLoc-0), ``"scalar"`` one per member (P-AutoAdaLoc-1). The published
    code's taper is ``factor = 2`` (its ``2 (l + 1e-6)``), the default (the user, 2026-10-11:
    with the paper's formula, 1, and its prior, the taper zeroed most of the gain on 5Spots).

    The hyper-update taper is the same rule on the correlations of the length scales with
    the new forecast, column ``s`` scaled by ``l_j[s]`` (the code's ``update_hyper``).

    Parameters
    ----------
    mode
        ``data`` (default) gives each member one length scale per datum, ``scalar`` one
        length scale for all data.
    correlation
        The correlations the taper is computed from: ``prior`` (default) those of the
        initial ensemble, ``current`` those recomputed before each update.
    factor
        The multiplier of the length scales in the taper (positive; default 2, the
        published code's; 1 is the paper's formula).
    low, high
        The range of the uniform prior each member's length scales is drawn from (defaults
        0.23 and 0.43).
    transform
        The scale the hyper-parameter update works on: ``none`` (natural values, floored
        at ``floor``), ``log`` (needs ``low > 0``) or ``logit`` (kept within ``lower`` and
        ``upper``).
    floor
        The smallest value kept with ``transform = "none"`` (default 1e-6).
    lower, upper
        ``logit``'s bounds, with ``lower < low <= high < upper``; by default ``low`` and
        ``high`` widened by their range, ``lower`` never below 0.
    """

    needs_correlation = True

    def __init__(
        self,
        mode: Literal["data", "scalar"] = "data",
        correlation: Literal["prior", "current"] = "prior",
        factor: float = 2.0,
        low: float = 0.23,
        high: float = 0.43,
        transform: Transform = "none",
        floor: float = 1e-6,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        if mode not in ("data", "scalar"):
            raise ValueError(f"mode must be data or scalar, got {mode!r}")
        if correlation not in ("prior", "current"):
            raise ValueError(f"correlation must be prior or current, got {correlation!r}")
        if factor <= 0:
            raise ValueError(f"factor must be positive, got {factor}")
        if low <= 0:
            raise ValueError(f"length scales must be positive: low = {low}")
        super().__init__(low, high, transform, floor, lower, upper)
        self.mode = mode
        self.correlation = correlation
        self.scale_factor = float(factor)

    def size(self, context: ComponentContext) -> int:
        """One length scale per datum, or one per member."""
        return context.n_data if self.mode == "data" else 1

    def _scales(self, theta: NDArray[np.float64], n_data: int) -> NDArray[np.float64]:
        # Guards a zero floor; the transforms keep the scales positive otherwise.
        scales = self.scale_factor * np.maximum(theta, 1e-12)
        return np.broadcast_to(scales, (n_data,)) if scales.size == 1 else scales

    def factor(
        self,
        theta: NDArray[np.float64],
        rows: slice,
        context: ComponentContext,
        correlation: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return ``f_GC((1 - |rho|) / (factor * l_j))`` on the rows, ``(rows, n_data)``."""
        if correlation is None:
            raise ValueError("length_scale needs the state's correlations with the data")
        scales = self._scales(theta, context.n_data)
        return gaspari_cohn((1.0 - np.abs(correlation)) / scales[None, :])

    def hyper_taper(
        self,
        theta: NDArray[np.float64],
        correlation: NDArray[np.float64],
        context: ComponentContext,
        n_ensemble: int,
    ) -> NDArray[np.float64]:
        """Return the same rule on the length scales' correlations with the new forecast."""
        scales = self._scales(theta, context.n_data)
        return gaspari_cohn((1.0 - np.abs(correlation)) / scales[None, :])


@chop_components.register("gain_inflation")
class GainInflation(ChopComponent):
    """Per-member multiplicative inflation of the gain: ``dm_j = (D_j o K) r_j``.

    ``mode = "variable"``: one factor per state variable (``D_j = delta_j[var(k)]``, a
    vector inflation); ``"scalar"``: one per member; ``"variable_datatype"``: one per
    state variable and data type (a block matrix inflation, ``D_j[k, s] =
    delta_j[var(k), type(s)]``). A factor 1 leaves the gain as it is.

    Parameters
    ----------
    mode
        ``variable`` (default) gives each member one inflation factor per state variable,
        ``scalar`` one factor, ``variable_datatype`` one per state variable and data type.
    low, high
        The range of the uniform prior each member's inflation factors is drawn from (defaults
        0.5 and 1.5).
    transform
        The scale the hyper-parameter update works on: ``none`` (natural values, floored
        at ``floor``), ``log`` (needs ``low > 0``) or ``logit`` (kept within ``lower`` and
        ``upper``).
    floor
        The smallest value kept with ``transform = "none"`` (default 1e-6).
    lower, upper
        ``logit``'s bounds, with ``lower < low <= high < upper``; by default ``low`` and
        ``high`` widened by their range, ``lower`` never below 0.
    threshold
        The hyper-update keeps a hyper-parameter's correlation with a datum only when
        ``abs(corr) > threshold / sqrt(N)``, ``N`` the ensemble size (default 3).
    """

    def __init__(
        self,
        mode: Literal["variable", "scalar", "variable_datatype"] = "variable",
        low: float = 0.5,
        high: float = 1.5,
        transform: Transform = "none",
        floor: float = 1e-6,
        lower: float | None = None,
        upper: float | None = None,
        threshold: float = 3.0,
    ) -> None:
        if mode not in ("variable", "scalar", "variable_datatype"):
            raise ValueError(f"mode must be variable, scalar or variable_datatype, got {mode!r}")
        super().__init__(low, high, transform, floor, lower, upper, threshold)
        self.mode = mode

    def size(self, context: ComponentContext) -> int:
        """One factor per variable, per member, or per variable and data type."""
        n_variables = len(context.block_sizes)
        if self.mode == "scalar":
            return 1
        if self.mode == "variable":
            return n_variables
        return n_variables * len(context.type_of_data()[0])

    def factor(
        self,
        theta: NDArray[np.float64],
        rows: slice,
        context: ComponentContext,
        correlation: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return the rows' factors: ``(rows, 1)``, or ``(rows, n_data)`` per data type."""
        if self.mode == "scalar":
            return np.full((1, 1), float(theta[0]))
        variables = context.variable_of_rows()[rows]
        if self.mode == "variable":
            per_row: NDArray[np.float64] = theta[variables][:, None]
            return per_row
        names, types = context.type_of_data()
        table = theta.reshape(len(context.block_sizes), len(names))
        block: NDArray[np.float64] = table[variables][:, types]
        return block


@chop_components.register("data_weights")
class DataWeights(ChopComponent):
    """Per-member weights of the innovations in the gain: ``dm_j = K diag(w_j) r_j``.

    A Schur factor ``1 w_j^T``, a data-error inflation within the gain only: ``mode =
    "datatype"`` has one weight per data type, ``"datum"`` one per datum.

    Parameters
    ----------
    mode
        ``datatype`` (default) gives each member one weight per data type, ``datum`` one
        per datum.
    low, high
        The range of the uniform prior each member's weights is drawn from (defaults
        0.5 and 1.5).
    transform
        The scale the hyper-parameter update works on: ``none`` (natural values, floored
        at ``floor``), ``log`` (needs ``low > 0``) or ``logit`` (kept within ``lower`` and
        ``upper``).
    floor
        The smallest value kept with ``transform = "none"`` (default 1e-6).
    lower, upper
        ``logit``'s bounds, with ``lower < low <= high < upper``; by default ``low`` and
        ``high`` widened by their range, ``lower`` never below 0.
    threshold
        The hyper-update keeps a hyper-parameter's correlation with a datum only when
        ``abs(corr) > threshold / sqrt(N)``, ``N`` the ensemble size (default 3).
    """

    def __init__(
        self,
        mode: Literal["datatype", "datum"] = "datatype",
        low: float = 0.5,
        high: float = 1.5,
        transform: Transform = "none",
        floor: float = 1e-6,
        lower: float | None = None,
        upper: float | None = None,
        threshold: float = 3.0,
    ) -> None:
        if mode not in ("datatype", "datum"):
            raise ValueError(f"mode must be datatype or datum, got {mode!r}")
        super().__init__(low, high, transform, floor, lower, upper, threshold)
        self.mode = mode

    def size(self, context: ComponentContext) -> int:
        """One weight per data type, or per datum."""
        return len(context.type_of_data()[0]) if self.mode == "datatype" else context.n_data

    def factor(
        self,
        theta: NDArray[np.float64],
        rows: slice,
        context: ComponentContext,
        correlation: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Return the data's weights, ``(1, n_data)``."""
        if self.mode == "datum":
            return theta[None, :]
        _, types = context.type_of_data()
        weights: NDArray[np.float64] = theta[types][None, :]
        return weights


def stack_sizes(components: Sequence[ChopComponent], context: ComponentContext) -> list[slice]:
    """Return each component's rows in the stacked hyper-parameter vector."""
    slices, start = [], 0
    for component in components:
        size = component.size(context)
        slices.append(slice(start, start + size))
        start += size
    return slices
