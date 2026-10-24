"""Performance-metric plugins: normalized data mismatch, model RMSE and model SSIM.

Definitions follow PET-fork (``not public`` @ ``1348861``,
``Plotting/metric_analyses.py``), as used in its Brugge study:

- ``normalized_data_mismatch`` (``calculate_data_mismatch``):
  ``mean(((d_obs - y_j) / sigma)^2)`` per
  member (a mean square, as PET-fork; 1 means a fit at the noise level), over all data
  and per data type; real and synthetic cases. It is computed in the analysis data space,
  i.e. after data transforms such as wavelet compression. Differences: the per-type value
  is the mean over that type's data; PET-fork sums the type's per-time values and divides
  by the number of report times (the same when a type is observed once per time), and
  weights a vector datum by one standard deviation (pet_parity.md).
- ``rmse`` (``calculate_rmse_and_model_change``): ``||m_true - m_j||_2 / sqrt(n)`` per
  member, per state variable over its active cells and over all of them together
  (``all``); synthetic cases only.
- ``model_change`` (``calculate_rmse_and_model_change``, its model-change part):
  ``||m_j^(k) - m_j^(k-1)||_2 / sqrt(n)`` per member, between the scored ensemble and the one
  it was updated from, per state variable over its active cells and over all of them
  (``all``); from the first update on, in real and synthetic cases. PET-fork compares
  consecutive saved ensembles; APEX compares a rejected trial with the iterate it was
  proposed from.
- ``ssim`` (``calculate_ssim``): mean SSIM of each member's field to the true one, per
  state variable; synthetic cases only. Inactive cells are filled with the mean of the
  active values (the truth's own; for the ensemble, over all members), and ``data_range``
  spans the truth and the whole ensemble, as PET-fork does, so a member's SSIM depends
  on the ensemble's range. APEX additions: ``per_layer`` (2-D SSIM per layer, averaged,
  for grids too thin for the window), ``win_size`` and a fixed ``data_range``.
"""

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.metrics.kernels import normalized_l2, ssim
from apex.core.metrics import MetricContext
from apex.core.options import TextList
from apex.core.plugins import metrics


@metrics.register("normalized_data_mismatch")
class NormalizedDataMismatch:
    """Normalized data mismatch ``mean(((d_obs - y_j) / sigma)^2)`` against the observed values.

    The observations are the unperturbed ones, in the analysis data space. Labels: ``all``
    and each data type. 1 means a fit at the noise level.
    """

    requires_truth = False

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{"all": (n,), <datatype>: (n,), ...}``."""
        observations = context.observations
        sigma = np.sqrt(observations.variances)
        squared = ((observations.values[:, None] - context.predicted) / sigma[:, None]) ** 2
        result = {"all": squared.mean(axis=0)}
        rows: dict[str, list[int]] = {}
        for i, key in enumerate(observations.keys):
            rows.setdefault(key.datatype, []).append(i)
        for datatype, r in rows.items():
            result[datatype] = squared[r].mean(axis=0)
        return result


class _TruthMetric:
    """Shared option handling of metrics against the true model."""

    requires_truth = True

    def __init__(self, variables: Sequence[str] | None, actnum: ArrayLike | None) -> None:
        self.variables = None if variables is None else tuple(variables)
        self.active = None if actnum is None else np.asarray(actnum).reshape(-1) != 0
        if self.active is not None and not self.active.any():
            raise ValueError("actnum has no active cells")

    def _selected(self, context: MetricContext, default: Sequence[str] | None = None) -> list[str]:
        """Return the variables to score: ``variables``, else ``default``, else all.

        Raises
        ------
        ValueError
            Without a truth, for unknown or empty selections, or when the truth does not
            match a variable's size (or ``actnum`` a named variable's).
        """
        if context.truth is None:
            raise ValueError("this metric needs the true model (a synthetic case study)")
        if self.variables is not None:
            names = list(self.variables)
        elif default is not None:
            names = list(default)
        else:  # every quantity: not codes (facies) nor model-error parameters
            names = [n for n in context.state if n not in context.categorical | context.auxiliary]
        unknown = [n for n in names if n not in context.state]
        if unknown:
            raise ValueError(f"unknown state variables {', '.join(unknown)}")
        if not names:
            if self.variables is None and default is None:
                return []  # only codes (facies): nothing for a quantity metric to score
            raise ValueError("no state variable to score")
        for name in names:
            truth = context.truth.get(name)
            size = context.state[name].shape[0]
            if truth is None or truth.shape != (size,):
                raise ValueError(f"the true model has no {size} values of {name!r}")
            if self.active is not None and self.variables is not None and self.active.size != size:
                raise ValueError(
                    f"actnum has {self.active.size} cells but {name!r} has {size} values"
                )
        return names

    def _mask(self, size: int) -> NDArray[np.bool_] | None:
        """``actnum`` for a variable of ``size`` values; ``None`` (all) for other sizes."""
        return self.active if self.active is not None and self.active.size == size else None


@metrics.register("rmse")
class ModelRMSE(_TruthMetric):
    """RMSE of each member's parameters against the true model.

    Parameters
    ----------
    variables
        State variables to score; default all.
    actnum
        Active-cell indicator (non-zero = active); default all cells. It applies to the
        variables with as many values (e.g. the grid fields, not scalar parameters); named
        ``variables`` must all match it. Configurations give ``actnum_file``.

    Labels: each variable, and ``all`` (their active values concatenated, as PET-fork).
    """

    def __init__(self, variables: TextList | None = None, actnum: ArrayLike | None = None) -> None:
        super().__init__(variables, actnum)

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{<variable>: (n,), ..., "all": (n,)}``."""
        names = self._selected(context)
        assert context.truth is not None
        if not names:
            return {}
        result: dict[str, NDArray[np.float64]] = {}
        truths, members = [], []
        for name in names:
            truth, ensemble = context.truth[name], context.state[name]
            mask = self._mask(truth.size)
            if mask is not None:
                truth, ensemble = truth[mask], ensemble[mask]
            result[name] = normalized_l2(truth, ensemble)
            truths.append(truth)
            members.append(ensemble)
        result["all"] = normalized_l2(np.concatenate(truths), np.concatenate(members))
        return result


@metrics.register("facies_mismatch")
class FaciesMismatch(_TruthMetric):
    """Fraction of cells whose facies differs from the true model's.

    Facies are integer codes, as a parameterization produces them (``pgs_truncation``'s
    ``facies``); values are rounded to the nearest code before comparing.

    Parameters
    ----------
    variables
        Facies properties to score; default ``["facies"]``.
    actnum
        Active-cell indicator (non-zero = active); default all cells. Configurations give
        ``actnum_file``.

    Labels: each variable, in ``[0, 1]`` per member.
    """

    def __init__(self, variables: TextList | None = None, actnum: ArrayLike | None = None) -> None:
        super().__init__(variables, actnum)

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{<variable>: (n,)}``, the misclassified fraction of each member."""
        names = self._selected(context, ["facies"])
        assert context.truth is not None
        result: dict[str, NDArray[np.float64]] = {}
        for name in names:
            truth, ensemble = np.rint(context.truth[name]), np.rint(context.state[name])
            mask = self._mask(truth.size)
            if mask is not None:
                truth, ensemble = truth[mask], ensemble[mask]
            result[name] = (ensemble != truth[:, None]).mean(axis=0)
        return result


def _facies_counts(
    context: MetricContext, name: str, active: NDArray[np.bool_] | None
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    """Return the ensemble's facies proportions per cell ``(n_cells, K)`` and the true codes."""
    assert context.truth is not None
    truth = np.rint(context.truth[name]).astype(np.int64)
    ensemble = np.rint(context.state[name]).astype(np.int64)
    if active is not None and active.size == truth.size:
        truth, ensemble = truth[active], ensemble[active]
    if truth.min() < 0 or ensemble.min() < 0:
        raise ValueError(f"{name}: facies codes must be non-negative integers")
    n_facies = int(max(truth.max(), ensemble.max())) + 1
    counts = np.stack([(ensemble == k).sum(axis=-1) for k in range(n_facies)], axis=-1)
    return counts.astype(np.float64), truth, n_facies


def _brier(probabilities: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
    """Mean over cells of sum_k (p_k - 1[truth = k])^2, for ``(n_cells, K)`` probabilities."""
    outcome = np.zeros_like(probabilities)
    outcome[np.arange(truth.size), truth] = 1.0
    return float(np.mean(np.sum((probabilities - outcome) ** 2, axis=-1)))


class _FaciesScore(_TruthMetric):
    """An ensemble score of facies against the truth, with constant baselines.

    The score is of the ensemble's facies probabilities per cell (each facies' share of
    the members there), so it is one value per ensemble; metrics are per member, so it is
    repeated for every member (means and summaries are the value itself).
    """

    def __init__(self, variables: TextList | None = None, actnum: ArrayLike | None = None) -> None:
        super().__init__(variables, actnum)

    def _score(self, probabilities: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
        raise NotImplementedError

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return the score, and the baselines' scores, each repeated over the members."""
        names = self._selected(context, ["facies"])
        n_members = next(iter(context.state.values())).shape[-1]
        result: dict[str, NDArray[np.float64]] = {}
        for name in names:
            counts, truth, n_facies = _facies_counts(context, name, self.active)
            probabilities = counts / counts.sum(axis=-1, keepdims=True)
            scores = {name: self._score(probabilities, truth)}
            # Baselines: one facies everywhere, and the ensemble's overall proportions
            # everywhere (no spatial information): a score no better than these has learnt
            # nothing about where the facies are.
            for k in range(n_facies):
                constant = np.zeros_like(probabilities)
                constant[:, k] = 1.0
                scores[f"{name}:all_{k}"] = self._score(constant, truth)
            overall = np.broadcast_to(probabilities.mean(axis=0), probabilities.shape)
            scores[f"{name}:proportions"] = self._score(np.array(overall), truth)
            result.update({label: np.full(n_members, value) for label, value in scores.items()})
        return result


@metrics.register("facies_brier")
class FaciesBrier(_FaciesScore):
    """Brier score of the ensemble's facies probabilities (Brier, 1950).

    ``mean over cells of sum_k (p_k - 1[truth = k])^2``, with ``p_k`` the share of members
    with facies ``k`` in the cell: 0 is perfect, lower is better. Labels: ``<variable>``
    (default ``facies``); the baselines ``<variable>:all_<k>`` (facies ``k`` everywhere,
    ``2 x`` the fraction of cells that are not ``k``) and ``<variable>:proportions`` (the
    ensemble's overall proportions in every cell); and ``<variable>:skill``, the Brier skill
    score ``1 - score / proportions`` (positive: better than knowing only the proportions).
    """

    def _score(self, probabilities: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
        return _brier(probabilities, truth)

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return the Brier score, its baselines and its skill, repeated over the members."""
        result = super().evaluate(context)
        for label in [k for k in result if ":" not in k]:
            reference = result[f"{label}:proportions"]
            with np.errstate(divide="ignore", invalid="ignore"):
                result[f"{label}:skill"] = 1.0 - result[label] / reference
        return result


@metrics.register("facies_log_score")
class FaciesLogScore(_FaciesScore):
    """Logarithmic score of the ensemble's facies probabilities (Good, 1952).

    ``mean over cells of -ln p_truth``, the probability the ensemble gives the true facies,
    smoothed as ``(count + 0.5) / (N + 0.5 K)`` (Jeffreys) so that a facies no member has
    scores finitely: lower is better. Labels and baselines as :class:`FaciesBrier` (the
    one-facies baselines are smoothed with N = 1, i.e. probabilities 0.5/(1 + 0.5 K) and
    1.5/(1 + 0.5 K)).
    """

    #: Jeffreys' additive smoothing: half a member per facies.
    SMOOTHING = 0.5

    def _score(self, probabilities: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
        n_facies = probabilities.shape[-1]
        smoothed = (probabilities + self.SMOOTHING) / (1.0 + self.SMOOTHING * n_facies)
        return float(np.mean(-np.log(smoothed[np.arange(truth.size), truth])))

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return the log score and its baselines, repeated over the members."""
        names = self._selected(context, ["facies"])
        n_members = next(iter(context.state.values())).shape[-1]
        result: dict[str, NDArray[np.float64]] = {}
        for name in names:
            counts, truth, n_facies = _facies_counts(context, name, self.active)
            total = counts.sum(axis=-1, keepdims=True)
            smoothed = (counts + self.SMOOTHING) / (total + self.SMOOTHING * n_facies)
            scores = {name: float(np.mean(-np.log(smoothed[np.arange(truth.size), truth])))}
            for k in range(n_facies):
                constant = np.zeros_like(counts)
                constant[:, k] = 1.0
                scores[f"{name}:all_{k}"] = self._score(constant, truth)
            # The overall proportions, smoothed alike (as mean counts of the N members).
            overall = (counts.mean(axis=0) + self.SMOOTHING) / (
                total[0] + self.SMOOTHING * n_facies
            )
            scores[f"{name}:proportions"] = float(np.mean(-np.log(overall[truth])))
            result.update({label: np.full(n_members, value) for label, value in scores.items()})
        return result


@metrics.register("model_change")
class ModelChange:
    """Change of each member's parameters since the ensemble it was updated from.

    Parameters
    ----------
    variables
        State variables to score; default all.
    actnum
        Active-cell indicator (non-zero = active), as for :class:`ModelRMSE`.

    Labels: each variable, and ``all`` (their active values concatenated). Not defined for
    the prior (``{}``).
    """

    requires_truth = False
    requires_previous = True

    def __init__(self, variables: TextList | None = None, actnum: ArrayLike | None = None) -> None:
        self.variables = None if variables is None else tuple(variables)
        self.active = None if actnum is None else np.asarray(actnum).reshape(-1) != 0
        if self.active is not None and not self.active.any():
            raise ValueError("actnum has no active cells")

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{<variable>: (n,), ..., "all": (n,)}``, or ``{}`` for the prior.

        Raises
        ------
        ValueError
            For unknown or empty selections, a previous ensemble of another shape, or a
            named variable of another size than ``actnum`` (unnamed variables of another
            size are scored on all their values, as in :class:`ModelRMSE`).
        """
        names = (
            [n for n in context.state if n not in context.categorical | context.auxiliary]
            if self.variables is None
            else list(self.variables)
        )
        unknown = [n for n in names if n not in context.state]
        if unknown:
            raise ValueError(f"unknown state variables {', '.join(unknown)}")
        if not names:
            if self.variables is None:
                return {}  # only codes (facies): no quantity to score
            raise ValueError("no state variable to score")
        for name in names:
            size = context.state[name].shape[0]
            if self.active is not None and self.variables is not None and self.active.size != size:
                raise ValueError(
                    f"actnum has {self.active.size} cells but {name!r} has {size} values"
                )
        previous = context.previous_state
        if previous is None:
            return {}
        result: dict[str, NDArray[np.float64]] = {}
        before, after = [], []
        for name in names:
            old, new = previous.get(name), context.state[name]
            if old is None or old.shape != new.shape:
                raise ValueError(f"the previous ensemble has no {new.shape} values of {name!r}")
            mask = (
                self.active
                if self.active is not None and self.active.size == new.shape[0]
                else None
            )
            if mask is not None:
                old, new = old[mask], new[mask]
            result[name] = _member_l2(old, new)
            before.append(old)
            after.append(new)
        result["all"] = _member_l2(np.concatenate(before), np.concatenate(after))
        return result


def _member_l2(old: NDArray[np.float64], new: NDArray[np.float64]) -> NDArray[np.float64]:
    """``||old_j - new_j||_2 / sqrt(n)`` of every member ``j`` (PET-fork ``normalized_l2``)."""
    change: NDArray[np.float64] = np.linalg.norm(old - new, axis=0) / math.sqrt(old.shape[0])
    return change


@metrics.register("ssim")
class ModelSSIM(_TruthMetric):
    """Structural similarity of each member's fields to the true model's.

    Parameters
    ----------
    grid
        Field shape ``[nx, ny, nz]`` (or 1-2 dimensions), including inactive cells; default
        each variable's own ``grid`` (``[[state.variables]]``).
    variables
        State variables to score; default every variable with a grid (or, with ``grid``,
        every variable with ``prod(grid)`` values).
    actnum
        Active-cell indicator over the grid; inactive cells are filled with the active
        mean. Configurations give ``actnum_file``.
    order
        Cell order of the state vectors, ``"F"`` (Eclipse, x fastest; default) or ``"C"``.
    win_size
        Odd SSIM window (>= 3), at most every (per-layer: every horizontal) grid dimension.
    per_layer
        Average 2-D SSIMs over the layers (last grid axis) instead of one 3-D SSIM.
    data_range
        Fixed value range; default the range of the truth and the ensemble together.

    Labels: each variable.
    """

    def __init__(
        self,
        grid: Sequence[int] | None = None,
        variables: TextList | None = None,
        actnum: ArrayLike | None = None,
        order: Literal["F", "C"] = "F",
        win_size: int = 7,
        per_layer: bool = False,
        data_range: float | None = None,
    ) -> None:
        super().__init__(variables, actnum)
        if order not in ("F", "C"):
            raise ValueError(f"order must be 'F' or 'C', got {order!r}")
        if data_range is not None and not data_range > 0:
            raise ValueError(f"data_range must be positive, got {data_range}")
        if win_size < 3 or win_size % 2 == 0:
            raise ValueError(f"win_size must be odd and >= 3, got {win_size}")
        self.order: Literal["F", "C"] = order
        self.win_size = int(win_size)
        self.per_layer = bool(per_layer)
        self.data_range = data_range
        self.grid = None if grid is None else tuple(int(n) for n in grid)
        if self.grid is not None:
            self._check_grid(self.grid)

    def _check_grid(self, grid: tuple[int, ...]) -> None:
        if not 1 <= len(grid) <= 3 or any(n < 1 for n in grid):
            raise ValueError(f"grid must have 1 to 3 positive dimensions, got {grid}")
        if self.per_layer and len(grid) < 2:
            raise ValueError("per_layer needs a grid of 2 or 3 dimensions")
        horizontal = grid[:-1] if self.per_layer else grid
        if any(n < self.win_size for n in horizontal):
            raise ValueError(
                f"win_size {self.win_size} does not fit the grid {horizontal}"
                + ("" if self.per_layer else "; per_layer = true uses 2-D windows per layer")
            )

    def _grids(self, context: MetricContext) -> dict[str, tuple[int, ...]]:
        """Return the grid of each variable to score."""
        if self.grid is not None:
            n_cells = math.prod(self.grid)
            fitting = [
                n
                for n, block in context.state.items()
                if block.shape[0] == n_cells and n not in context.categorical | context.auxiliary
            ]
            if not fitting and self.variables is None:
                raise ValueError(f"no state variable has {n_cells} cells")
            return dict.fromkeys(self._selected(context, fitting), self.grid)
        names = self._selected(
            context,
            [
                n
                for n in context.state
                if n in context.grids and n not in context.categorical | context.auxiliary
            ],
        )
        missing = [n for n in names if n not in context.grids]
        if missing:
            raise ValueError(
                f"no grid for {', '.join(missing)}; give [[state.variables]] grid or the "
                "metric's grid option"
            )
        grids = {n: tuple(context.grids[n]) for n in names}
        for grid in set(grids.values()):
            self._check_grid(grid)
        return grids

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{<variable>: (n,), ...}``."""
        grids = self._grids(context)
        assert context.truth is not None
        result = {}
        for name, grid in grids.items():
            n_cells = math.prod(grid)
            if self.active is not None and self.active.size != n_cells:
                raise ValueError(f"actnum has {self.active.size} cells, the grid {n_cells}")
            active = np.ones(n_cells, dtype=bool) if self.active is None else self.active
            truth = context.truth[name].astype(np.float64)
            ensemble = context.state[name].astype(np.float64)
            if truth.size != n_cells:
                raise ValueError(f"{name!r} has {truth.size} values, the grid {n_cells} cells")
            truth, ensemble = truth.copy(), ensemble.copy()
            truth[~active] = truth[active].mean()
            ensemble[~active] = ensemble[active].mean()
            data_range = self.data_range or float(
                max(truth.max(), ensemble.max()) - min(truth.min(), ensemble.min())
            )
            if not data_range > 0:
                raise ValueError(f"{name!r}: the truth and ensemble are constant (no range)")
            field = truth.reshape(grid, order=self.order)
            members = ensemble.reshape(*grid, ensemble.shape[-1], order=self.order)
            if self.per_layer:
                layers = [
                    ssim(field[..., k], members[..., k, :], data_range, self.win_size)
                    for k in range(grid[-1])
                ]
                result[name] = np.mean(layers, axis=0)
            else:
                result[name] = ssim(field, members, data_range, self.win_size)
        return result
