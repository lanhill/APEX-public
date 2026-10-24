"""Plurigaussian simulation: the ``gstools_pgs`` prior and ``pgs_truncation``.

The prior draws the latent Gaussian fields of a plurigaussian simulation with GSTools
(Müller et al., 2022; the ``facies`` extra), conditioned to facies at wells by APEX's
Gibbs sampler (:func:`~apex.algorithms.facies.rules.condition_latent`) and GSTools'
``CondSRF``. The methods update the latent fields; ``pgs_truncation`` maps them, by the
same rule, to facies and to each facies' properties before every forecast (Liu & Oliver,
2005; Sebacher, Hanea & Heemink, 2013).

Grids are Eclipse ``[nx, ny, nz]`` with ``i`` fastest; GSTools' structured fields are
indexed ``[i, j(, k)]`` and are flattened in Fortran order. Distances are in cells times
``cell_size``.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.facies.distance import FaciesProperties
from apex.algorithms.facies.rules import LithotypeRule, condition_latent
from apex.core.options import TEXT, InputFile, mapping
from apex.core.plugins import parameterizations, priors
from apex.domain.state import StateVariable

#: GSTools covariance models a field may name.
MODELS = ("Gaussian", "Exponential", "Spherical", "Matern", "Stable", "Cubic", "Linear")

#: What a field's and the conditioning's strings are: model names, data.
FIELD_ROLES = mapping(model=TEXT)
CONDITIONING_ROLES = mapping()


def load_rule(
    rule_file: str | Path | None, proportions: Sequence[float] | None, extent: float
) -> LithotypeRule:
    """Return the rule: a lithotype array (``rule_file``) or a truncated Gaussian.

    Raises
    ------
    ValueError
        Unless exactly one of ``rule_file`` and ``proportions`` is given.
    """
    if (rule_file is None) == (proportions is None):
        raise ValueError("give rule_file (a lithotype array) or proportions, not both")
    if proportions is not None:
        return LithotypeRule.from_proportions(proportions)
    codes = np.load(Path(rule_file))  # type: ignore[arg-type]
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError(f"{rule_file} must hold integer facies codes, got {codes.dtype}")
    return LithotypeRule.on_box(codes, extent)


def _gstools() -> Any:  # noqa: ANN401 (an optional module)
    try:
        import gstools
    except ImportError as error:  # pragma: no cover (the extra is installed in tests)
        raise ValueError(
            "gstools_pgs needs the 'facies' extra (pip install apex[facies])"
        ) from error
    return gstools


@priors.register("gstools_pgs")
class PGSPrior:
    """The latent fields of a plurigaussian simulation, optionally conditioned to well facies.

    Parameters
    ----------
    fields
        One covariance model per latent field, in the order of the group's variables:
        ``{model = "Gaussian", len_scale = 10 or [10, 5(, 2)], angles = 0 or [...]}``
        (GSTools' ``CovModel`` arguments; variance 1). ``model`` is one of :data:`MODELS`.
    rule_file, proportions, extent
        The truncation rule, as for ``pgs_truncation`` (the composition checks they match).
    conditioning
        ``{cells = [[i, j, k], ...], facies = [code, ...]}``, 1-based Eclipse cells and the
        facies observed there; every realisation honours them.
    cell_size
        ``[dx, dy, dz]``, the cell size in the units of ``len_scale``; default 1.
    gibbs_sweeps
        Iterations of the conditioning sampler (exact HMC and a coordinate sweep per field;
        :func:`~apex.algorithms.facies.rules.condition_latent`), default 100.
    """

    #: Latent fields are standard Gaussian; there is nothing to clip.
    clip = False

    def __init__(
        self,
        fields: Annotated[Sequence[Mapping[str, Any]], FIELD_ROLES],
        rule_file: InputFile | None = None,
        proportions: Sequence[float] | None = None,
        extent: float = 3.0,
        conditioning: Annotated[Mapping[str, Sequence[Any]] | None, CONDITIONING_ROLES] = None,
        cell_size: Sequence[float] = (1.0, 1.0, 1.0),
        gibbs_sweeps: int = 100,
    ) -> None:
        self.rule = load_rule(rule_file, proportions, extent)
        if len(fields) != self.rule.n_fields:
            raise ValueError(
                f"the rule has {self.rule.n_fields} axes but {len(fields)} fields are given"
            )
        for field in fields:
            unknown = set(field) - {"model", "len_scale", "angles"}
            if unknown or field.get("model") not in MODELS or "len_scale" not in field:
                raise ValueError(
                    f"each field needs model (one of {', '.join(MODELS)}) and len_scale, and "
                    f"may give angles; got {dict(field)}"
                )
        self.fields = [dict(f) for f in fields]
        self.cell_size = tuple(float(s) for s in cell_size)
        if len(self.cell_size) != 3 or min(self.cell_size) <= 0:
            raise ValueError(f"cell_size must be [dx, dy, dz] > 0, got {cell_size}")
        if gibbs_sweeps < 1:
            raise ValueError("gibbs_sweeps must be at least 1")
        self.gibbs_sweeps = int(gibbs_sweeps)
        self.cells: NDArray[np.int64] | None = None
        self.observed: NDArray[np.int64] | None = None
        if conditioning is not None and len(conditioning.get("cells", [])) > 0:
            cells = np.asarray(conditioning.get("cells", []), dtype=np.int64)
            observed = np.asarray(conditioning.get("facies", []), dtype=np.int64)
            if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] != observed.size:
                raise ValueError("conditioning needs cells [[i, j, k], ...] and one facies each")
            # One value per cell: a cell listed twice (two wells in it) would make the
            # covariance singular; listed with two facies, it contradicts itself.
            unique: dict[tuple[int, ...], int] = {}
            for cell, code in zip(map(tuple, cells.tolist()), observed.tolist(), strict=True):
                if unique.setdefault(cell, code) != code:
                    raise ValueError(
                        f"conditioning cell {list(cell)} is given facies {unique[cell]} and {code}"
                    )
            self.cells = np.array(list(unique), dtype=np.int64).reshape(-1, 3)
            self.observed = np.array(list(unique.values()), dtype=np.int64)

    def generate_joint(
        self,
        variables: Sequence[StateVariable],
        n_ensemble: int | None,
        rng: np.random.Generator,
    ) -> dict[str, NDArray[np.float64]]:
        """Return each latent field's realisations, ``{name: (n_cells, n_ensemble)}``.

        Raises
        ------
        ValueError
            If the variables are not one grid each of the same shape, or ``n_ensemble``
            is missing, or a conditioning cell is off the grid.
        """
        if n_ensemble is None:
            raise ValueError("gstools_pgs needs [prior] ensemble_size")
        if len(variables) != self.rule.n_fields:
            raise ValueError(f"gstools_pgs draws {self.rule.n_fields} fields, not {len(variables)}")
        grids = {v.grid for v in variables}
        if len(grids) != 1 or None in grids:
            raise ValueError("the latent fields must all be grid fields of the same [nx, ny, nz]")
        grid = tuple(int(n) for n in next(iter(grids)))  # type: ignore[union-attr]
        if any(v.size != math.prod(grid) for v in variables):
            raise ValueError(f"each latent field must have {math.prod(grid)} cells")
        gs = _gstools()
        dim = 2 if grid[2] == 1 else 3
        axes = [np.arange(n) * self.cell_size[d] for d, n in enumerate(grid[:dim])]
        models = [self._model(gs, f, dim) for f in self.fields]
        positions = self._positions(grid, dim)
        covariances = None
        if positions is not None:
            # Unit-variance covariance between the conditioning cells, per field.
            offsets = positions[:, :, None] - positions[:, None, :]  # (dim, n_c, n_c)
            covariances = [
                model.cov_spatial(offsets.reshape(dim, -1)).reshape(offsets.shape[1:])
                for model in models
            ]
        seeds = rng.integers(0, 2**31 - 1, size=(n_ensemble, len(models)))
        blocks = [np.empty((math.prod(grid), n_ensemble)) for _ in models]
        for j in range(n_ensemble):
            values = None
            if covariances is not None:
                assert self.observed is not None
                values = condition_latent(
                    self.rule, covariances, self.observed, rng, self.gibbs_sweeps
                )
            for d, model in enumerate(models):
                if values is None:
                    field = gs.SRF(model, seed=int(seeds[j, d])).structured(axes)
                else:
                    assert positions is not None
                    krige = gs.krige.Simple(model, cond_pos=positions, cond_val=values[d], mean=0.0)
                    conditioned = gs.CondSRF(krige)
                    conditioned.set_pos(axes, "structured")
                    field = conditioned(seed=int(seeds[j, d]))
                blocks[d][:, j] = np.asarray(field).reshape(grid[:dim]).ravel(order="F")
        return {v.name: np.ascontiguousarray(b) for v, b in zip(variables, blocks, strict=True)}

    def _model(self, gs: Any, field: Mapping[str, Any], dim: int) -> Any:  # noqa: ANN401
        options: dict[str, Any] = {"dim": dim, "var": 1.0, "len_scale": field["len_scale"]}
        if "angles" in field:
            options["angles"] = field["angles"]
        return getattr(gs, str(field["model"]))(**options)

    def _positions(self, grid: tuple[int, ...], dim: int) -> NDArray[np.float64] | None:
        """Return the conditioning cells' coordinates ``(dim, n_c)``, or ``None``."""
        if self.cells is None or self.cells.size == 0:
            return None
        for cell in self.cells:
            if not all(1 <= c <= n for c, n in zip(cell, grid, strict=True)):
                raise ValueError(f"conditioning cell {cell.tolist()} is outside the grid {grid}")
        zero_based = (self.cells - 1).astype(np.float64).T[:dim]
        scaled: NDArray[np.float64] = zero_based * np.asarray(self.cell_size[:dim])[:, None]
        return scaled


@parameterizations.register("pgs_truncation")
class PGSTruncation:
    """Facies from latent fields by a fixed truncation rule, and properties by facies.

    Parameters
    ----------
    fields
        The latent state variables, one per rule axis, in order.
    facies
        Facies names; facies code ``k`` is ``facies[k]``.
    rule_file, proportions, extent
        A lithotype array (``.npy`` of integer codes, one axis per field) laid over
        ``[-extent, extent]`` standard deviations on each axis (default 3; edge bins extend
        to infinity), or, for one field, the facies' target ``proportions`` (a truncated
        Gaussian).
    values
        ``{facies: {property: value}}``: constant properties per facies, e.g.
        ``{sand = {poro = 0.25, permx = 500}, shale = {poro = 0.05, permx = 1}}``.
    property_fields
        ``{property: {facies: state variable}}``: properties taken cell by cell from a
        state variable of the facies there, e.g. ``{permx = {sand = "permx_sand", ...}}``
        (updated with the latent fields). A property is in ``values`` or here, not both.

    Outputs: ``facies`` (the codes, as floats) and every property.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {}
    #: Outputs that are codes, not quantities: scored by ``facies_mismatch``, not RMSE.
    categorical: tuple[str, ...] = ("facies",)

    def __init__(
        self,
        fields: Annotated[Sequence[str], TEXT],
        facies: Annotated[Sequence[str], TEXT],
        rule_file: InputFile | None = None,
        proportions: Sequence[float] | None = None,
        extent: float = 3.0,
        values: Annotated[Mapping[str, Mapping[str, float]] | None, TEXT] = None,
        property_fields: Annotated[Mapping[str, Mapping[str, str]] | None, TEXT] = None,
    ) -> None:
        self.rule = load_rule(rule_file, proportions, extent)
        if len(fields) != self.rule.n_fields:
            raise ValueError(
                f"the rule has {self.rule.n_fields} axes but fields names {len(fields)}"
            )
        self.fields = tuple(fields)
        self.facies = tuple(facies)
        codes = set(self.rule.codes.ravel().tolist())
        if not codes <= set(range(len(self.facies))):
            raise ValueError(
                f"the rule uses codes {sorted(codes)}; facies names {len(self.facies)} (codes "
                f"0..{len(self.facies) - 1})"
            )
        self.mapping = FaciesProperties(self.facies, values, property_fields)
        self.values = self.mapping.values
        self.property_fields = self.mapping.property_fields
        self.variables: tuple[str, ...] = (*self.fields, *self.mapping.variables)
        self.outputs: tuple[str, ...] = ("facies", *self.mapping.names)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{"facies": codes, property: values}`` for one member, ``(n_cells,)`` each.

        Raises
        ------
        ValueError
            If a variable is missing or the sizes differ.
        """
        missing = [v for v in self.variables if v not in state]
        if missing:
            raise ValueError(f"pgs_truncation needs {', '.join(missing)}")
        latent = np.stack([np.asarray(state[f], dtype=np.float64) for f in self.fields])
        codes = self.rule.facies(latent)  # (n_cells,)
        return {"facies": codes.astype(np.float64), **self.mapping.of(codes, state)}
