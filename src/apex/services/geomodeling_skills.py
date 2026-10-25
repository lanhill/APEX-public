"""The geomodeler's skills: continuous, correlated and facies fields.

- ``continuous``: a Gaussian field per variable (ln PERMX, PORO, ...), around a mean with a
  standard deviation and a variogram, drawn by ``gaussian_field`` (a dense Cholesky factor
  per layer, up to ``DENSE_LAYER_CELLS`` cells) or ``geostatspy_sgsim`` (larger layers,
  the ``geostatspy`` extra).
- ``facies_pluri``: facies from one latent Gaussian field truncated at the facies'
  proportions (``gstools_pgs``, conditioned to wells' facies where given; the ``facies``
  extra), and each facies' property values (``pgs_truncation``).
- ``correlated_continuous`` (phase 3): correlated Gaussian fields with one variogram (ln
  PERMX and PORO, as 3dBox's ``prepare.py``), drawn now into a stored ensemble beside the
  configuration: the generators draw variables independently.
- ``facies_mps`` (phase 3): facies from a training image by MPSlib's SNESIM
  (``mpslib_snesim``), as one signed distance field per facies, mapped to the facies and
  their property values by ``facies_distance``.

Each skill checks its parameters against the configuration's facts, and a drawn sample
against its parameters (:meth:`qc`): code, not the model, says whether the prior is the
one asked for.

Hard data: ``continuous`` and ``correlated_continuous`` condition
their priors to the data in the facts' estimates (``condition``, on by default), by kriging
each realisation's residuals; code, not the model, passes the cells and values on. A
reference field's estimates carry no data: a prior from them shares its statistics only.
"""

import io
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apex.algorithms.facies.rules import LithotypeRule
from apex.algorithms.geostatistics.conditioning import LARGE_CORRECTION
from apex.core.geomodeling import GeomodelFacts, GeomodelingSkill, PriorDraft
from apex.core.plugins import geomodeling_skills

#: Layers up to this many cells get ``gaussian_field``: it factors a dense (cells x cells)
#: covariance per layer, 0.8 GB at 10^4 cells (algorithms/priors.py); larger ones SGSIM.
DENSE_LAYER_CELLS = 10_000
#: The largest |value| of a log property a facies may take: e^30 is beyond any property,
#: so a larger one is the value itself given by mistake (e.g. 500 mD for ln PERMX).
LOG_LIMIT = 30.0
#: Standard errors of the drawn statistic allowed before the QC's own tolerance.
SAMPLING_SIGMAS = 3.0


def effective_members(members: int, cells: int, correlated_cells: float) -> float:
    """Return a sample's effective size: members times the independent patches in a field.

    A correlated field of ``cells`` cells holds about ``cells / correlated_cells``
    independent values; never fewer than one per member.
    """
    return members * max(1.0, cells / max(correlated_cells, 1.0))


class FieldChoice(BaseModel):
    """One variable's Gaussian field, in state space (ln(value) for a log variable)."""

    model_config = ConfigDict(extra="forbid")

    variable: str
    mean: float
    std: float = Field(gt=0)
    variogram: Literal["spherical", "exponential", "gaussian"] = "spherical"
    major_range: float = Field(gt=0, description="Cells along the major axis.")
    minor_range: float | None = Field(default=None, gt=0, description="Cells across it.")
    azimuth: float = Field(
        default=0.0,
        ge=-180,
        le=180,
        description="The major axis' direction, degrees counter-clockwise from the i axis.",
    )
    assumed: list[str] = Field(
        default_factory=list,
        description="The values that are assumed, not given by the user nor estimated.",
    )
    estimated: list[str] = Field(
        default_factory=list,
        description="The values taken from the facts' estimates (the user's hard data).",
    )
    condition: bool = Field(
        default=True,
        description="Condition the field to the hard data, where the facts' estimate of the "
        "variable has some (source data or reference+data): every member then takes the "
        "data's values at their cells.",
    )


class ContinuousParameters(BaseModel):
    """Gaussian fields, one per variable."""

    model_config = ConfigDict(extra="forbid")

    fields: list[FieldChoice] = Field(min_length=1)


class FaciesParameters(BaseModel):
    """Facies by proportion from one latent field, and each facies' property values."""

    model_config = ConfigDict(extra="forbid")

    latent: str = Field(description="The latent state variable (standard normal).")
    facies: list[str] = Field(min_length=2)
    proportions: list[float] = Field(min_length=2)
    model: Literal["Gaussian", "Exponential", "Spherical"] = "Gaussian"
    len_scale: float = Field(gt=0, description="The latent field's length scale, cells.")
    values: dict[str, dict[str, float]] = Field(
        description="Per facies, its value of each property the deck reads, e.g. poro, in "
        "the space the deck reads it: ln(mD) for the facts' log_variables (e.g. permx)."
    )
    conditioning_cells: list[list[int]] = Field(default_factory=list)
    conditioning_facies: list[int] = Field(default_factory=list)
    assumed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "FaciesParameters":
        if len(self.proportions) != len(self.facies):
            raise ValueError("one proportion per facies")
        if abs(sum(self.proportions) - 1.0) > 1e-6 or min(self.proportions) <= 0:
            raise ValueError("the proportions must be positive and sum to 1")
        if set(self.values) != set(self.facies):
            raise ValueError("values must give every facies, and only them")
        if len(self.conditioning_cells) != len(self.conditioning_facies):
            raise ValueError("one facies per conditioning cell")
        return self


@geomodeling_skills.register("continuous")
class ContinuousSkill(GeomodelingSkill):
    """Gaussian fields of continuous properties (ln permeability, porosity, ...)."""

    name = "continuous"
    scenario = "continuous properties (ln PERMX, PORO, NTG) as Gaussian fields with variograms"
    guidance = (
        "One field per variable, in its state space: ln(mD) for a log-permeability. The mean "
        "and std come from the user, a reference field (the deck's values) or, if neither, "
        "the bounds; ranges are in cells (a quarter of the grid's larger side when nothing "
        "else is known). Where the facts' `estimates` give a variable's statistics and "
        "variogram (from hard data or a reference field), use them (unless the user gives "
        "other values) and list them in `estimated`. Mark every value neither given nor "
        "estimated in `assumed`. Fields are conditioned to hard data unless the user asks "
        "otherwise (`condition`)."
    )
    parameters: ClassVar[type[BaseModel]] = ContinuousParameters

    def check(self, parameters: BaseModel, facts: GeomodelFacts) -> list[str]:
        """Return why the fields do not fit the variables (empty: they do)."""
        assert isinstance(parameters, ContinuousParameters)
        problems: list[str] = []
        seen: set[str] = set()
        for field in parameters.fields:
            found = facts.variables.get(field.variable)
            if field.variable in seen:
                problems.append(f"{field.variable} is given twice")
            seen.add(field.variable)
            if found is None:
                problems.append(f"{field.variable} is not a state variable")
                continue
            if found.prior in ("file", "group"):
                problems.append(f"{field.variable} is read from a stored ensemble or drawn in "
                                "a group: it is not one to model")  # fmt: skip
                continue
            if not found.grid or len(found.grid) != 3:
                problems.append(f"{field.variable} has no [nx, ny, nz] grid")
                continue
            extent = max(found.grid[0], found.grid[1])
            for label, value in (("major_range", field.major_range),
                                 ("minor_range", field.minor_range)):  # fmt: skip
                if value is not None and not 0.5 <= value <= 4 * extent:
                    problems.append(f"{field.variable}: {label} {value:g} cells is outside "
                                    f"[0.5, {4 * extent}] for a {extent}-cell grid")  # fmt: skip
            low, high = found.lower, found.upper
            if low is not None and high is not None:
                if not low <= field.mean <= high:
                    problems.append(f"{field.variable}: mean {field.mean:g} is outside its "
                                    f"bounds [{low:g}, {high:g}]")  # fmt: skip
                if field.std > high - low:
                    problems.append(f"{field.variable}: std {field.std:g} is wider than its "
                                    "bounds, which would clip most of the prior")  # fmt: skip
            if field.condition:
                problems += conditioning_problems(field, facts)
        return problems

    def prior(self, parameters: BaseModel, facts: GeomodelFacts) -> PriorDraft:
        """Return each variable's generator section, conditioned to its hard data."""
        assert isinstance(parameters, ContinuousParameters)
        draft = PriorDraft()
        for field in parameters.fields:
            grid = facts.variables[field.variable].grid or []
            options: dict[str, Any] = {
                "grid": grid, "mean": field.mean, "variance": field.std**2,
                "major_range": field.major_range, "azimuth": field.azimuth,
                "variogram": field.variogram,
            }  # fmt: skip
            if field.minor_range is not None:
                options["minor_range"] = field.minor_range
            conditioning = (
                _conditioning(field.variable, field.std, facts) if field.condition else None
            )
            if conditioning is not None:
                options["conditioning"] = conditioning
                draft.notes.append(_conditioning_note(field.variable, conditioning))
            generator = "gaussian_field"
            if grid[0] * grid[1] > DENSE_LAYER_CELLS:
                generator = "geostatspy_sgsim"
                draft.notes.append(f"{field.variable}: layers of {grid[0] * grid[1]} cells: "
                                   "SGSIM (the geostatspy extra)")  # fmt: skip
            draft.variables[field.variable] = {"name": generator, "options": options}
            if field.estimated:
                draft.notes.append(f"{field.variable}: estimated from "
                                   f"{_source(field.variable, facts)}: "
                                   f"{', '.join(field.estimated)}")  # fmt: skip
            if field.assumed:
                draft.notes.append(f"{field.variable}: assumed {', '.join(field.assumed)}")
        return draft

    def qc(
        self,
        parameters: BaseModel,
        sample: dict[str, NDArray[np.float64]],
        tolerance: float,
        facts: GeomodelFacts,
    ) -> list[str]:
        """Return how a drawn sample misses its fields' mean and std (empty: it does not).

        ``sample`` is each variable's ``(cells, members)`` block, clipped to its bounds as
        the run draws it. Allowed: ``tolerance`` stds of the field, plus the sample's own
        noise from its effective size (members times the field's independent patches), so
        that bounds that clip it are found, not sampling noise.
        """
        assert isinstance(parameters, ContinuousParameters)
        problems = []
        for field in parameters.fields:
            block = sample[field.variable]
            minor = field.minor_range or field.major_range
            grid = facts.variables[field.variable].grid or []
            if field.condition:
                problems += _honoured(field.variable, field.std, block, grid, facts)
                # Near the data the field is the data's, not the prior's: check the
                # statistics where the data do not reach.
                block = _beyond_data(field.variable, block, grid, field.major_range, facts)
                if block.size == 0:
                    continue
            n = effective_members(block.shape[-1], block.shape[0], field.major_range * minor)
            allowed = (tolerance + SAMPLING_SIGMAS / math.sqrt(n)) * field.std
            mean, std = float(block.mean()), float(block.std())
            if abs(mean - field.mean) > allowed:
                problems.append(f"{field.variable}: the drawn mean is {mean:.4g}, not "
                                f"{field.mean:.4g} (the bounds clip it?)")  # fmt: skip
            if abs(std - field.std) > allowed:
                problems.append(f"{field.variable}: the drawn std is {std:.4g}, not "
                                f"{field.std:.4g} (the bounds clip it, or the range is "
                                "large for the grid)")  # fmt: skip
        return problems


def _data_of(variable: str, facts: GeomodelFacts) -> tuple[list[list[int]], list[float]]:
    estimate = facts.estimates.get(variable)
    return ([], []) if estimate is None else (estimate.cells, estimate.values)


def _source(variable: str, facts: GeomodelFacts) -> str:
    """Return what a variable's estimate came from, in words."""
    estimate = facts.estimates.get(variable)
    source = "data" if estimate is None else estimate.source
    return {"reference": "the reference field", "reference+data": "the reference field"}.get(
        source, "the hard data"
    )


def _improbable(what: str, excursion: float) -> str:
    return (f"{what}: the hard data are improbable under this variogram: conditioning would "
            f"move the field's mean by up to {excursion:.3g} prior stds (beyond "
            f"{LARGE_CORRECTION:g}; rough values close together under a smooth model?): a "
            "shorter range, a rougher model (spherical, exponential) or another mean; or set "
            "condition false")  # fmt: skip


def conditioning_problems(field: FieldChoice, facts: GeomodelFacts) -> list[str]:
    """Return why a field cannot be conditioned to its variable's hard data (empty: it can).

    Data outside the bounds cannot be honoured (the prior is clipped), and data improbable
    under the variogram (the kriged mean departs from the prior mean by more than
    ``LARGE_CORRECTION`` prior stds; e.g. rough values close together under a smooth
    model) would make the conditioned field swing far outside the prior. The harm itself
    is the test, not a chi-square test of the data, which refused real data whenever the
    estimated sill was a little low.
    """
    problems = _data_problems(field.variable, facts)
    cells, _ = _data_of(field.variable, facts)
    fact = facts.variables.get(field.variable)
    if problems or not cells or fact is None or not fact.grid or len(fact.grid) != 3:
        return problems
    from apex.algorithms.priors import GaussianFieldPrior

    conditioning = _conditioning(field.variable, field.std, facts)
    try:
        prior = GaussianFieldPrior(
            grid=fact.grid, mean=field.mean, variance=field.std**2,
            major_range=field.major_range, minor_range=field.minor_range,
            azimuth=field.azimuth, variogram=field.variogram, conditioning=conditioning,
        )  # fmt: skip
        assert prior.hard is not None
        excursion = prior.hard.excursion(prior)
    except ValueError as error:
        return [f"{field.variable}: its hard data: {error}"]
    if excursion > LARGE_CORRECTION:
        problems.append(_improbable(field.variable, excursion))
    return problems


def _data_problems(variable: str, facts: GeomodelFacts) -> list[str]:
    """Return why a variable's hard data cannot be honoured (outside its bounds)."""
    _, values = _data_of(variable, facts)
    fact = facts.variables.get(variable)
    if not values or fact is None:
        return []
    low = -math.inf if fact.lower is None else fact.lower
    high = math.inf if fact.upper is None else fact.upper
    outside = [v for v in values if not low <= v <= high]
    if not outside:
        return []
    return [f"{variable}: {len(outside)} hard data lie outside its bounds [{low:g}, {high:g}] "
            f"(e.g. {outside[0]:.4g}), where the prior is clipped: widen the bounds, or set "
            "condition false"]  # fmt: skip


def _conditioning(variable: str, std: float, facts: GeomodelFacts) -> dict[str, Any] | None:
    """Return a generator's ``conditioning`` option from the variable's hard data, if any.

    A nugget in the data's variogram is variation the generators (which have none) cannot
    draw: it is given as the data's error, so that the members scatter around the data
    as much as the data scatter around the smooth field.
    """
    cells, values = _data_of(variable, facts)
    if not cells:
        return None
    option: dict[str, Any] = {"cells": cells, "values": values}
    estimate = facts.estimates[variable]
    if estimate.nugget_fraction:
        option["error_std"] = math.sqrt(estimate.nugget_fraction) * std
    return option


def _conditioning_note(variable: str, conditioning: dict[str, Any]) -> str:
    error = conditioning.get("error_std")
    how = "taken exactly" if not error else f"within an error of std {error:.3g} (the nugget)"
    return f"{variable}: conditioned to {len(conditioning['values'])} hard data, {how}"


def _cell_rows(cells: list[list[int]], grid: list[int]) -> NDArray[np.int64]:
    """Return 1-based ``[i, j, k]`` cells' rows in a state block (Eclipse order)."""
    index = np.asarray(cells, dtype=np.int64) - 1
    return np.asarray(index[:, 0] + grid[0] * (index[:, 1] + grid[1] * index[:, 2]),
                      dtype=np.int64)  # fmt: skip


#: An exactly conditioned member may miss a datum by this many of the field's stds (the
#: jitter of a singular data covariance moves it by about 1e-4).
HONOURED_STDS = 1e-3


def _honoured(
    variable: str, std: float, block: NDArray[np.float64], grid: list[int], facts: GeomodelFacts
) -> list[str]:
    """Return how the members miss exact hard data at their cells (empty: they do not)."""
    cells, values = _data_of(variable, facts)
    estimate = facts.estimates.get(variable)
    if not cells or estimate is None or estimate.nugget_fraction:
        return []  # none, or data with errors, which members only scatter around
    missed = np.abs(block[_cell_rows(cells, grid)] - np.asarray(values)[:, None])
    worst = float(missed.max())
    if worst <= HONOURED_STDS * std:
        return []
    return [f"{variable}: the members miss the hard data by up to {worst:.3g} (the bounds "
            "clip them, or the variogram is too smooth for them?)"]  # fmt: skip


def _beyond_data(
    variables: str | list[str],
    block: NDArray[np.float64],
    grid: list[int],
    reach: float,
    facts: GeomodelFacts,
) -> NDArray[np.float64]:
    """Return the block's rows of cells farther than ``reach`` cells from every datum.

    ``variables`` whose data count: for cokriged fields all of them, since each moves the
    others through the correlation.
    """
    names = [variables] if isinstance(variables, str) else variables
    cells = [cell for name in names for cell in _data_of(name, facts)[0]]
    if not cells or len(grid) != 3:
        return block
    nx, ny, nz = grid
    k, rest = np.divmod(np.arange(nx * ny * nz), nx * ny)
    j, i = np.divmod(rest, nx)
    far = np.ones(i.size, dtype=bool)
    for ci, cj, ck in cells:
        # Layers are independent fields: a datum reaches its own layer only.
        near = (k == ck - 1) & (np.hypot(i - (ci - 1), j - (cj - 1)) < reach)
        far &= ~near
    return block[far]


@geomodeling_skills.register("facies_pluri")
class FaciesPluriSkill(GeomodelingSkill):
    """Facies by proportion from a truncated latent Gaussian field, conditioned to wells."""

    name = "facies_pluri"
    scenario = "facies (e.g. shale, sand) by proportion, with each facies' property values"
    guidance = (
        "The state is a latent standard-normal field (a state variable such as y1); facies "
        "are its truncation at the proportions; each facies gives the deck its property "
        "values, in the space the deck reads them: ln(mD) for the facts' log_variables (a "
        "converted case's permx). The case's state must be the latent field only: the "
        "properties are the truncation's outputs. Condition to the wells' facies where they "
        "are known (cells 1-based [i, j, k], facies by index). Where the facts' `facies_data` "
        "come from hard data, take its proportions, len_scale (Gaussian model) and "
        "conditioning cells and facies, unless the user gives others."
    )
    parameters: ClassVar[type[BaseModel]] = FaciesParameters

    def check(self, parameters: BaseModel, facts: GeomodelFacts) -> list[str]:
        """Return why the facies do not fit the variables (empty: they do)."""
        assert isinstance(parameters, FaciesParameters)
        found = facts.variables.get(parameters.latent)
        if found is None:
            return [f"{parameters.latent} is not a state variable (the latent field)"]
        problems = []
        if not found.grid or len(found.grid) != 3:
            problems.append(f"{parameters.latent} has no [nx, ny, nz] grid")
        for cell in parameters.conditioning_cells:
            if found.grid and (len(cell) != 3 or not all(
                    1 <= c <= n for c, n in zip(cell, found.grid, strict=False))):  # fmt: skip
                problems.append(f"conditioning cell {cell} is outside the grid {found.grid}")
        problems += [f"conditioning facies {f} is not one of 0..{len(parameters.facies) - 1}"
                     for f in parameters.conditioning_facies
                     if not 0 <= f < len(parameters.facies)]  # fmt: skip
        for facies, values in parameters.values.items():
            for name, value in values.items():
                if name in facts.variables:
                    problems.append(
                        f"{name} is a state variable as well as the truncation's output: the "
                        "case's state must be the latent field only"
                    )
                if name in facts.log_variables and abs(value) > LOG_LIMIT:
                    problems.append(
                        f"{facies}'s {name} {value:g}: the deck takes exp({name}), so give "
                        f"ln(value) (e.g. ln(500 mD) = {math.log(500):.3g})"
                    )
        return sorted(set(problems), key=problems.index)

    def prior(self, parameters: BaseModel, facts: GeomodelFacts) -> PriorDraft:
        """Return the latent field's group and the truncation that maps it to properties."""
        assert isinstance(parameters, FaciesParameters)
        options: dict[str, Any] = {
            "fields": [{"model": parameters.model, "len_scale": parameters.len_scale}],
            "proportions": parameters.proportions,
        }
        if parameters.conditioning_cells:
            options["conditioning"] = {"cells": parameters.conditioning_cells,
                                       "facies": parameters.conditioning_facies}  # fmt: skip
        draft = PriorDraft()
        draft.groups["facies"] = {"variables": [parameters.latent], "name": "gstools_pgs",
                                  "options": options}  # fmt: skip
        draft.parameterizations.append({"name": "pgs_truncation", "options": {
            "fields": [parameters.latent], "facies": parameters.facies,
            "proportions": parameters.proportions, "values": parameters.values,
        }})  # fmt: skip
        if parameters.assumed:
            draft.notes.append(f"facies: assumed {', '.join(parameters.assumed)}")
        return draft

    def qc(
        self,
        parameters: BaseModel,
        sample: dict[str, NDArray[np.float64]],
        tolerance: float,
        facts: GeomodelFacts,
    ) -> list[str]:
        """Return how far the drawn facies' proportions are from the asked ones.

        The facies are the rule's own (``LithotypeRule.from_proportions``, as
        ``pgs_truncation`` draws them). Allowed per facies: ``tolerance``, plus the sample's
        own noise from its effective size (members times the field's independent patches:
        a long length scale makes each member nearly one facies).
        """
        assert isinstance(parameters, FaciesParameters)
        latent = sample[parameters.latent]
        rule = LithotypeRule.from_proportions(parameters.proportions)
        facies = rule.facies(latent.reshape(1, -1))  # one axis: (1, cells x members)
        n = effective_members(latent.shape[-1], latent.shape[0], parameters.len_scale**2)
        problems = []
        for index, (name, wanted) in enumerate(zip(parameters.facies, parameters.proportions,
                                                    strict=True)):  # fmt: skip
            drawn = float(np.mean(facies == index))
            allowed = tolerance + SAMPLING_SIGMAS * math.sqrt(wanted * (1 - wanted) / n)
            if not math.isclose(drawn, wanted, abs_tol=allowed):
                problems.append(f"facies {name}: drawn in {drawn:.0%} of the cells, not "
                                f"{wanted:.0%}")  # fmt: skip
        return problems


# ------------------------------------------------------------------ phase 3

#: The drawn correlation may differ from the asked one by this much beyond its own noise.
CORRELATION_TOLERANCE = 0.1
#: The largest training image read for a plan's checks.
TI_MAX_MB = 200


class CorrelatedField(BaseModel):
    """One variable of a correlated set: its mean and std, in state space."""

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    mean: float
    std: float = Field(gt=0)


class CorrelatedParameters(BaseModel):
    """Correlated Gaussian fields with one variogram, e.g. ln PERMX and PORO.

    Each variable is ``mean + std * (L z)``, ``z`` independent unit Gaussian fields (one per
    variable, drawn in order) and ``L`` the Cholesky factor of ``correlation``: for two,
    ``mean + std (rho z_1 + sqrt(1 - rho^2) z_2)``, as 3dBox's ``prepare.py``.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[CorrelatedField] = Field(min_length=2)
    correlation: list[list[float]] = Field(
        description="The variables' correlation matrix, in the order of `fields`."
    )
    variogram: Literal["spherical", "exponential", "gaussian"] = "spherical"
    major_range: float = Field(gt=0, description="Cells along the major axis.")
    minor_range: float | None = Field(default=None, gt=0, description="Cells across it.")
    azimuth: float = Field(default=0.0, ge=-180, le=180)
    seed: int = Field(default=2026, ge=0, description="The stored ensemble's seed.")
    assumed: list[str] = Field(default_factory=list)
    estimated: list[str] = Field(default_factory=list)
    condition: bool = Field(
        default=True,
        description="Condition the fields to the hard data in the facts' estimates (each "
        "variable's own, cokriged through the correlation).",
    )

    @model_validator(mode="after")
    def _correlation(self) -> "CorrelatedParameters":
        k = len(self.fields)
        matrix = np.asarray(self.correlation, dtype=np.float64)
        if matrix.shape != (k, k):
            raise ValueError(f"correlation must be {k} x {k}, one row per field")
        if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1.0):
            raise ValueError("correlation must be symmetric with a unit diagonal")
        if np.any(np.abs(matrix) > 1):
            raise ValueError("correlations lie in [-1, 1]")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("correlation must be positive definite") from None
        return self


def _grid_problems(
    names: list[str], facts: GeomodelFacts, *, dense: bool = False
) -> tuple[list[int] | None, list[str]]:
    """Return the variables' common grid, and why they cannot share one (or be modelled)."""
    problems: list[str] = []
    grids = []
    for name in names:
        found = facts.variables.get(name)
        if found is None:
            problems.append(f"{name} is not a state variable")
        elif found.prior in ("file", "group"):
            problems.append(f"{name} is read from a stored ensemble or drawn in a group: it is "
                            "not one to model")  # fmt: skip
        elif not found.grid or len(found.grid) != 3:
            problems.append(f"{name} has no [nx, ny, nz] grid")
        else:
            grids.append(found.grid)
    if len({tuple(g) for g in grids}) > 1:
        problems.append(f"{', '.join(names)} must share one grid")
    grid = grids[0] if grids and not problems else None
    if grid is not None and dense and grid[0] * grid[1] > DENSE_LAYER_CELLS:
        problems.append(f"layers of {grid[0] * grid[1]} cells: correlated fields are drawn "
                        f"densely, up to {DENSE_LAYER_CELLS} cells a layer")  # fmt: skip
    if len(set(names)) != len(names):
        problems.append("a variable is given twice")
    return grid, problems


@geomodeling_skills.register("correlated_continuous")
class CorrelatedContinuousSkill(GeomodelingSkill):
    """Correlated Gaussian fields (e.g. porosity and ln permeability), as a stored ensemble."""

    name = "correlated_continuous"
    scenario = ("correlated continuous properties (ln PERMX and PORO, ...) with one variogram, "
                "as a stored ensemble")  # fmt: skip
    guidance = (
        "For properties that vary together: porosity with ln permeability (correlations of "
        "0.5 to 0.9 are common), or ln PERMX with ln PERMZ. One variogram for all; each "
        "variable's mean and std in its state space (ln(mD) for a log variable). The prior is "
        "drawn once, now, into a stored ensemble beside the new configuration (the generators "
        "cannot correlate variables): give `seed` to draw another. Clipped to the bounds. "
        "Mark assumed values in `assumed`, those from hard-data estimates in `estimated`. "
        "Conditioned to hard data unless the user asks otherwise (`condition`)."
    )
    parameters: ClassVar[type[BaseModel]] = CorrelatedParameters

    def check(self, parameters: BaseModel, facts: GeomodelFacts) -> list[str]:
        """Return why the fields do not fit the variables (empty: they do)."""
        assert isinstance(parameters, CorrelatedParameters)
        names = [f.variable for f in parameters.fields]
        grid, problems = _grid_problems(names, facts, dense=True)
        if grid is not None:
            extent = max(grid[0], grid[1])
            for label, value in (("major_range", parameters.major_range),
                                 ("minor_range", parameters.minor_range)):  # fmt: skip
                if value is not None and not 0.5 <= value <= 4 * extent:
                    problems.append(f"{label} {value:g} cells is outside [0.5, {4 * extent}] "
                                    f"for a {extent}-cell grid")  # fmt: skip
        for field in parameters.fields:
            fact = facts.variables.get(field.variable)
            if fact is None or fact.lower is None or fact.upper is None:
                continue
            if not fact.lower <= field.mean <= fact.upper:
                problems.append(f"{field.variable}: mean {field.mean:g} is outside its bounds "
                                f"[{fact.lower:g}, {fact.upper:g}]")  # fmt: skip
            if field.std > fact.upper - fact.lower:
                problems.append(f"{field.variable}: std {field.std:g} is wider than its bounds")
        if parameters.condition and grid is not None and not problems:
            for field in parameters.fields:
                problems += _data_problems(field.variable, facts)
            if not problems:
                try:
                    excursion = _cokriging_excursion(parameters, facts, grid)
                except ValueError as error:
                    return [f"the hard data: {error}"]
                if excursion > LARGE_CORRECTION:
                    problems.append(_improbable("the fields", excursion))
        return problems

    def prior(self, parameters: BaseModel, facts: GeomodelFacts) -> PriorDraft:
        """Return the stored ensemble and each variable's ``file`` generator."""
        assert isinstance(parameters, CorrelatedParameters)
        members = facts.ensemble_size or 100
        drawn = realize_correlated(parameters, facts, members)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **drawn)  # type: ignore[arg-type]  # (numpy's stubs)
        name = facts.ensemble_file
        draft = PriorDraft(files={name: buffer.getvalue()})
        for field in parameters.fields:
            draft.variables[field.variable] = {
                "name": "file",
                "options": {"ensemble_file": name, "key": field.variable},
            }
        rho = ", ".join(f"{a.variable}-{b.variable} {parameters.correlation[i][j]:g}"
                        for i, a in enumerate(parameters.fields)
                        for j, b in enumerate(parameters.fields) if j > i)  # fmt: skip
        draft.notes.append(f"{members} members drawn into {name} (seed "
                           f"{parameters.seed}; correlations {rho})")  # fmt: skip
        if parameters.condition:
            for field in parameters.fields:
                conditioning = _conditioning(field.variable, field.std, facts)
                if conditioning is not None:
                    draft.notes.append(_conditioning_note(field.variable, conditioning))
        if parameters.estimated:
            sources = sorted({_source(f.variable, facts) for f in parameters.fields})
            draft.notes.append(f"estimated from {' and '.join(sources)}: "
                               f"{', '.join(parameters.estimated)}")  # fmt: skip
        if parameters.assumed:
            draft.notes.append(f"assumed {', '.join(parameters.assumed)}")
        return draft

    def qc(
        self,
        parameters: BaseModel,
        sample: dict[str, NDArray[np.float64]],
        tolerance: float,
        facts: GeomodelFacts,
    ) -> list[str]:
        """Return how a drawn sample misses the means, stds and correlations.

        As ``continuous`` for each variable; the pooled correlation of each pair within
        ``CORRELATION_TOLERANCE`` plus three of its standard errors at the effective size.
        """
        assert isinstance(parameters, CorrelatedParameters)
        problems = []
        minor = parameters.minor_range or parameters.major_range
        grid = facts.variables[parameters.fields[0].variable].grid or []
        for field in parameters.fields:
            block = sample[field.variable]
            if parameters.condition:
                problems += _honoured(field.variable, field.std, block, grid, facts)
                block = _beyond_data([f.variable for f in parameters.fields], block, grid,
                                     parameters.major_range, facts)  # fmt: skip
                if block.size == 0:
                    continue
            n = effective_members(block.shape[-1], block.shape[0],
                                  parameters.major_range * minor)  # fmt: skip
            allowed = (tolerance + SAMPLING_SIGMAS / math.sqrt(n)) * field.std
            mean, std = float(block.mean()), float(block.std())
            if abs(mean - field.mean) > allowed:
                problems.append(f"{field.variable}: the drawn mean is {mean:.4g}, not "
                                f"{field.mean:.4g} (the bounds clip it?)")  # fmt: skip
            if abs(std - field.std) > allowed:
                problems.append(f"{field.variable}: the drawn std is {std:.4g}, not "
                                f"{field.std:.4g} (the bounds clip it?)")  # fmt: skip
        for i, a in enumerate(parameters.fields):
            for j, b in enumerate(parameters.fields[i + 1 :], start=i + 1):
                x, y = sample[a.variable].ravel(), sample[b.variable].ravel()
                n = effective_members(sample[a.variable].shape[-1], sample[a.variable].shape[0],
                                      parameters.major_range * minor)  # fmt: skip
                wanted = parameters.correlation[i][j]
                drawn = float(np.corrcoef(x, y)[0, 1])
                allowed = CORRELATION_TOLERANCE + SAMPLING_SIGMAS * (1 - wanted**2) / math.sqrt(n)
                if abs(drawn - wanted) > allowed:
                    problems.append(f"{a.variable}-{b.variable}: drawn correlation {drawn:.3g}, "
                                    f"not {wanted:.3g} (the bounds clip it?)")  # fmt: skip
        return problems


def realize_correlated(
    parameters: CorrelatedParameters, facts: GeomodelFacts, members: int
) -> dict[str, NDArray[np.float64]]:
    """Return the correlated fields, ``(cells, members)`` per variable, clipped to bounds.

    The unit fields come from ``gaussian_field``'s sampler, one per variable in order from
    one generator seeded with ``parameters.seed`` (as ``prepare.py``'s ``z_k``, ``z_p``).
    With ``condition``, the fields are then conditioned to every variable's hard data at
    once (:func:`_cokrige`), before the clipping.
    """
    from apex.algorithms.priors import GaussianFieldPrior
    from apex.domain.state import StateVariable

    grid = facts.variables[parameters.fields[0].variable].grid
    assert grid is not None
    sampler = GaussianFieldPrior(grid=grid, mean=0.0, variance=1.0,
                                 major_range=parameters.major_range,
                                 minor_range=parameters.minor_range,
                                 azimuth=parameters.azimuth,
                                 variogram=parameters.variogram)  # fmt: skip
    rng = np.random.default_rng(parameters.seed)
    size = grid[0] * grid[1] * grid[2]
    unit = [sampler.generate(StateVariable(name=f.variable, size=size), members, rng)
            for f in parameters.fields]  # fmt: skip
    factor = np.linalg.cholesky(np.asarray(parameters.correlation, dtype=np.float64))
    fields = []
    for i, field in enumerate(parameters.fields):
        mixed = sum(factor[i, j] * unit[j] for j in range(i + 1))
        fields.append(field.mean + field.std * np.asarray(mixed))
    if parameters.condition:
        fields = _cokrige(parameters, facts, fields, grid, rng)
    drawn = {}
    for field, values in zip(parameters.fields, fields, strict=True):
        fact = facts.variables[field.variable]
        low = -np.inf if fact.lower is None else fact.lower
        high = np.inf if fact.upper is None else fact.upper
        # A stored prior is not clipped by the run (as PET's importstate): clip it here.
        drawn[field.variable] = np.ascontiguousarray(np.clip(values, low, high))
    return drawn


@dataclass(frozen=True)
class _CokrigingLayer:
    """One layer's simple-cokriging system over the stacked variables (see :func:`_cokrige`).

    ``here`` selects the layer's data, ``rows`` are their entries in the stacked layer
    ``(n_var * cells,)``, ``cross`` is ``(n_var * cells, n_here)``.
    """

    k: int
    here: NDArray[np.bool_]
    rows: NDArray[np.int64]
    cross: NDArray[np.float64]


@dataclass(frozen=True)
class _CokrigingData:
    """All variables' hard data: variable index, value and error variance per datum."""

    which: NDArray[np.int64]
    values: NDArray[np.float64]
    error_variance: NDArray[np.float64]
    layers: list[_CokrigingLayer]


def _cokriging_data(
    parameters: CorrelatedParameters, facts: GeomodelFacts, grid: list[int]
) -> _CokrigingData | None:
    """Return the correlated fields' hard data and per-layer systems, or ``None`` if none.

    The fields' joint covariance is ``s_a s_b R_ab C(x, y)`` (one unit variogram ``C``, the
    correlation ``R``, the stds ``s``), so block ``a`` of the cross-covariance with datum
    ``q`` of variable ``b`` is ``s_a s_b R_ab C(x, x_q)``.
    """
    from apex.algorithms.geostatistics import layer_cross_covariance

    nx, ny, _ = grid
    layer = nx * ny
    variable, columns, layers, values, error = [], [], [], [], []
    for a, field in enumerate(parameters.fields):
        option = _conditioning(field.variable, field.std, facts)
        if option is None:
            continue
        rows = _cell_rows(option["cells"], grid)
        variable += [a] * rows.size
        layers += (rows // layer).tolist()
        columns += (rows % layer).tolist()
        values += option["values"]
        error += [option.get("error_std", 0.0) ** 2] * rows.size
    if not values:
        return None
    which, column, depth = np.array(variable), np.array(columns), np.array(layers)
    correlation = np.asarray(parameters.correlation, dtype=np.float64)
    std = np.array([f.std for f in parameters.fields])
    systems = []
    for k in sorted(set(depth.tolist())):
        here = depth == k
        unit = layer_cross_covariance(
            nx, ny, column[here], model=parameters.variogram, sill=1.0,
            major_range=parameters.major_range, minor_range=parameters.minor_range,
            azimuth=parameters.azimuth,
        )  # fmt: skip
        scale = std[:, None] * (correlation[:, which[here]] * std[which[here]][None, :])
        cross = np.vstack([unit * scale[a][None, :] for a in range(len(parameters.fields))])
        systems.append(_CokrigingLayer(k, here, which[here] * layer + column[here], cross))
    return _CokrigingData(which, np.array(values), np.array(error), systems)


def _stacked_variance(parameters: CorrelatedParameters, layer: int) -> NDArray[np.float64]:
    """Return each stacked entry's prior variance, ``(n_var * layer,)``."""
    return np.repeat(np.array([f.std**2 for f in parameters.fields]), layer)


def _cokriging_excursion(
    parameters: CorrelatedParameters, facts: GeomodelFacts, grid: list[int]
) -> float:
    """Return how far the cokriged mean departs from the prior means, in prior stds.

    One test of all variables' data together, so data that contradict the correlation
    (a high ln k beside a low porosity at a strong positive one) count too.
    """
    from apex.algorithms.geostatistics import kriged_excursion

    found = _cokriging_data(parameters, facts, grid)
    if found is None:
        return 0.0
    means = np.array([f.mean for f in parameters.fields])
    variance = _stacked_variance(parameters, grid[0] * grid[1])
    worst = 0.0
    for system in found.layers:
        departure = found.values[system.here] - means[found.which[system.here]]
        error = found.error_variance[system.here]
        found_here = kriged_excursion(system.cross, system.rows, departure, error, variance)
        worst = max(worst, found_here)
    return worst


def _cokrige(
    parameters: CorrelatedParameters,
    facts: GeomodelFacts,
    fields: list[NDArray[np.float64]],
    grid: list[int],
    rng: np.random.Generator,
) -> list[NDArray[np.float64]]:
    """Return the correlated fields conditioned to all their variables' hard data.

    Kriging the stacked residuals with the joint covariance (simple cokriging;
    :func:`_cokriging_data`) conditions every variable to its own data and, through ``R``,
    to the others' (:func:`~apex.algorithms.geostatistics.condition_by_kriging`). Layers
    are independent. The data's error draws come from ``rng`` after the fields', only for
    data with errors.
    """
    from apex.algorithms.geostatistics import condition_by_kriging

    found = _cokriging_data(parameters, facts, grid)
    if found is None:
        return fields
    layer = grid[0] * grid[1]
    members = fields[0].shape[1]
    noise = (rng.standard_normal((found.values.size, members))
             if np.any(found.error_variance > 0) else None)  # fmt: skip
    variance = _stacked_variance(parameters, layer)
    out = [np.array(f, copy=True) for f in fields]
    n_var = len(fields)
    for system in found.layers:
        k, here = system.k, system.here
        stacked = np.vstack([out[a][k * layer : (k + 1) * layer] for a in range(n_var)])
        conditioned = condition_by_kriging(
            stacked, system.cross, system.rows, found.values[here], found.error_variance[here],
            None if noise is None else noise[here], prior_variance=variance,
        )  # fmt: skip
        for a in range(n_var):
            out[a][k * layer : (k + 1) * layer] = conditioned[a * layer : (a + 1) * layer]
    return out


class MpsParameters(BaseModel):
    """SNESIM facies from a training image, as signed distance fields, and their values."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(
        min_length=2, description="One distance state variable per facies, in code order."
    )
    facies: list[str] = Field(min_length=2, description="Facies names, code 0 first.")
    training_image_file: str = Field(
        description="The training image (.npy codes [i, j(, k)] or a GSLIB grid), relative to "
        "the configuration's folder; its codes are 0..K-1."
    )
    values: dict[str, dict[str, float]] = Field(
        description="Per facies, its value of each property the deck reads (ln(mD) for the "
        "facts' log_variables)."
    )
    conditioning_cells: list[list[int]] = Field(default_factory=list)
    conditioning_facies: list[int] = Field(default_factory=list)
    template_size: list[int] = Field(default=[8, 7, 1], min_length=3, max_length=3)
    n_multiple_grids: int = Field(default=3, ge=1, le=6)
    distance_band: float | None = Field(default=None, gt=0)
    assumed: list[str] = Field(default_factory=list)
    estimated: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "MpsParameters":
        if len(self.fields) != len(self.facies):
            raise ValueError("one distance field per facies")
        if set(self.values) != set(self.facies):
            raise ValueError("values must give every facies, and only them")
        if len(self.conditioning_cells) != len(self.conditioning_facies):
            raise ValueError("one facies per conditioning cell")
        if any(n < 1 for n in self.template_size):
            raise ValueError("template_size is [tx, ty, tz], each at least 1")
        return self


@geomodeling_skills.register("facies_mps")
class FaciesMpsSkill(GeomodelingSkill):
    """Facies from a training image (MPSlib's SNESIM), as signed distances, with values."""

    name = "facies_mps"
    scenario = ("facies with shapes from a training image (channels, lobes) by multiple-point "
                "statistics, conditioned to wells")  # fmt: skip
    guidance = (
        "Needs a training image of the facies' shapes (codes 0..K-1). The state is one signed "
        "distance field per facies (state variables, in code order); facies_distance maps "
        "them to the facies and each facies' property values, in the space the deck reads "
        "them (ln(mD) for the facts' log_variables). Condition to the wells' facies (cells "
        "1-based [i, j, k], facies by code), from the facts' `facies_data` when hard data "
        "give them. template_size [tx, ty, tz] should cover the training image's features."
    )
    parameters: ClassVar[type[BaseModel]] = MpsParameters

    def _training_image(
        self, parameters: MpsParameters, facts: GeomodelFacts
    ) -> tuple[NDArray[np.int64] | None, str | None]:
        from apex.algorithms.facies.snesim import read_training_image

        # The model names the file: read it only inside the case's folder, and say the same
        # whatever stands in the way (missing, outside, too large, unreadable), so that a
        # plan never tells whether or what a file elsewhere is.
        refused = (f"no training image {parameters.training_image_file!r} in the case's "
                   f"folder (an .npy or GSLIB grid of codes, at most {TI_MAX_MB} MB)")  # fmt: skip
        path = facts.own_file(parameters.training_image_file)
        if path is None or path.stat().st_size > TI_MAX_MB * 2**20:
            return None, refused
        try:
            return read_training_image(path), None
        except (OSError, ValueError, UnicodeDecodeError):
            return None, refused

    def check(self, parameters: BaseModel, facts: GeomodelFacts) -> list[str]:
        """Return why the facies do not fit the variables or the training image."""
        assert isinstance(parameters, MpsParameters)
        grid, problems = _grid_problems(parameters.fields, facts)
        image, why = self._training_image(parameters, facts)
        if why:
            problems.append(why)
        elif image is not None:
            codes = sorted(int(c) for c in np.unique(image))
            if codes != list(range(len(parameters.facies))):
                problems.append(f"the training image's codes are {codes}, not 0..."
                                f"{len(parameters.facies) - 1} for {len(parameters.facies)} "
                                "facies")  # fmt: skip
        for cell in parameters.conditioning_cells:
            if grid and (len(cell) != 3 or not all(
                    1 <= c <= n for c, n in zip(cell, grid, strict=False))):  # fmt: skip
                problems.append(f"conditioning cell {cell} is outside the grid {grid}")
        problems += [f"conditioning facies {f} is not one of 0..{len(parameters.facies) - 1}"
                     for f in parameters.conditioning_facies
                     if not 0 <= f < len(parameters.facies)]  # fmt: skip
        for facies, values in parameters.values.items():
            for name, value in values.items():
                if name in facts.variables:
                    problems.append(f"{name} is a state variable as well as a facies' value: "
                                    "the state must be the distance fields only")  # fmt: skip
                if name in facts.log_variables and abs(value) > LOG_LIMIT:
                    problems.append(f"{facies}'s {name} {value:g}: the deck takes exp({name}), "
                                    f"so give ln(value)")  # fmt: skip
        return sorted(set(problems), key=problems.index)

    def prior(self, parameters: BaseModel, facts: GeomodelFacts) -> PriorDraft:
        """Return the distance fields' SNESIM group and the facies_distance mapping."""
        assert isinstance(parameters, MpsParameters)
        options: dict[str, Any] = {
            "training_image_file": parameters.training_image_file,
            "template_size": parameters.template_size,
            "n_multiple_grids": parameters.n_multiple_grids,
        }
        if parameters.distance_band is not None:
            options["distance_band"] = parameters.distance_band
        if parameters.conditioning_cells:
            options["conditioning"] = {"cells": parameters.conditioning_cells,
                                       "facies": parameters.conditioning_facies}  # fmt: skip
        draft = PriorDraft()
        draft.groups["facies"] = {"variables": list(parameters.fields), "name": "mpslib_snesim",
                                  "options": options}  # fmt: skip
        draft.parameterizations.append({"name": "facies_distance", "options": {
            "fields": list(parameters.fields), "facies": list(parameters.facies),
            "values": parameters.values,
        }})  # fmt: skip
        if parameters.estimated:
            draft.notes.append(f"facies: estimated from the hard data: "
                               f"{', '.join(parameters.estimated)}")  # fmt: skip
        if parameters.assumed:
            draft.notes.append(f"facies: assumed {', '.join(parameters.assumed)}")
        return draft

    def qc(
        self,
        parameters: BaseModel,
        sample: dict[str, NDArray[np.float64]],
        tolerance: float,
        facts: GeomodelFacts,
    ) -> list[str]:
        """Return how far the drawn facies' proportions are from the training image's.

        The facies are ``facies_distance``'s (the largest signed distance). Allowed per
        facies: ``tolerance`` plus three standard errors over the members alone (the
        realisations' own spatial correlation is unknown here: a conservative size).
        """
        assert isinstance(parameters, MpsParameters)
        image, _ = self._training_image(parameters, facts)
        if image is None:
            return []
        stacked = np.stack([sample[f] for f in parameters.fields])  # (facies, cells, members)
        drawn_facies = np.argmax(stacked, axis=0)
        members = stacked.shape[-1]
        problems = []
        for code, name in enumerate(parameters.facies):
            wanted = float(np.mean(image == code))
            drawn = float(np.mean(drawn_facies == code))
            allowed = tolerance + SAMPLING_SIGMAS * math.sqrt(wanted * (1 - wanted) / members)
            if abs(drawn - wanted) > allowed:
                problems.append(f"facies {name}: drawn in {drawn:.0%} of the cells, the "
                                f"training image has {wanted:.0%}")  # fmt: skip
        return problems
