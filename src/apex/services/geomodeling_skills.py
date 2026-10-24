"""The geomodeler's skills of phase 1: continuous fields and plurigaussian facies.

- ``continuous``: a Gaussian field per variable (ln PERMX, PORO, ...), around a mean with a
  standard deviation and a variogram, drawn by ``gaussian_field`` (a dense Cholesky factor
  per layer, up to ``DENSE_LAYER_CELLS`` cells) or ``geostatspy_sgsim`` (larger layers,
  the ``geostatspy`` extra).
- ``facies_pluri``: facies from one latent Gaussian field truncated at the facies'
  proportions (``gstools_pgs``, conditioned to wells' facies where given; the ``facies``
  extra), and each facies' property values (``pgs_truncation``).

Each skill checks its parameters against the configuration's facts, and a drawn sample
against its parameters (:meth:`qc`): code, not the model, says whether the prior is the
one asked for.
"""

import math
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apex.algorithms.facies.rules import LithotypeRule
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
        "else is known). Mark every value neither given nor estimated in `assumed`."
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
        return problems

    def prior(self, parameters: BaseModel, facts: GeomodelFacts) -> PriorDraft:
        """Return each variable's generator section."""
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
            generator = "gaussian_field"
            if grid[0] * grid[1] > DENSE_LAYER_CELLS:
                generator = "geostatspy_sgsim"
                draft.notes.append(f"{field.variable}: layers of {grid[0] * grid[1]} cells: "
                                   "SGSIM (the geostatspy extra)")  # fmt: skip
            draft.variables[field.variable] = {"name": generator, "options": options}
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
        "are known (cells 1-based [i, j, k], facies by index)."
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
