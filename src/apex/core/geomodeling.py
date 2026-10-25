"""Geomodeling skills: what the geomodeler knows of each modeling scenario.

A skill turns validated parameters into a prior APEX's generators draw (the ``[prior]``
sections, and ``[[parameterizations]]`` where the state is latent), and checks a sample
of it against its own parameters. Skills are plugins (``core.plugins.geomodeling_skills``,
the entry-point group ``apex.geomodeling_skills``): a package adds scenarios without a
change to the agent.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class VariableFact(BaseModel):
    """A state variable the geomodeler may give a prior.

    Its grid, size, bounds and generator, and the statistics of a reference field (the
    deck's values, in state space), if one is given.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    size: int
    grid: list[int] | None = None
    lower: float | None = None
    upper: float | None = None
    reference_mean: float | None = None
    reference_std: float | None = None
    prior: str | None = Field(default=None, description="Its generator now, if it has one.")


class PropertyEstimate(BaseModel):
    """A continuous property's statistics and variogram estimated from hard data.

    In state space (``ln`` of a log variable's values). Ranges in cells, the azimuth in
    degrees counter-clockwise from the i axis; ``None`` where the data could not tell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable: str
    n_data: int
    mean: float
    std: float
    variogram: str | None = None
    major_range: float | None = None
    minor_range: float | None = None
    azimuth: float | None = None
    nugget_fraction: float | None = Field(
        default=None,
        description="The nugget's share of the total sill (the generators have "
        "none: a large one means the data vary on scales finer than a cell or are noisy).",
    )
    source: str = Field(
        default="data",
        description="Where the statistics come from: `data` (the hard data, which the prior "
        "is conditioned to), `reference` (a reference field: statistics only, nothing "
        "conditioned) or `reference+data` (the reference's statistics, conditioned to the "
        "hard data).",
    )
    notes: list[str] = Field(default_factory=list)
    cells: list[list[int]] = Field(
        default_factory=list,
        exclude=True,
        description="The hard data's 1-based cells, which a conditioned prior honours; "
        "code passes them on, the model only chooses whether to condition.",
    )
    values: list[float] = Field(
        default_factory=list, exclude=True, description="The hard data, in state space."
    )


class FaciesEstimate(BaseModel):
    """Facies estimated from hard data: proportions, a length scale, the wells' facies.

    ``len_scale`` is for ``facies_pluri``'s ``Gaussian`` latent field, from the indicator
    variograms mapped back to the latent field's (``indicator_ranges``: the latent practical
    range found at each facies boundary, by the code above it).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_data: int
    codes: list[int]
    proportions: list[float]
    indicator_ranges: dict[int, float] = Field(default_factory=dict)
    len_scale: float | None = None
    conditioning_cells: list[list[int]] = Field(default_factory=list)
    conditioning_facies: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GeomodelFacts(BaseModel):
    """What code found in the configuration: the facts every choice is held to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variables: dict[str, VariableFact]
    ensemble_size: int | None = None
    log_variables: list[str] = Field(
        default_factory=list,
        description="The names the deck template exponentiates: their values are ln(value) "
        "(e.g. permx in ln mD), whether state variables or a parameterization's outputs.",
    )
    kind: str = Field(description="The case's kind: synthetic (it needs a truth) or real.")
    estimates: dict[str, PropertyEstimate] = Field(
        default_factory=dict,
        description="Continuous properties estimated from the user's hard data, by state "
        "variable: use them unless the user gives other values.",
    )
    facies_data: FaciesEstimate | None = Field(
        default=None, description="Facies estimated from the user's hard data (facies codes)."
    )
    folder: str | None = Field(
        default=None,
        exclude=True,
        description="The configuration's folder, against which skills resolve the files their "
        "parameters name (a training image), and within which they must lie; never shown to "
        "the model.",
    )
    ensemble_file: str = Field(
        default="geomodel_prior.npz",
        exclude=True,
        description="The name a skill's stored ensemble is written under, beside the new "
        "configuration (the agent names it after it).",
    )

    def own_file(self, name: str) -> Path | None:
        """Return ``name`` resolved in :attr:`folder` if it is a file there, else ``None``.

        A path a model chose: absolute paths, ``..`` and links out of the folder are refused,
        so that a skill never reads another user's (or the server's) file for a plan.
        """
        if self.folder is None or not name or Path(name).is_absolute():
            return None
        folder = Path(self.folder).resolve()
        path = (folder / name).resolve()
        return path if folder in path.parents and path.is_file() else None


@dataclass
class PriorDraft:
    """A prior written by a skill: the configuration's sections it sets, and notes.

    ``prior`` replaces ``[prior]``'s generators for the skill's variables
    (``variables``/``groups`` entries); ``parameterizations`` are added; ``files`` are
    written beside the configuration.
    """

    variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameterizations: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class GeomodelingSkill:
    """A geomodeling skill's interface (see the module)."""

    name: ClassVar[str]
    #: One line for the model: the scenario it models.
    scenario: ClassVar[str]
    #: When it fits, what its parameters mean, typical values and traps.
    guidance: ClassVar[str] = ""
    #: The pydantic model of its parameters (checked when the model chooses them).
    parameters: ClassVar[type[BaseModel]]

    def check(self, parameters: BaseModel, facts: GeomodelFacts) -> list[str]:
        """Return why ``parameters`` do not fit ``facts`` (empty: they do)."""
        return []

    def prior(self, parameters: BaseModel, facts: GeomodelFacts) -> PriorDraft:
        """Return the prior ``parameters`` make for ``facts``' variables."""
        raise NotImplementedError

    def qc(
        self,
        parameters: BaseModel,
        sample: dict[str, NDArray[np.float64]],
        tolerance: float,
        facts: GeomodelFacts,
    ) -> list[str]:
        """Return how a drawn sample (``(cells, members)`` per variable) misses ``parameters``.

        ``tolerance`` is the skill's allowance beyond the sample's own noise, which the
        skill estimates from its effective size. Empty: it does not, or no check.
        """
        return []


@dataclass(frozen=True)
class VariogramEstimate:
    """A variogram estimated from hard data, in the skills' terms.

    Ranges are practical ranges in cells, the azimuth in degrees counter-clockwise from the
    i axis (as ``gaussian_field``'s); ``sill`` is the partial sill (the structured variance),
    ``nugget`` the rest. ``residual`` is the fit's weighted residual, ``residuals`` each
    model's. ``mean`` and ``variance`` are the data's own.
    """

    model: str
    nugget: float
    sill: float
    major_range: float
    minor_range: float
    azimuth: float
    residual: float
    n_data: int
    mean: float
    variance: float
    residuals: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class VariogramEstimator(Protocol):
    """Estimates a variogram from values at cells (``core.plugins.variogram_estimators``)."""

    name: str

    def estimate(
        self,
        cells: NDArray[np.int64],
        values: NDArray[np.float64],
        *,
        model: str | None = None,
        anisotropic: bool = True,
    ) -> VariogramEstimate:
        """Return the variogram of ``values`` (``(n,)``) at ``cells`` (``(n, 3)``, 1-based)."""
        ...
