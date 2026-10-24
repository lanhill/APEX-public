"""Geomodeling skills: what the geomodeler knows of each modeling scenario.

A skill turns validated parameters into a prior APEX's generators draw (the ``[prior]``
sections, and ``[[parameterizations]]`` where the state is latent), and checks a sample
of it against its own parameters. Skills are plugins (``core.plugins.geomodeling_skills``,
the entry-point group ``apex.geomodeling_skills``): a package adds scenarios without a
change to the agent.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

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
