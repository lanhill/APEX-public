"""The deck skills of the case-converter: ``opm_flow`` and ``eclipse``.

Both read an Eclipse-format deck (:mod:`~apex.simulation.conversion.deck_reader`) and write
an APEX case on their simulator (:mod:`~apex.simulation.conversion.deck_template`). The case
is written as follows:

- **the template**: ``deck.mako``, with the deck's included files beside it;
- **the state**: one gridded variable per chosen property, over every cell. Cells that are 0
  in the deck (inactive, barriers) stay 0 whatever the state (the template keeps them); the
  statistics below are of the other cells. Bounds that the choices leave out
  come from the deck's values: a log property's range widened by
  ``LOG_MARGIN``, other properties' range scaled by ``LINEAR_MARGIN``, within physical
  limits;
- **the prior**: centred on the deck's values. Its spread is the deck's own, or an assumed
  one where the deck's values are uniform. Its correlation range is ``RANGE_FRACTION`` of the
  grid's larger side. Layers of up to ``DENSE_LAYER_CELLS`` cells use the ``gaussian_field``
  generator, which needs one dense Cholesky factor per layer; larger layers use independent
  ``gaussian`` values. Every assumed value is noted for the geomodeler to replace;
- **the observations**: the chosen vectors at the report dates. Each value's variance is
  ``max(relative_error * |value|, floor)²``. A value at a date the deck does not report is
  left out, with a note;
- **the method**: GIES with ``rlmmac`` (APEX's default) or ES-MDA with equal inflations,
  ``adaptive_prior`` localization unless it is ``none``;
- **the case**: ``real``; a converted case has no truth.
"""

import csv
import io
import math
from pathlib import Path
from typing import ClassVar

import numpy as np

from apex.core.conversion import (
    Capability,
    CaseFacts,
    ConversionChoices,
    ConversionDraft,
    ConverterSkill,
)
from apex.core.errors import ConfigError
from apex.core.plugins import converter_skills
from apex.domain.deck_spec import DeckSpec
from apex.simulation.conversion.deck_reader import DeckData, read_deck
from apex.simulation.conversion.deck_template import StateProperty, runs, write_template
from apex.simulation.conversion.deck_writer import write_deck

#: Gaussian fields on layers of up to this many cells: ``gaussian_field`` factors a dense
#: (cells x cells) covariance per layer, 0.8 GB at 10^4 cells (algorithms/priors.py).
DENSE_LAYER_CELLS = 10_000
#: The prior's assumed correlation range, as a fraction of the grid's larger side (cells).
RANGE_FRACTION = 0.25
#: A log property's bounds: the deck's range of ln(value), widened by this on each side.
LOG_MARGIN = 2.0
#: Other properties' bounds: the deck's minimum / maximum, divided / multiplied by this.
LINEAR_MARGIN = 2.0
#: The assumed prior standard deviation where the deck's values are uniform: of ln(value)
#: for log properties, and as a fraction of the value for others.
ASSUMED_LOG_STD = 1.0
ASSUMED_RELATIVE_STD = 0.2
#: Physical limits of fractions (porosity, net-to-gross, saturations).
FRACTIONS = frozenset({"PORO", "NTG", "SWAT", "SGAS", "SWL", "SWCR", "SGU", "SOWCR"})
TEMPLATE = "deck.mako"
OBSERVATIONS = "observations.csv"
TRUTH = "truth.npz"
#: The created case's description, kept so that its observations can be made again.
SPEC = "case_spec.json"
#: The draft's own files: an included file of the same name is refused.
RESERVED = frozenset({TEMPLATE, OBSERVATIONS, TRUTH, SPEC, "config.toml", "CONVERSION.md"})
SEED = 2026


class DeckSkill(ConverterSkill):
    """An Eclipse-format deck converted to an APEX case on ``simulator`` (see the module)."""

    formats: ClassVar[tuple[str, ...]] = ("eclipse_deck", "apex_config")
    capabilities: ClassVar[frozenset[Capability]] = frozenset({"read", "write_config", "create"})
    unsupported: ClassVar[tuple[str, ...]] = (
        "properties set only by EQUALS or COPY cannot be the state (give them as arrays)",
        "arrays over a BOX, and keywords changed in a box after they are set, are read as "
        "they are, not as the state",
        "decks with Mako in them are already APEX templates",
    )

    def accepts(self, source: Path) -> bool:
        """Whether ``source`` is a ``.DATA`` deck."""
        return source.suffix.upper() == ".DATA"

    def read(self, source: Path, history: Path | None = None) -> CaseFacts:
        """Return the deck's facts (:func:`~.deck_reader.read_deck`)."""
        return read_deck(source, skill=self.name, history=history, allowed=self.allowed).facts

    def write_config(
        self,
        source: Path,
        facts: CaseFacts,
        choices: ConversionChoices,
        history: Path | None = None,
    ) -> ConversionDraft:
        """Return the APEX case of the deck (see the module)."""
        data = read_deck(source, skill=self.name, history=history, allowed=self.allowed)
        notes: list[str] = []
        files: dict[str, str | bytes] = {}
        chosen: dict[str, StateProperty] = {}
        for choice in choices.variables:
            if choice.property not in data.arrays:
                raise ConfigError(f"{choice.property} is not set as an array in the deck")
            values = data.arrays[choice.property]
            kept = None
            if np.any(values == 0):
                # Inactive cells and barriers keep the deck's 0, whatever the member's state.
                kept = runs(values != 0)
                notes.append(f"{choice.property}: {int(np.sum(values == 0))} cells are 0 in the "
                             "deck and stay 0 (not matched)")  # fmt: skip
            chosen[choice.property] = StateProperty(choice.property.lower(), choice.transform,
                                                    kept)  # fmt: skip
        for name in ("PERMY", "PERMZ"):
            found = data.facts.properties.get(name)
            if "PERMX" in chosen and name not in chosen and found and found.source == "array":
                notes.append(f"{name} is an array of its own and stays as in the deck while "
                             "PERMX is matched; choose it too, or COPY it from PERMX in the "
                             "deck, to keep them together")  # fmt: skip
        template = write_template(source, chosen, self.allowed)
        notes += template.notes
        files[TEMPLATE] = template.text
        clash = sorted(set(template.files) & RESERVED)
        if clash:
            raise ConfigError(f"the deck includes files named as the converter's own: {clash}")
        files.update({rel: path.read_bytes() for rel, path in template.files.items()})
        files[OBSERVATIONS] = self._observations(data, choices, notes)
        grid = data.facts.grid or []
        variables, priors = [], {}
        for choice in choices.variables:
            name = choice.property.lower()
            values = data.arrays[choice.property]
            set_ = values[values != 0]
            if choice.transform == "log" and not np.all(set_ > 0):
                raise ConfigError(f"{choice.property} has values < 0: it cannot be log")
            state = np.log(set_) if choice.transform == "log" else set_
            lower, upper = self._bounds(choice.property, choice.transform, state,
                                        choice.lower, choice.upper, notes)  # fmt: skip
            variables.append({"name": name, "size": int(values.size), "grid": grid,
                              "lower": lower, "upper": upper})  # fmt: skip
            priors[name] = self._prior(choice.property, choice.transform, state, grid, notes)
        config: dict[str, object] = {
            "run": {"name": choices.name, "seed": SEED},
            "state": {"variables": variables},
            "prior": {"ensemble_size": choices.ensemble_size, "variables": priors},
            "observations": {"file": OBSERVATIONS},
            "case": {"kind": "real"},
            "simulator": {"name": self.simulator, "options": self._simulator_options()},
            "execution": {"name": "process", "options": {"workers": 10}},
            "method": method_section(choices.method, choices.iterations, choices.localization),
        }
        return ConversionDraft(config, files, notes)

    def create(self, spec: DeckSpec) -> ConversionDraft:
        """Return the synthetic case of ``spec``: its deck, simulator, state and case.

        The state is ln PERMX and PORO over every cell, bounded five standard deviations
        about their means (PORO within physical limits); the case is synthetic, its truth
        ``truth.npz``, its observations ``observations.csv`` (written by the agent).
        """
        grid = [spec.grid.nx, spec.grid.ny, spec.grid.nz]
        size = spec.grid.nx * spec.grid.ny * spec.grid.nz
        props = spec.properties
        variables = [
            {"name": "permx", "size": size, "grid": grid,
             "lower": props.log_permx - 5 * props.permx_log_std,
             "upper": props.log_permx + 5 * props.permx_log_std},
            {"name": "poro", "size": size, "grid": grid,
             "lower": max(1e-3, props.poro - 5 * props.poro_std),
             "upper": min(0.6, props.poro + 5 * props.poro_std)},
        ]  # fmt: skip
        config: dict[str, object] = {
            "run": {"name": spec.name, "seed": SEED},
            "state": {"variables": variables},
            "observations": {"file": OBSERVATIONS},
            "case": {"kind": "synthetic", "truth_file": TRUTH},
            "simulator": {"name": self.simulator, "options": self._simulator_options()},
            "execution": {"name": "process", "options": {"workers": 10}},
        }
        files: dict[str, str | bytes] = {TEMPLATE: write_deck(spec),
                                         SPEC: spec.model_dump_json(indent=2)}  # fmt: skip
        notes = [
            f"a synthetic {spec.fluid.preset} case on {self.simulator}: {size} cells, "
            f"{len(spec.wells)} wells, {spec.schedule.steps} report steps"
        ]
        return ConversionDraft(config, files, notes)

    def _simulator_options(self) -> dict[str, object]:
        return {"deck_template_file": TEMPLATE}

    def _observations(self, data: DeckData, choices: ConversionChoices, notes: list[str]) -> str:
        reported = set(data.facts.report_dates)
        wanted = {o.vector: o for o in choices.observations}
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["time", "datatype", "value", "variance"])
        dropped = 0
        written: set[str] = set()
        for (date, vector), value in sorted(data.history.items()):
            choice = wanted.get(vector)
            if choice is None:
                continue
            if reported and date not in reported:
                dropped += 1
                continue
            sigma = max(choice.relative_error * abs(value), choice.floor)
            writer.writerow([date.isoformat(), vector, repr(value), repr(sigma * sigma)])
            written.add(vector)
        if dropped:
            notes.append(f"{dropped} observed values at dates the deck does not report: left out")
        missing = sorted(set(wanted) - written)
        if missing:
            raise ConfigError(
                f"no observed data at the deck's report dates for {', '.join(missing)}"
            )
        return buffer.getvalue()

    @staticmethod
    def _bounds(
        name: str,
        transform: str,
        state: np.ndarray,
        lower: float | None,
        upper: float | None,
        notes: list[str],
    ) -> tuple[float, float]:
        low, high = float(state.min()), float(state.max())
        if transform == "log":
            found = (low - LOG_MARGIN, high + LOG_MARGIN)
        else:
            found = (low / LINEAR_MARGIN if low > 0 else low, high * LINEAR_MARGIN if high > 0
                     else high + abs(high))  # fmt: skip
            if name in FRACTIONS:
                found = (max(found[0], 1e-4), min(found[1], 1.0))
        if lower is None or upper is None:
            notes.append(f"{name.lower()}: bounds [{found[0]:.4g}, {found[1]:.4g}] from the "
                         "deck's values (assumed)")  # fmt: skip
        chosen = (found[0] if lower is None else lower, found[1] if upper is None else upper)
        if not chosen[0] < chosen[1]:
            raise ConfigError(f"{name}: the lower bound must be below the upper bound")
        return chosen

    @staticmethod
    def _prior(
        name: str, transform: str, state: np.ndarray, grid: list[int], notes: list[str]
    ) -> dict[str, object]:
        mean = float(state.mean())
        variance = float(state.var())
        if variance <= 1e-12 * max(mean * mean, 1.0):
            std = ASSUMED_LOG_STD if transform == "log" else ASSUMED_RELATIVE_STD * abs(mean)
            variance = std * std
            notes.append(f"{name.lower()}: the deck's values are uniform; the prior's standard "
                         f"deviation {std:.3g} is assumed")  # fmt: skip
        nx, ny = grid[0], grid[1]
        if nx * ny > DENSE_LAYER_CELLS:
            notes.append(f"{name.lower()}: layers of {nx * ny} cells are too large for "
                         "gaussian_field; independent gaussian values (assumed: ask the "
                         "geomodeler for a correlated prior)")  # fmt: skip
            return {"name": "gaussian", "options": {"mean": mean, "std": math.sqrt(variance)}}
        major = max(1.0, round(RANGE_FRACTION * max(nx, ny), 1))
        notes.append(f"{name.lower()}: a Gaussian field around the deck's values, range "
                     f"{major:g} cells (assumed: ask the geomodeler to estimate it)")  # fmt: skip
        return {"name": "gaussian_field", "options": {
            "grid": grid, "mean": mean, "variance": variance, "major_range": major,
        }}  # fmt: skip


def method_section(method_name: str, iterations: int, localization: str) -> dict[str, object]:
    """Return the [method] section: GIES with rlmmac, or ES-MDA with equal inflations."""
    method: dict[str, object]
    if method_name == "gies":
        method = {"name": "gies", "strategy": {"name": "rlmmac"},
                  "max_iterations": iterations}  # fmt: skip
    else:
        inflation = [iterations] * iterations  # equal steps, summing to 1
        method = {"name": "esmda", "options": {"inflation": inflation},
                  "strategy": {"name": "approx"}}  # fmt: skip
    if localization != "none":
        method["localization"] = {"name": localization}
    return method


@converter_skills.register("opm_flow")
class OpmFlowSkill(DeckSkill):
    """OPM Flow decks (``.DATA``): read, and converted to APEX cases on ``opm_flow``."""

    name = "opm_flow"
    simulator = "opm_flow"
    guidance = (
        "OPM Flow reads Eclipse-format decks. Permeability (PERMX, often with PERMY/PERMZ "
        "copied from it) is usually matched as its logarithm (transform 'log'); porosity and "
        "net-to-gross as they are. Rates and pressures of wells under history control "
        "(WCONHIST, WCONINJH) are the usual observations. CO2STORE decks are OPM-only."
    )


@converter_skills.register("eclipse")
class EclipseSkill(DeckSkill):
    """Eclipse decks (E100/E300, ``.DATA``): read, and converted to APEX cases on ``eclipse``.

    The cases set ``licenses = 3``, the Eclipse runs at once; set it to the licences you have.
    """

    name = "eclipse"
    simulator = "eclipse"
    guidance = (
        "Eclipse decks (E100 black oil, E300 compositional) are converted the same way as "
        "OPM Flow's; choose opm_flow instead unless the user needs Eclipse, which uses "
        "licences. Permeability is matched as its logarithm; porosity as it is."
    )

    def _simulator_options(self) -> dict[str, object]:
        return {"deck_template_file": TEMPLATE, "licenses": 3}
