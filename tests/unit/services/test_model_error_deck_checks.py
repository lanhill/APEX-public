"""The composition's deck checks of a model-error model: INIT and the phases (2026-11-08)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from apex.core.errors import ConfigError
from apex.services.composition import check_model_error_summary


def _check(
    tmp_path: Path, runspec: str, phases: list[str], grid: str = "INIT", simulator: str = "opm_flow"
) -> None:
    deck = tmp_path / "deck.mako"
    deck.write_text(f"RUNSPEC\n{runspec}\nGRID\n{grid}\nSOLUTION\nSUMMARY\nSCHEDULE\nEND\n")
    config = SimpleNamespace(
        simulator=SimpleNamespace(name=simulator, options={"deck_template_file": str(deck)}),
        model_error=SimpleNamespace(name="pg_lstm"),
    )
    model = SimpleNamespace(phases=tuple(phases), summary_vectors=[])
    check_model_error_summary(cast(Any, config), cast(Any, model))


def test_a_deck_without_init_is_refused(tmp_path: Path) -> None:
    """The case converter's CO2STORE deck had no INIT: every member failed after its run."""
    with pytest.raises(ConfigError, match="GRID section has no INIT"):
        _check(tmp_path, "OIL\nWATER", ["oil", "water"], grid="DX\n 1*10 /")
    _check(tmp_path, "OIL\nWATER", ["oil", "water"])
    _check(tmp_path, "OIL\nWATER", ["oil", "water"], grid="% if a:\nINIT\n% endif")  # unknown


@pytest.mark.parametrize(
    ("runspec", "phases", "message"),
    [
        # a phase the deck lacks: no density to read
        (
            "OIL\nWATER",
            ["oil", "water", "gas"],
            r"include \['gas'\].*set phases = \['oil', 'water'\]",
        ),
        # OPM's gas-water CO2STORE (and Eclipse 300's, naming no phase): water and gas
        ("CO2STORE\nWATER\nGAS", ["oil", "water"], r"include \['oil'\].*a CO2STORE deck"),
        ("CO2STORE\nCOMPS\n 3 /", ["oil", "water"], r"\['water', 'gas'\]"),
        # OPM's older oil-gas CO2STORE: the brine is the oil phase
        ("CO2STORE\nOIL\nGAS", ["water", "gas"], r"include \['water'\].*\['oil', 'gas'\]"),
        # a CO2STORE deck's gas left out: the CO2 is what the model reads
        ("CO2STORE\nWATER\nGAS", ["water"], "a CO2STORE deck's gas"),
        # leaving out the deck's oil: its features would be read as 0
        ("OIL\nWATER\nGAS", ["water", "gas"], "the deck has oil"),
    ],
)
def test_phases_the_deck_contradicts_are_refused(
    tmp_path: Path, runspec: str, phases: list[str], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _check(tmp_path, runspec, phases)


@pytest.mark.parametrize(
    ("runspec", "phases"),
    [
        ("OIL\nWATER\nGAS", ["oil", "water", "gas"]),
        ("OIL\nWATER\nGAS", ["oil", "water"]),  # leaving out gas: as before
        ("CO2STORE\nWATER\nGAS", ["water", "gas"]),
        ("CO2STORE\nCOMPS\n 3 /", ["water", "gas"]),
        ("CO2STORE\nWATER\nCOMPS\n 3 /", ["water", "gas"]),  # Eclipse 300's form: water, gas
        ("CO2STORE\nOIL\nGAS", ["oil", "gas"]),
        ("WATER\nCOMPS\n 9 /", ["oil", "water"]),  # another compositional deck: not judged
        ("INCLUDE\n 'phases.inc' /", ["water", "gas"]),  # phases unknown: not judged
    ],
)
def test_phases_that_fit_or_cannot_be_judged_are_taken(
    tmp_path: Path, runspec: str, phases: list[str]
) -> None:
    _check(tmp_path, runspec, phases)


def test_eclipse_300s_co2store_is_refused_for_every_model(tmp_path: Path) -> None:
    """Its compositional restarts name the densities DENW, DENG (the review: pg_lstm too)."""
    with pytest.raises(ConfigError, match="Eclipse 300's CO2STORE"):
        _check(tmp_path, "CO2STORE\nCOMPS\n 3 /", ["water", "gas"], simulator="eclipse")
    _check(tmp_path, "CO2STORE\nCOMPS\n 3 /", ["water", "gas"], simulator="intersect")
    with pytest.raises(ConfigError, match="Eclipse 300's GASWAT"):
        _check(tmp_path, "WATER\nGASWAT\nCOMPS\n 3 /", ["water", "gas"], simulator="eclipse")
