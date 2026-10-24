"""Fidelity levels in the deck adapters: per-level decks and COARSEN records."""

from pathlib import Path

import numpy as np
import pytest

from apex.core.simulation import MemberInput
from apex.simulation.deck import FidelityLevels, render_deck
from apex.simulation.eclipse import EclipseSimulator
from apex.simulation.opm_flow import OPMFlowSimulator
from apex.simulation.upscaling import coarsen

TEMPLATE = """RUNSPEC
-- level ${level}
% if coarse:
COARSEN
% for r in coarse:
${' '.join(str(v) for v in r)} /
% endfor
/
% endif
"""


def _templates(tmp_path: Path) -> tuple[Path, Path]:
    fine, coarse = tmp_path / "FINE.mako", tmp_path / "COARSE.mako"
    fine.write_text(TEMPLATE)
    coarse.write_text("-- a coarse deck, level ${level}\n")
    return fine, coarse


def test_each_level_gets_its_deck_and_coarsening(tmp_path: Path) -> None:
    fine, coarse = _templates(tmp_path)
    upscale = {
        "method": "uniform",
        "dim": [8, 8],
        "max_trunc": 0.5,
        "wells": [[2, 2]],
        "radius": [1],
    }
    levels = FidelityLevels(
        [
            {"first": 0, "size": 3, "upscale": upscale},
            {"first": 3, "size": 2, "deck_template_file": str(coarse)},
            {"first": 5, "size": 4},
        ],
        fine,
    )
    state = {"permx": np.ones(64)}

    template0, values0 = levels.for_member(MemberInput(1, state))
    template1, values1 = levels.for_member(MemberInput(4, state))
    template2, values2 = levels.for_member(MemberInput(8, state))
    _, outside = levels.for_member(MemberInput(-1, state))  # e.g. a mean-model run

    records = coarsen(np.ones(64), (8, 8), max_trunc=0.5, wells=[[2, 2]], radius=[1])
    assert (template0, values0) == (fine, {"level": 0, "coarse": records})
    assert (template1, values1["level"]) == (coarse, 1)
    assert (template2, values2) == (fine, {"level": 2, "coarse": []})
    assert outside == {"level": -1, "coarse": []}
    deck = render_deck(template0, MemberInput(1, state), values0)
    assert "COARSEN" in deck
    assert f"{' '.join(map(str, records[0]))} /" in deck
    assert levels.upscales


def test_levels_are_validated(tmp_path: Path) -> None:
    fine, _ = _templates(tmp_path)
    with pytest.raises(ValueError, match="must start at member 2"):
        FidelityLevels([{"first": 0, "size": 2}, {"first": 3, "size": 2}], fine)
    with pytest.raises(ValueError, match="dim and max_trunc"):
        FidelityLevels([{"first": 0, "size": 2, "upscale": {"dim": [4, 4]}}], fine)
    with pytest.raises(ValueError, match="state it follows"):
        FidelityLevels(
            [{"first": 0, "size": 2, "upscale": {"method": "haar", "dim": [4, 4], "max_trunc": 1}}],
            fine,
        )
    with pytest.raises(ValueError, match="deck template not found"):
        FidelityLevels([{"first": 0, "size": 2, "deck_template_file": "nope.mako"}], fine)


def test_opm_flow_refuses_coarsening_but_takes_coarse_decks(tmp_path: Path) -> None:
    fine, coarse = _templates(tmp_path)
    upscale = {"dim": [4, 4], "max_trunc": 1.0}
    with pytest.raises(ValueError, match="does not support COARSEN"):
        OPMFlowSimulator(fine, levels=[{"first": 0, "size": 2, "upscale": upscale}])
    simulator = OPMFlowSimulator(
        fine, levels=[{"first": 0, "size": 2, "deck_template_file": coarse}]
    )
    assert "coarse deck, level 0" in simulator._render(MemberInput(1, {"permx": np.ones(16)}))
    eclipse = EclipseSimulator(
        fine, licenses=3, levels=[{"first": 0, "size": 2, "upscale": upscale}]
    )
    assert eclipse.levels.upscales
