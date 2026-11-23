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


def test_upscale_coarsens_every_member_whatever_its_id(tmp_path: Path) -> None:
    """Amended: one coarsening for every member (an optimization's ids
    count its journal's rows, so no finite level covers them)."""
    fine, _ = _templates(tmp_path)
    upscale = {"method": "uniform", "dim": [8, 8], "max_trunc": 0.5, "wells": [[2, 2]],
               "radius": [1]}  # fmt: skip
    levels = FidelityLevels(None, fine, upscale)
    records = coarsen(np.ones(64), (8, 8), max_trunc=0.5, wells=[[2, 2]], radius=[1])
    assert records
    for member_id in (0, 7, 10**6, -1):
        template, values = levels.for_member(MemberInput(member_id, {"permx": np.ones(64)}))
        assert (template, values) == (fine, {"level": 0, "coarse": records})
    assert levels.upscales
    assert levels.templates() == [fine]
    # haar follows the member's own field, so two members may get different grids.
    haar = FidelityLevels(
        None, fine, {"method": "haar", "dim": [8, 8], "max_trunc": 0.1, "state": "permx"}
    )
    field = np.ones(64)
    field[:8] = 50.0
    _, own = haar.for_member(MemberInput(3, {"permx": field}))
    assert own["coarse"] == coarsen(field, (8, 8), method="haar", max_trunc=0.1)


def test_upscale_is_validated_and_excludes_levels(tmp_path: Path) -> None:
    fine, _ = _templates(tmp_path)
    with pytest.raises(ValueError, match="one or the other"):
        FidelityLevels([{"first": 0, "size": 2}], fine, {"dim": [4, 4], "max_trunc": 1.0})
    with pytest.raises(ValueError, match="upscale needs dim and max_trunc"):
        FidelityLevels(None, fine, {"dim": [4, 4]})
    with pytest.raises(ValueError, match="state it follows"):
        FidelityLevels(None, fine, {"method": "haar", "dim": [4, 4], "max_trunc": 1.0})
    with pytest.raises(ValueError, match="one radius per well"):
        FidelityLevels(None, fine, {"dim": [4, 4], "max_trunc": 1.0, "wells": [[1, 1]]})
    with pytest.raises(ValueError, match="must be a table"):
        FidelityLevels(None, fine, [4, 4])  # type: ignore[arg-type]


def test_the_simulators_upscale_option(tmp_path: Path) -> None:
    fine, _ = _templates(tmp_path)
    upscale = {"dim": [4, 4], "max_trunc": 1.0}
    with pytest.raises(ValueError, match="does not support COARSEN"):
        OPMFlowSimulator(fine, upscale=upscale)
    eclipse = EclipseSimulator(fine, licenses=3, upscale=upscale)
    template, values = eclipse.levels.for_member(MemberInput(12345, {"permx": np.ones(16)}))
    deck = render_deck(template, MemberInput(12345, {"permx": np.ones(16)}), values)
    assert "COARSEN" in deck
    assert "-- level 0" in deck
    # Without it, the deck is the fine one: no records.
    plain = EclipseSimulator(fine, licenses=3)
    _, none = plain.levels.for_member(MemberInput(12345, {"permx": np.ones(16)}))
    assert none == {"level": -1, "coarse": []}
