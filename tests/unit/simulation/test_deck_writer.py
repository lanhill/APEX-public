"""Synthetic decks from a description: what the writer writes reads back."""

import datetime as dt
import itertools
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from apex.core.simulation import MemberInput
from apex.domain.deck_spec import DeckSpec, FluidSpec, GridSpec, WellSpec, five_spot
from apex.simulation.conversion.deck_reader import read_deck
from apex.simulation.conversion.deck_writer import swof, write_deck
from apex.simulation.deck import render_deck


def test_the_five_spot_deck_reads_back(tmp_path: Path) -> None:
    spec = five_spot("box", n=11, layers=2)
    (tmp_path / "deck.mako").write_text(write_deck(spec))
    cells = 11 * 11 * 2
    state = {"permx": np.full(cells, np.log(100.0)), "poro": np.full(cells, 0.2)}

    rendered = render_deck(tmp_path / "deck.mako", MemberInput(0, state), {})
    (tmp_path / "BOX.DATA").write_text(rendered)
    facts = read_deck(tmp_path / "BOX.DATA", skill="opm_flow").facts

    assert facts.grid == [11, 11, 2]
    assert facts.properties["PERMX"].maximum == pytest.approx(100.0)
    assert facts.properties["PERMZ"].derived_from == "PERMX"
    assert facts.start == dt.date(2025, 1, 1)
    assert facts.report_dates == spec.schedule.report_dates()
    assert {w: f.kind for w, f in facts.wells.items()} == {
        "PROD1": "producer", "PROD2": "producer", "PROD3": "producer", "PROD4": "producer",
        "INJ": "injector"}  # fmt: skip
    assert facts.wells["INJ"].cells == [[6, 6, 1], [6, 6, 2]]


def test_swof_runs_from_connate_water_to_one() -> None:
    rows = swof(FluidSpec(swc=0.2, sorw=0.25))

    sw = [r[0] for r in rows]
    assert sw[0] == pytest.approx(0.2)
    assert sw[-1] == 1.0
    assert all(b > a for a, b in itertools.pairwise(sw))
    assert rows[0][1] == 0.0  # no water flows at connate water
    assert rows[-2][2] == pytest.approx(0.0)  # no oil flows at residual oil


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"wells": [WellSpec(name="P", kind="producer", i=30, j=1, control="bhp", target=100,
                             bhp_limit=100)]}, "outside"),
        ({"wells": [WellSpec(name="I", kind="injector", i=1, j=1, control="water_rate",
                             target=100, bhp_limit=300)]}, "producer"),
    ],
)  # fmt: skip
def test_a_description_is_checked(change: dict[str, object], message: str) -> None:
    base = five_spot().model_dump()
    with pytest.raises(ValidationError, match=message):
        DeckSpec.model_validate(
            {
                **base,
                **{
                    k: [w.model_dump() for w in v]  # type: ignore[attr-defined]
                    for k, v in change.items()
                },
            }
        )
    with pytest.raises(ValidationError, match="controlled by"):
        WellSpec(name="P", kind="producer", i=1, j=1, control="water_rate", target=1, bhp_limit=1)
    with pytest.raises(ValidationError, match="more than"):
        GridSpec(nx=1000, ny=1000, nz=2)


def test_the_co2_deck_reads_back_with_its_keywords(tmp_path: Path) -> None:
    from apex.domain.deck_spec import co2_injection
    from apex.simulation.conversion.deck_writer import well_vectors

    spec = co2_injection("co2", n=11, layers=3)
    template = write_deck(spec)
    (tmp_path / "deck.mako").write_text(template)
    cells = 11 * 11 * 3
    state = {"permx": np.full(cells, np.log(200.0)), "poro": np.full(cells, 0.25)}
    rendered = render_deck(tmp_path / "deck.mako", MemberInput(0, state), {})
    (tmp_path / "CO2.DATA").write_text(rendered)
    facts = read_deck(tmp_path / "CO2.DATA", skill="opm_flow").facts

    for keyword in ("CO2STORE", "DISGASW", "SALINITY", "RTEMPVD", "SGFN", "SWFN"):
        assert f"\n{keyword}\n" in rendered
    assert " 'INJ' 'GAS' 'OPEN' 'RATE' 730000 1* 400 /" in rendered
    assert facts.grid == [11, 11, 3]
    assert facts.wells["INJ"].kind == "injector"
    assert facts.wells["INJ"].cells == [[6, 6, 3]]
    assert well_vectors(spec)[:2] == ["WGIR:INJ", "WBHP:INJ"]
    assert "WGPR:PROD1" in well_vectors(spec)


def test_the_co2_saturation_tables_span_their_ranges() -> None:
    from apex.domain.deck_spec import Co2BrineSpec
    from apex.simulation.conversion.deck_writer import sgfn, swfn

    fluid = Co2BrineSpec(preset="co2_brine", swc=0.2, sgc=0.05)
    gas, water = sgfn(fluid), swfn(fluid)

    assert gas[0] == (0.0, 0.0, 0.0)
    assert gas[1] == (0.05, 0.0, 0.0)
    assert gas[-1] == (1.0, 1.0, 0.0)
    assert [g[0] for g in gas] == sorted({g[0] for g in gas})  # increasing, no repeats
    assert gas[-2][0] == pytest.approx(0.8)  # 1 - swc: the most CO2 drainage reaches
    assert water[0] == (0.2, 0.0, 0.0)
    assert water[-1] == pytest.approx((1.0, 1.0, 0.0))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fluid": {"preset": "co2_brine"}, "wells": "producers only"}, "CO2 injector"),
        ({"fluid": {"preset": "co2_brine"}, "control": "water_rate"}, "inject CO2"),
        ({"control": "gas_rate"}, "gas_rate is co2_brine's"),
        ({"fluid": {"preset": "co2_brine", "swc": 0.6, "sgc": 0.5}}, "mobile CO2"),
    ],
)  # fmt: skip
def test_a_co2_case_that_cannot_run_is_refused(change: dict[str, object], message: str) -> None:
    raw = five_spot("box", n=11, layers=2).model_dump()
    if "fluid" in change:
        raw["fluid"] = change["fluid"]
    if change.get("wells") == "producers only":
        raw["wells"] = [w for w in raw["wells"] if w["kind"] == "producer"]
    if "control" in change:
        injector = next(w for w in raw["wells"] if w["kind"] == "injector")
        injector["control"] = change["control"]

    with pytest.raises(ValidationError, match=message):
        DeckSpec.model_validate(raw)


def test_eclipse_refuses_a_co2_case() -> None:
    from apex.core.errors import ConfigError
    from apex.core.plugins import converter_skills
    from apex.domain.deck_spec import co2_injection

    with pytest.raises(ConfigError, match="E300's compositional CO2STORE"):
        converter_skills.create("eclipse").create(co2_injection())
