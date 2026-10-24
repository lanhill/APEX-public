"""The case-converter's deck skills: a deck read into facts, and back as a template.

The deck is written here: 4 x 3 x 2 cells, PERMX inline, PORO in an included file, PERMY
copied from PERMX, NTG by EQUALS, a history of a producer and an injector over DATES and
TSTEP. The template rendered with the deck's own values gives the deck's values back.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from apex.core.conversion import ConversionChoices, ObservationChoice, VariableChoice
from apex.core.errors import ConfigError
from apex.core.plugins import converter_skills
from apex.core.simulation import MemberInput
from apex.simulation.conversion.deck_reader import read_deck
from apex.simulation.conversion.deck_template import StateProperty, write_template
from apex.simulation.deck import render_deck
from apex.testing import SMALL_DECK_PERMX as PERMX
from apex.testing import SMALL_DECK_PORO as PORO
from apex.testing import write_small_deck as write_deck


def choices(**changes: object) -> ConversionChoices:
    base: dict[str, object] = {
        "name": "case",
        "variables": [VariableChoice(property="PERMX", transform="log"),
                      VariableChoice(property="PORO")],
        "observations": [ObservationChoice(vector=v, relative_error=0.1, floor=1.0)
                         for v in ("WOPR:PROD", "WWPR:PROD", "WWIR:INJ")],
    }  # fmt: skip
    return ConversionChoices.model_validate({**base, **changes})


def test_the_facts_of_a_deck(tmp_path: Path) -> None:
    facts = read_deck(write_deck(tmp_path), skill="opm_flow").facts

    assert facts.grid == [4, 3, 2]
    assert facts.cells == 24
    assert facts.units == "METRIC"
    assert facts.phases == ["OIL", "WATER"]
    assert facts.start == dt.date(2020, 1, 1)
    assert facts.report_dates == [dt.date(2020, 2, 1), dt.date(2020, 3, 1), dt.date(2020, 3, 31)]
    props = facts.properties
    assert props["PERMX"].choosable
    assert props["PERMX"].file == "CASE.DATA"
    assert props["PORO"].choosable
    assert props["PORO"].file == "props/poro.inc"
    assert props["PERMY"].source == "derived"
    assert props["PERMY"].derived_from == "PERMX"
    assert props["NTG"].source == "uniform"
    assert not props["NTG"].choosable
    assert not props["DX"].choosable  # geometry
    assert facts.wells["PROD"].kind == "producer"
    assert facts.wells["PROD"].cells == [[4, 3, 1], [4, 3, 2]]
    assert facts.wells["INJ"].kind == "injector"
    assert facts.includes == ["props/poro.inc", "pvt.inc"]


def test_a_history_rate_is_observed_at_the_end_of_its_step(tmp_path: Path) -> None:
    """A control set at d_k holds over [d_k, d_k+1): the summary reports it at d_k+1."""
    data = read_deck(write_deck(tmp_path), skill="opm_flow")
    feb, mar, end = dt.date(2020, 2, 1), dt.date(2020, 3, 1), dt.date(2020, 3, 31)

    assert data.history[(feb, "WOPR:PROD")] == 100.0
    assert data.history[(feb, "WBHP:PROD")] == 250.0
    assert data.history[(mar, "WOPR:PROD")] == 90.0
    assert (mar, "WBHP:PROD") not in data.history  # defaulted BHP: not observed
    assert (feb, "WWIR:INJ") not in data.history
    assert data.history[(mar, "WWIR:INJ")] == 200.0
    assert data.history[(end, "WWIR:INJ")] == 200.0  # controls hold until changed
    assert data.history[(end, "WOPR:PROD")] == 90.0


def test_a_history_csv_replaces_the_decks(tmp_path: Path) -> None:
    deck = write_deck(tmp_path)
    csv = tmp_path / "history.csv"
    csv.write_text("time,datatype,value\n2020-02-01,WOPR:PROD,123.5\n")

    data = read_deck(deck, skill="opm_flow", history=csv)

    assert data.history == {(dt.date(2020, 2, 1), "WOPR:PROD"): 123.5}
    assert [c.source for c in data.facts.candidates] == ["csv"]


def test_the_template_gives_the_decks_values_back(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    template = write_template(deck, {"PERMX": StateProperty("permx", "log"),
                                     "PORO": StateProperty("poro")})  # fmt: skip
    out = tmp_path / "draft"
    out.mkdir()
    (out / "deck.mako").write_text(template.text)
    for relative, source in template.files.items():
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        (out / relative).write_bytes(source.read_bytes())
    state = {"permx": np.log(np.array(PERMX)), "poro": np.array(PORO)}

    rendered = render_deck(out / "deck.mako", MemberInput(0, state), {})
    (out / "MEMBER.DATA").write_text(rendered)
    back = read_deck(out / "MEMBER.DATA", skill="opm_flow")

    assert sorted(template.files) == ["pvt.inc"]  # poro.inc is inlined
    np.testing.assert_allclose(back.arrays["PERMX"], PERMX, rtol=1e-9)
    np.testing.assert_allclose(back.arrays["PORO"], PORO, rtol=1e-9)
    assert back.facts.properties["PERMY"].derived_from == "PERMX"
    assert back.history == read_deck(deck, skill="opm_flow").history


def test_a_property_not_set_as_an_array_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="NTG"):
        write_template(write_deck(tmp_path), {"NTG": StateProperty("ntg")})


def test_a_template_is_not_a_deck(tmp_path: Path) -> None:
    deck = write_deck(tmp_path)
    deck.write_text(deck.read_text().replace("TITLE", "<%! import math %>\nTITLE"))

    with pytest.raises(ConfigError, match="Mako"):
        read_deck(deck, skill="opm_flow")


@pytest.mark.parametrize("skill", ["opm_flow", "eclipse"])
def test_the_skill_writes_a_case(tmp_path: Path, skill: str) -> None:
    deck = write_deck(tmp_path)
    converter = converter_skills.create(skill)
    facts = converter.read(deck)

    draft = converter.write_config(deck, facts, choices())

    config = draft.config
    assert config["simulator"] == {
        "name": skill,
        "options": {
            "deck_template_file": "deck.mako",
            **({"licenses": 3} if skill == "eclipse" else {}),
        },
    }
    variables = config["state"]["variables"]  # type: ignore[index]
    assert [v["name"] for v in variables] == ["permx", "poro"]
    assert variables[0]["grid"] == [4, 3, 2]
    assert variables[0]["size"] == 24
    assert config["method"]["name"] == "gies"  # type: ignore[index]
    rows = str(draft.files["observations.csv"]).splitlines()
    assert rows[0] == "time,datatype,value,variance"
    # 10 % of 100, above the floor of 1.
    assert f"2020-02-01,WOPR:PROD,100.0,{(0.1 * 100.0) ** 2!r}" in rows
    # 10 % of 5 is below the floor of 1: the floor.
    assert "2020-02-01,WWPR:PROD,5.0,1.0" in rows
    assert any("assumed" in note for note in draft.notes)


def test_a_vector_with_no_data_is_refused(tmp_path: Path) -> None:
    deck = write_deck(tmp_path)
    converter = converter_skills.create("opm_flow")
    wanted = choices(observations=[ObservationChoice(vector="WGPR:NONE", relative_error=0.1,
                                                     floor=1.0)])  # fmt: skip

    with pytest.raises(ConfigError, match="no observed"):
        converter.write_config(deck, converter.read(deck), wanted)


# ------------------------------------------------------------- the review's cases (2026-10-24)


def _edit(deck: Path, old: str, new: str) -> None:
    text = deck.read_text()
    assert old in text
    deck.write_text(text.replace(old, new, 1))


def _render(deck: Path, chosen: dict[str, StateProperty], state: dict[str, np.ndarray],
            out: Path) -> Path:  # fmt: skip
    template = write_template(deck, chosen)
    out.mkdir(parents=True)
    (out / "deck.mako").write_text(template.text)
    for relative, source in template.files.items():
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        (out / relative).write_bytes(source.read_bytes())
    (out / "MEMBER.DATA").write_text(render_deck(out / "deck.mako", MemberInput(0, state), {}))
    return out / "MEMBER.DATA"


def test_nested_includes_and_a_section_inside_an_include(tmp_path: Path) -> None:
    """A copied file has no includes of its own, and sections are seen inside includes."""
    deck = write_deck(tmp_path / "deck")
    folder = deck.parent
    (folder / "g").mkdir()
    # Relative to the deck's folder, as OPM and Eclipse resolve nested includes.
    (folder / "g" / "geom.inc").write_text("GRID\nINCLUDE\n 'g/tops.inc' /\n")
    (folder / "g" / "tops.inc").write_text("TOPS\n 12*2000 /\n")
    _edit(deck, "GRID\nDX", "INCLUDE\n 'g/geom.inc' /\nDX")
    _edit(deck, "TOPS\n 12*2000 /\n", "")
    state = {"permx": np.log(np.array(PERMX)) + 1.0}

    member = _render(deck, {"PERMX": StateProperty("permx", "log")}, state, tmp_path / "out")
    back = read_deck(member, skill="opm_flow")

    assert "g/tops.inc" in back.facts.includes  # copied, and found from the member's deck
    np.testing.assert_allclose(back.arrays["PERMX"], np.exp(state["permx"]), rtol=1e-9)
    assert not back.facts.unread


def test_cells_that_are_zero_stay_zero_without_reading_a_file(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    values = list(PERMX)
    values[3] = values[4] = 0.0
    _edit(deck, " ".join(repr(v) for v in PERMX), " ".join(repr(v) for v in values))
    flags = [v != 0 for v in values]
    from apex.simulation.conversion.deck_template import runs

    state = {"permx": np.full(24, np.log(7.0))}
    member = _render(deck, {"PERMX": StateProperty("permx", "log", runs(flags))}, state,
                     tmp_path / "out")  # fmt: skip
    back = read_deck(member, skill="opm_flow")

    expected = np.where(np.array(flags), 7.0, 0.0)
    np.testing.assert_allclose(back.arrays["PERMX"], expected, rtol=1e-9)
    assert "open(" not in (tmp_path / "out" / "deck.mako").read_text()


def test_a_property_changed_after_it_is_set_cannot_be_the_state(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    _edit(deck, "PROPS\n", "EDIT\nMULTIPLY\n 'PERMX' 2 /\n/\nPORO\n 24*0.2 /\nPROPS\n")

    props = read_deck(deck, skill="opm_flow").facts.properties

    assert not props["PERMX"].choosable
    assert "MULTIPLY" in (props["PERMX"].reason or "")
    assert not props["PORO"].choosable
    assert "more than once" in (props["PORO"].reason or "")


def test_an_equals_in_a_box_is_not_uniform(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    _edit(deck, "EQUALS\n 'NTG' 0.8 /", "EQUALS\n 'NTG' 0.8 /\n 'SWL' 0.2 1 2 1 2 1 1 /")

    props = read_deck(deck, skill="opm_flow").facts.properties

    assert props["SWL"].minimum is None
    assert "in a box" in (props["SWL"].reason or "")


def test_history_control_ends_with_wconprod_and_a_shut_well_is_zero(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    _edit(
        deck,
        "TSTEP\n 30 /\n",
        "WCONPROD\n 'PROD' 'OPEN' 'ORAT' 50 /\n/\nWELOPEN\n 'INJ' 'SHUT' /\n/\nTSTEP\n 30 /\n",
    )
    data = read_deck(deck, skill="opm_flow")
    end = dt.date(2020, 3, 31)

    assert (end, "WOPR:PROD") not in data.history
    assert data.history[(end, "WWIR:INJ")] == 0.0
    assert (end, "WBHP:INJ") not in data.history


def test_fractional_report_steps_keep_their_time(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    _edit(deck, "TSTEP\n 30 /\n", "TSTEP\n 0.5 0.5 /\nTSTEP\n 10 /\n")

    data = read_deck(deck, skill="opm_flow")

    assert data.facts.report_dates[-3:] == [dt.date(2020, 3, 1), dt.date(2020, 3, 2),
                                            dt.date(2020, 3, 12)]  # fmt: skip


def test_a_file_may_be_included_twice(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    (deck.parent / "w.inc").write_text("WELOPEN\n 'INJ' 'OPEN' /\n/\n")
    _edit(deck, "TSTEP\n 30 /\n", "INCLUDE\n 'w.inc' /\nTSTEP\n 30 /\nINCLUDE\n 'w.inc' /\n")

    data = read_deck(deck, skill="opm_flow")

    assert data.facts.includes.count("w.inc") == 2


def test_an_included_file_named_as_the_drafts_own_is_refused(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    (deck.parent / "deck.mako").write_text("PVDO\n 1 1 1 /\n/\n")
    _edit(deck, "PROPS\n", "PROPS\nINCLUDE\n 'deck.mako' /\n")
    converter = converter_skills.create("opm_flow")

    with pytest.raises(ConfigError, match="converter's own"):
        converter.write_config(deck, converter.read(deck), choices())


def test_a_box_redefinition_and_an_equals_before_an_array(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    _edit(deck, "COPY\n", "BOX\n 1 1 1 1 1 1 /\nPERMX\n 5 /\nENDBOX\nEQUALS\n 'SWL' 0.1 /\n/\n"
          "SWL\n 24*0.2 /\nCOPY\n")  # fmt: skip

    props = read_deck(deck, skill="opm_flow").facts.properties

    assert not props["PERMX"].choosable
    assert "box" in (props["PERMX"].reason or "")
    assert props["SWL"].choosable  # the array overrides the EQUALS
    assert props["SWL"].source == "array"


def test_shutting_connections_does_not_shut_the_well(tmp_path: Path) -> None:
    deck = write_deck(tmp_path / "deck")
    _edit(deck, "TSTEP\n 30 /\n", "WELOPEN\n 'PROD' 'SHUT' 4 3 1 /\n/\nTSTEP\n 30 /\n")

    data = read_deck(deck, skill="opm_flow")

    assert data.history[(dt.date(2020, 3, 31), "WOPR:PROD")] == 90.0
    assert any("connections" in u for u in data.facts.unread)
