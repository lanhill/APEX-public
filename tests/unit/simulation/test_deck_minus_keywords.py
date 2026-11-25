"""Keywords ending in "-" (MULTZ- and the like) are keywords, not data (2026-11-25)."""

from pathlib import Path

import pytest

from apex.simulation.conversion.deck_reader import walk
from apex.simulation.deck import (
    DASHED_KEYWORDS,
    KeywordLine,
    _keyword_records,
    is_keyword_name,
)


def test_a_minus_keyword_is_its_own_with_its_own_records() -> None:
    """MULTZ-'s records were taken as MULTZ's before: a reader of MULTZ got them too."""
    text = "GRID\nMULTZ\n 4*0.5 /\nMULTZ-\n 4*0.1 /\nMULTTHT-\n 4*1 /\nPORO\n 4*0.2 /\n"
    assert _keyword_records(text) == [
        ("GRID", []),
        ("MULTZ", ["4*0.5"]),
        ("MULTZ-", ["4*0.1"]),
        ("MULTTHT-", ["4*1"]),
        ("PORO", ["4*0.2"]),
    ]


def test_keyword_names_follow_opm_s_rule(tmp_path: Path) -> None:
    """A letter, then letters, digits, "_", "-" or "+", within 8 columns (opm-common's
    ParserKeyword::validDeckName), a dashed name only if OPM has it. Before, "-" was refused,
    and PVT-M was data."""
    deck = tmp_path / "D.DATA"
    deck.write_text(
        "RUNSPEC\nDIMENS\n 2 2 1 /\nPVT-M\nGRID\nMULTX-\n 4*1 /\nMULTY-\n 4*1 /\nMULTZ-\n 4*0 /\n"
    )
    assert {k.keyword for k in walk(deck)} >= {"PVT-M", "MULTX-", "MULTY-", "MULTZ-"}
    # Not keywords, so their records stay GRID's: nine columns, a leading "-", and lower case
    # (APEX's choice: OPM upper-cases a line's token first, so it would read multz- as MULTZ-).
    for name in ("MULTTHTX-", "-MULTZ", "multz-"):
        found = _keyword_records(f"GRID\nPORO\n 1 /\n{name}\n 2 /\n")
        assert [k for k, _ in found] == ["GRID", "PORO"], name
        assert found[1][1] == ["1", f"{name} 2"], name


def test_a_comment_after_a_keyword_is_cut_as_opm_does() -> None:
    """OPM strips a comment from the first "--": MULTZ---note is MULTZ, MULTZ- --note MULTZ-."""
    assert [k for k, _ in _keyword_records("GRID\nMULTZ---note\n 1 /\n")] == ["GRID", "MULTZ"]
    assert [k for k, _ in _keyword_records("GRID\nMULTZ- --note\n 1 /\n")] == ["GRID", "MULTZ-"]


def test_a_dashed_token_is_a_keyword_only_if_opm_has_it() -> None:
    """OPM knows its keywords' records; APEX's reader knows OPM's dashed names instead:
    MULTZ- and PVT-M are keywords, an unquoted well name OP-1 or A+B is data, also where a
    record starts (after a keyword, or after a record's "/")."""
    for name in ("MULTZ-", "PVT-M", "SWCRX-", "PORO"):
        assert is_keyword_name(name), name
    for name in ("OP-1", "A+B", "MUL-TZ", "H-K"):
        assert not is_keyword_name(name), name
    found = _keyword_records("SCHEDULE\nWELOPEN\n 'P1' SHUT /\nOP-1\n SHUT /\n/\n")
    assert found == [("SCHEDULE", []), ("WELOPEN", ["'P1' SHUT", "OP-1 SHUT"])]
    found = _keyword_records("SCHEDULE\nWELOPEN\nA+B\n SHUT /\n/\n")
    assert found == [("SCHEDULE", []), ("WELOPEN", ["A+B SHUT"])]
    assert {"MULTX-", "MULTY-", "MULTZ-", "MULTR-", "MULTTHT-", "PVT-M"} <= DASHED_KEYWORDS
    # Every "name" and "deck_names" entry with "-" or "+" in opm-common 2026.04 share/keywords:
    # 117 (H-K there is only an item's default value, WSEGDFMD's INCLINATION_FACTOR).
    assert len(DASHED_KEYWORDS) == 117
    found = _keyword_records("SCHEDULE\nWSEGDFMD\n 'W1'\nH-K\n NO /\n/\n")
    assert found == [("SCHEDULE", []), ("WSEGDFMD", ["'W1' H-K NO"])]


def test_a_pvt_m_deck_is_read_in_its_units(tmp_path: Path) -> None:
    """PVT-M was data before (its "-"): a PVT-M deck's units were never seen."""
    from apex.simulation.conversion.deck_reader import read_deck

    deck = tmp_path / "D.DATA"
    deck.write_text("RUNSPEC\nDIMENS\n 2 2 1 /\nOIL\nWATER\nPVT-M\nGRID\nPORO\n 4*0.2 /\n")
    assert read_deck(deck, skill="test").facts.units == "PVT-M"


def test_a_dashed_token_inside_an_unfinished_record_is_data() -> None:
    """An unquoted well name such as A- alone on a record's continuation line stays an item."""
    found = _keyword_records("SCHEDULE\nWELOPEN\n 'P1'\nA-\n SHUT /\n/\n")
    assert found == [("SCHEDULE", []), ("WELOPEN", ["'P1' A- SHUT"])]
    # A plain name there still starts a keyword, as before.
    assert [k for k, _ in _keyword_records("GRID\nPORO\n 1\nPERMX\n 2 /\n")] == [
        "GRID",
        "PORO",
        "PERMX",
    ]


def test_a_minus_array_is_read_but_not_choosable(tmp_path: Path) -> None:
    """Its variable would be named ``multz-``, which no template loop can use."""
    from apex.simulation.conversion.deck_reader import read_deck

    deck = tmp_path / "D.DATA"
    deck.write_text(
        "RUNSPEC\nDIMENS\n 2 2 1 /\nOIL\nWATER\nGRID\nDX\n 4*10 /\nDY\n 4*10 /\n"
        "DZ\n 4*2 /\nTOPS\n 4*1000 /\nPORO\n 4*0.2 /\nPERMX\n 4*100 /\nMULTZ-\n 4*0.5 /\n"
    )
    facts = read_deck(deck, skill="test").facts
    minus = facts.properties["MULTZ-"]
    assert not minus.choosable
    assert "no valid variable name" in (minus.reason or "")
    assert facts.properties["PORO"].choosable


def test_a_deck_with_dashed_keywords_is_cut_at_a_date() -> None:
    """stop_deck_at reads keywords with the same rule: MULTZ- and an unquoted OP-1 before the
    cut leave the report dates and the cut as they are."""
    import datetime as dt

    from apex.simulation.deck import stop_deck_at

    deck = (
        "RUNSPEC\nSTART\n 1 JAN 2020 /\nGRID\nMULTZ-\n 4*0.5 /\n"
        "SCHEDULE\nWELOPEN\nOP-1\n SHUT /\n/\n"
        "DATES\n 1 FEB 2020 /\n/\nDATES\n 1 MAR 2020 /\n/\n"
    )
    cut = stop_deck_at(deck, dt.date(2020, 2, 1))
    assert cut.endswith("DATES\n 1 FEB 2020 /\n/\n\nEND\n")
    assert "MAR" not in cut


def test_a_dashed_keyword_in_a_mako_block_is_seen() -> None:
    from apex.simulation.deck import _keywords_in_mako_blocks

    text = "GRID\n% if flag:\nMULTZ-\n 4*0.5 /\nOP-1\n% endif\n"
    assert _keywords_in_mako_blocks(text) == {"MULTZ-"}


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ("PORO", "PORO"),
        ("PORO  ", "PORO"),
        ("PORO -- a comment", "PORO"),
        ("PORO--c", "PORO"),
        ("PORO--", "PORO"),
        ("PORO---", "PORO"),
        ("GRID----------", "GRID"),
        ("MULTZ-", "MULTZ-"),
        ("MULTZ- -- a comment", "MULTZ-"),
        ("MULTZ---note", "MULTZ"),
        ("MULTZ--", "MULTZ"),
        ("ROCKTAB--", "ROCKTAB"),
        ("PVT-M", "PVT-M"),
        ("OP-1", None),
        ("OP-1 -- a well", None),
        ("A+B", None),
        ("H-K", None),
        ("PORO 1 /", None),
        ("1*0.5", None),
        ("multz-", None),
        ("-- only a comment", None),
    ],
)
def test_keyword_line_cuts_the_comment_first_as_opm_does(line: str, name: str | None) -> None:
    """The comment goes from the first "--" (OPM's strip_comments), then the rest must be one
    name (OPM's rule, a dashed name only if OPM has it)."""
    found = KeywordLine().match(line)
    assert (found.group(1) if found else None) == name


def test_e300_region_to_region_vectors_are_keywords_and_well_names_are_not() -> None:
    """OPM's E300 region-to-region pattern is kept (RGFR-, ROFT+ABC); its catch-all patterns
    (WU.+, SU.+, AA.+ ...) are not, so unquoted well names alone on a line stay data."""
    for name in ("RGFR-", "ROFT+", "RWFR-", "ROFT+ABC", "BFLOWI-", "BVELWK-"):
        assert is_keyword_name(name), name
    for name in ("OP-1", "A+B", "RGFX-", "ROFR-A", "WU-1", "FU-2", "ALPHA-1", "SUB-1", "RUN-2"):
        assert not is_keyword_name(name), name
    found = _keyword_records("SUMMARY\nRGFR-\n 1 2 /\n/\nFOPR\n")
    assert [k for k, _ in found] == ["SUMMARY", "RGFR-", "FOPR"]
    # A known divergence from OPM, which reads a lone undashed word inside a record of known
    # size as data: APEX's tokeniser has no record counts, so OP1 alone starts a keyword.
    found = _keyword_records("SUMMARY\nWOPR\nOP1\n/\n")
    assert [k for k, _ in found] == ["SUMMARY", "WOPR", "OP1"]
