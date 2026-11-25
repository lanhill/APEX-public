"""Reading a deck template's grid and well completions (the chat's case_info tool)."""

from pathlib import Path

import pytest

from apex.simulation import deck as deck_module
from apex.simulation.deck import deck_keywords, deck_layout, schedule_summary

TEMPLATE = """\
<%!
import numpy as np
%>
RUNSPEC
TITLE
  A deck whose title has no slash

DIMENS
-- nx ny nz
   10 12 3 /

GRID
INCLUDE
 '${case_dir}/include/GEOMETRY.GRDECL' /

PORO
% for v in poro:
${v}
% endfor
/

SCHEDULE
INCLUDE
 '${case_dir}/include/SCHEDULE.SCH' /
END
"""

SCHEDULE = """\
WELSPECS
 'P1' G1 2 3 2000 OIL /
 INJ G1 9 10 1* WATER /  -- a comment
/
COMPDAT
-- name I J K1 K2
 'P1' 2* 1 3 OPEN 2* 0.25 /
 INJ 9 10
     2 2 OPEN 2* 0.25 /
 P2 ${i} 4 1 1 OPEN /
/
"""


def test_the_grid_and_each_wells_completions_are_read(tmp_path: Path) -> None:
    (tmp_path / "include").mkdir()
    (tmp_path / "deck.mako").write_text(TEMPLATE)
    (tmp_path / "include" / "GEOMETRY.GRDECL").write_text("COORD\n 1 2 3 /\n")
    (tmp_path / "include" / "SCHEDULE.SCH").write_text(SCHEDULE)

    layout = deck_layout(tmp_path / "deck.mako")

    assert layout["grid"] == [10, 12, 3]
    assert layout["wells"] == {
        "P1": [[2, 3, 1], [2, 3, 2], [2, 3, 3]],  # I and J defaulted: from WELSPECS
        "INJ": [[9, 10, 2]],  # a record over two lines
    }
    assert layout["skipped"] == ["P2 ${i} 4 1 1 OPEN"]  # a Mako expression: not guessed
    assert [Path(str(f)).name for f in layout["files"]] == [  # type: ignore[attr-defined]
        "deck.mako", "GEOMETRY.GRDECL", "SCHEDULE.SCH",
    ]  # fmt: skip


def test_a_long_record_is_read_in_linear_time() -> None:
    """2026-10-06: the scan rescanned a record's text at every line, so one property array
    of 60,000 lines (Brugge's NTG include) took many minutes; case_info hung on it."""
    import time

    from apex.simulation.deck import _keyword_records

    values = "\n".join("4.840079e-02" for _ in range(200_000))
    text = f"NTG\n{values}\n/\nDIMENS\n 10 20 3 /\n"
    start = time.perf_counter()
    found = _keyword_records(text)
    assert time.perf_counter() - start < 30.0  # about 1 s; quadratic: hours at this size
    assert [k for k, _ in found] == ["NTG", "DIMENS"]
    assert len(found[0][1]) == 1  # one record
    assert found[0][1][0].count("4.840079e-02") == 200_000
    assert found[1][1] == ["10 20 3"]


def test_a_quote_carries_over_lines_and_hides_its_slash() -> None:
    from apex.simulation.deck import _keyword_records

    text = "WELSPECS\n 'P/1' 'G1'\n  'a / b' 5 6 /\n /\nCOMPDAT\n 'P1' 2* /\n /\n"
    found = dict(_keyword_records(text))
    assert found["WELSPECS"] == ["'P/1' 'G1' 'a / b' 5 6"]
    assert found["COMPDAT"] == ["'P1' 2*"]
    open_quote = "TITLE\n 'a quote\n spanning / lines' /\n"
    assert dict(_keyword_records(open_quote))["TITLE"] == ["'a quote spanning / lines'"]


def test_the_includes_are_read_from_case_dir(tmp_path: Path) -> None:
    """The deck scans follow the simulator's ``case_dir`` (a copied example's folder)."""
    (tmp_path / "deck.mako").write_text(TEMPLATE)
    copy = tmp_path / "copy"
    (copy / "include").mkdir(parents=True)
    (copy / "include" / "SCHEDULE.SCH").write_text(SCHEDULE)

    assert deck_layout(tmp_path / "deck.mako")["wells"] == {}  # nothing beside the template
    wells = deck_layout(tmp_path / "deck.mako", copy)["wells"]
    assert isinstance(wells, dict)
    assert set(wells) == {"P1", "INJ"}


def test_a_schedule_after_many_includes_is_read(tmp_path: Path) -> None:
    """A deck of more than 50 includes still has its last one read (Norne's schedule).

    The reader stopped at 50 files and dropped the rest without a word, so Norne's schedule,
    included last after 49 others, gave no wells.
    """
    (tmp_path / "inc").mkdir()
    for i in range(60):
        (tmp_path / "inc" / f"part{i}.inc").write_text(f"-- part {i}\n")
    includes = "".join(f"INCLUDE\n '${{case_dir}}/inc/part{i}.inc' /\n" for i in range(60))
    (tmp_path / "inc" / "SCHEDULE.SCH").write_text(
        "DATES\n 1 JAN 2000 /\n/\nWCONHIST\n 'P1' OPEN ORAT 100 5 /\n/\n"
        "WCONINJE\n 'I1' WATER OPEN RATE 300 /\n/\n"
    )
    template = tmp_path / "deck.mako"
    template.write_text(
        "RUNSPEC\nSTART\n 1 DEC 1999 /\nGRID\n"
        + includes
        + "SCHEDULE\nINCLUDE\n '${case_dir}/inc/SCHEDULE.SCH' /\nEND\n"
    )
    summary = schedule_summary(template, tmp_path)
    producers, injectors = summary["producers"], summary["injectors"]
    assert isinstance(producers, dict)
    assert isinstance(injectors, dict)
    assert set(producers) == {"P1"}
    assert set(injectors) == {"I1"}


def test_a_deck_beyond_the_file_bound_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the bound, the reader fails loudly instead of leaving the later files out."""
    monkeypatch.setattr(deck_module, "_MAX_DECK_FILES", 3)
    for i in range(4):
        (tmp_path / f"part{i}.inc").write_text("-- part\n")
    template = tmp_path / "deck.mako"
    template.write_text("RUNSPEC\n" + "".join(f"INCLUDE\n 'part{i}.inc' /\n" for i in range(4)))
    with pytest.raises(ValueError, match="more than 3 files"):
        schedule_summary(template, tmp_path)


def test_a_file_reached_by_two_spellings_is_read_once(tmp_path: Path) -> None:
    """A self-include written with '..' ends at once, and its wells are not repeated."""
    (tmp_path / "inc").mkdir()
    (tmp_path / "inc" / "wells.inc").write_text(
        "WELSPECS\n 'P1' G1 2 3 2000 OIL /\n/\nCOMPDAT\n 'P1' 2* 1 1 OPEN /\n/\n"
        "INCLUDE\n '../inc/wells.inc' /\n"
    )
    template = tmp_path / "deck.mako"
    template.write_text("RUNSPEC\nDIMENS\n 4 4 2 /\nSCHEDULE\nINCLUDE\n 'inc/wells.inc' /\n")
    layout = deck_layout(template)
    assert layout["wells"] == {"P1": [[2, 3, 1]]}
    files = layout["files"]
    assert isinstance(files, list)
    assert len(files) == 2


def test_paths_aliases_reach_the_wells(tmp_path: Path) -> None:
    """A '$ALIAS/...' include (Eclipse's PATHS) is read, as Brugge's well files are."""
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "Welspecs").write_text(
        "WELSPECS\n 'P1' G1 2 3 2000 OIL /\n/\nCOMPDAT\n 'P1' 2* 1 2 OPEN /\n/\n"
    )
    template = tmp_path / "deck.mako"
    template.write_text(
        "RUNSPEC\nDIMENS\n 4 4 2 /\nPATHS\n'FLOC' '${case_dir}/include' /\n/\n"
        "SCHEDULE\nINCLUDE\n '$FLOC/Welspecs' /\n"
    )
    assert deck_layout(template)["wells"] == {"P1": [[2, 3, 1], [2, 3, 2]]}


def test_deck_keywords_follow_the_includes(tmp_path: Path) -> None:
    """Tables in included files come back as records, a table's rows up to its / as one."""
    (tmp_path / "props").mkdir()
    (tmp_path / "props" / "fluids.inc").write_text(
        "SWOF\n-- Sw krw krow pc\n0.2 0 1 0\n0.8 0.5 0 0 /\n\nPVTW\n200 1.02 4E-5 0.5 0 /\n"
    )
    template = tmp_path / "deck.mako"
    template.write_text("RUNSPEC\nOIL\nWATER\nPROPS\nINCLUDE\n 'props/fluids.inc' /\n")
    found = dict(deck_keywords(template))
    assert found["SWOF"] == ["0.2 0 1 0 0.8 0.5 0 0"]
    assert found["PVTW"] == ["200 1.02 4E-5 0.5 0"]
    assert "OIL" in found


def test_the_public_reader_and_text_after_a_records_slash() -> None:
    """``keyword_records`` and ``include_path`` (public for plugins); as in Eclipse and OPM
    Flow, the text after a record's ``/`` on its line is a comment, so a second record on
    the same line is not read."""
    from apex.simulation.deck import include_path, keyword_records

    text = 'EQUALS\n PERMZ 1 / PERMY 2 /\n PORO 0.2 /\n/\nINCLUDE\n"${case_dir}/a.inc" /\n'
    found = dict(keyword_records(text))
    assert found["EQUALS"] == ["PERMZ 1", "PORO 0.2"]
    assert include_path(found["INCLUDE"][0]) == "${case_dir}/a.inc"
    assert include_path("grid.inc") == "grid.inc"
    assert include_path("") is None
