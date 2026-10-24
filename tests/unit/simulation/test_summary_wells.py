"""The deck's SUMMARY requests, from its text (the release tester, 2026-10-19).

``pg_lstm`` reads every listed well's WBHP; a deck without it was simulated in full before
the first member's read failed ("summary MODEL has no WBHP:PRO1"). ``apex validate`` and
the run's build now read the SUMMARY section first.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.core.errors import ConfigError
from apex.services.composition import check_model_error_summary
from apex.simulation.deck import missing_summary_wells

DECK = """RUNSPEC
OIL
WATER
DIMENS
 2 2 1 /
GRID
SUMMARY
FOPR
{summary}
SCHEDULE
TSTEP
 1 /
END
"""


def _deck(tmp_path: Path, summary: str) -> Path:
    path = tmp_path / "deck.mako"
    path.write_text(DECK.format(summary=summary))
    return path


@pytest.mark.parametrize(
    ("summary", "missing"),
    [
        ("WOPR\n/", ["PRO1", "INJ1"]),  # no WBHP at all
        ("WBHP\n 'PRO1' /", ["INJ1"]),
        ("WBHP\n/", []),  # every well
        ("WBHP\n 'PRO*' INJ1 /", []),  # a pattern and a bare name
        ("WBHP\n 'PRO1' /\nWBHP\n 'INJ1' /", []),
    ],
)
def test_the_wells_without_the_vector_are_found(
    tmp_path: Path, summary: str, missing: list[str]
) -> None:
    deck = _deck(tmp_path, summary)
    assert missing_summary_wells(deck, "WBHP", ["PRO1", "INJ1"]) == missing


def test_quoted_names_lower_case_keywords_and_well_lists(tmp_path: Path) -> None:
    """Names with spaces; a lower-case keyword, a well list or a broken record is unknown."""
    assert missing_summary_wells(_deck(tmp_path, "WBHP\n 'PRO 1' /"), "WBHP", ["PRO 1"]) == []
    # A lower-case keyword is not read as one: unknown, never missing.
    assert missing_summary_wells(_deck(tmp_path, "wbhp\n/"), "WBHP", ["PRO1"]) is None
    # ... also right after SUMMARY, where its record is SUMMARY's own.
    first = tmp_path / "first.mako"
    first.write_text(DECK.replace("SUMMARY\nFOPR\n{summary}", "SUMMARY\nwbhp\n/\nFOPR"))
    assert missing_summary_wells(first, "WBHP", ["PRO1"]) is None
    listed = _deck(tmp_path, "WBHP\n '*PRODS' /")
    assert missing_summary_wells(listed, "WBHP", ["PRO1"]) is None
    broken = _deck(tmp_path, "WBHP\n 'PRO1 /")  # an open quote: the reader drops the record
    assert missing_summary_wells(broken, "WBHP", ["PRO1"]) is None


def test_an_included_every_well_record_stays_unknown(tmp_path: Path) -> None:
    """``WBHP`` + ``/`` in an included file: unknown (never missing), as documented."""
    (tmp_path / "summary.inc").write_text("WBHP\n/\n")
    deck = _deck(tmp_path, "INCLUDE\n 'summary.inc' /")
    assert missing_summary_wells(deck, "WBHP", ["PRO1"]) is None


def test_what_the_text_cannot_tell_is_unknown(tmp_path: Path) -> None:
    assert missing_summary_wells(_deck(tmp_path, "ALL"), "WBHP", ["PRO1"]) is None
    templated = _deck(tmp_path, "WBHP\n ${wells} /")
    assert missing_summary_wells(templated, "WBHP", ["PRO1"]) is None
    unread = _deck(tmp_path, "INCLUDE\n 'missing.inc' /")
    assert missing_summary_wells(unread, "WBHP", ["PRO1"]) is None


def _config(deck: Path) -> SimpleNamespace:
    return SimpleNamespace(
        simulator=SimpleNamespace(options={"deck_template_file": str(deck)}),
        model_error=SimpleNamespace(name="pg_lstm"),
    )


def test_a_deck_without_the_models_wbhp_is_refused_before_any_run(tmp_path: Path) -> None:
    model = SimpleNamespace(summary_vectors=[("WBHP", "PRO1"), ("WBHP", "INJ1")])
    with pytest.raises(ConfigError, match="no WBHP of PRO1, INJ1"):
        check_model_error_summary(_config(_deck(tmp_path, "WOPR\n/")), model)  # type: ignore[arg-type]
    check_model_error_summary(_config(_deck(tmp_path, "WBHP\n/")), model)  # type: ignore[arg-type]
    check_model_error_summary(_config(_deck(tmp_path, "ALL")), model)  # type: ignore[arg-type]
