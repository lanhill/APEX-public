"""``stop_deck_at``: a rendered deck cut at a report date.

By hand: decks whose schedule is written out here, cut at each kind of report date, against
the text expected; the dates the cut deck reports, read back by the deck reader's own
schedule (``restart_schedule``), end at the cut.
"""

import datetime as dt
from pathlib import Path

import pytest

from apex.simulation.deck import restart_schedule, stop_deck_at

DECK = """RUNSPEC
TITLE
 A SMALL TEST DECK
DIMENS
 2 1 1 /
START
 1 'JAN' 2022 /
GRID
DX
 2*10 /
SCHEDULE
WCONINJE
'I' WATER 'OPEN' RATE 100 /
/
DATES
 1 'FEB' 2022 /
 1 'MAR' 2022 /
/
TSTEP
 10 2*20 /
DATES
 1 JUN 2022 /
/
"""


def _dates(text: str, tmp_path: Path) -> list[str]:
    path = tmp_path / "CUT.DATA"
    path.write_text(text)
    return [t.date().isoformat() for t in restart_schedule(path).report_times]


@pytest.mark.parametrize(
    ("stop", "last"),
    [
        (dt.date(2022, 2, 1), "2022-02-01"),  # the first record of a DATES keyword
        (dt.date(2022, 3, 1), "2022-03-01"),  # its last record
        (dt.date(2022, 3, 11), "2022-03-11"),  # the first TSTEP step
        (dt.date(2022, 3, 31), "2022-03-31"),  # a repeated one (2*20)
        (dt.date(2022, 6, 1), "2022-06-01"),  # the last report date
    ],
)
def test_the_deck_ends_at_the_report_date(tmp_path: Path, stop: dt.date, last: str) -> None:
    cut = stop_deck_at(DECK, stop)
    assert cut.endswith("END\n")
    dates = _dates(cut, tmp_path)
    assert dates[0] == "2022-01-01"
    assert dates[-1] == last
    assert dates == [d for d in _dates(DECK, tmp_path) if d <= last]


def test_the_text_before_the_cut_is_the_decks_own() -> None:
    feb = DECK.index(" 1 'FEB' 2022 /\n") + len(" 1 'FEB' 2022 /\n")
    assert stop_deck_at(DECK, dt.date(2022, 2, 1)) == DECK[:feb] + "/\n\nEND\n"


def test_a_tstep_record_is_shortened_to_the_steps_up_to_the_date() -> None:
    cut = stop_deck_at(DECK, dt.date(2022, 3, 31))
    assert "TSTEP\n 10 20 /\n\nEND\n" in cut
    assert "1 JUN 2022" not in cut


def test_a_date_that_is_no_report_date_is_refused() -> None:
    with pytest.raises(ValueError, match="not a report date"):
        stop_deck_at(DECK, dt.date(2022, 2, 15))
    with pytest.raises(ValueError, match="not a report date"):
        stop_deck_at(DECK, dt.date(2023, 1, 1))  # after the last


def test_a_tstep_after_a_schedule_include_is_refused_without_the_templates_dates() -> None:
    deck = DECK.replace("TSTEP\n", "INCLUDE\n 'more.sch' /\nTSTEP\n")
    with pytest.raises(ValueError, match="TSTEP follows an INCLUDE"):
        stop_deck_at(deck, dt.date(2022, 3, 11))
    # A DATES record after the include still places itself (its date is absolute).
    assert stop_deck_at(deck, dt.date(2022, 3, 1)).endswith("END\n")


def test_without_start_only_dates_records_place_themselves() -> None:
    deck = DECK.replace("START\n 1 'JAN' 2022 /\n", "")
    assert stop_deck_at(deck, dt.date(2022, 2, 1)).endswith("END\n")  # DATES: absolute
    # A TSTEP after a DATES record counts from it; one before any counts from START.
    assert stop_deck_at(deck, dt.date(2022, 3, 11)).endswith("END\n")
    first = deck.replace("DATES\n 1 'FEB' 2022 /\n 1 'MAR' 2022 /\n/\n", "")
    with pytest.raises(ValueError, match="no START"):
        stop_deck_at(first, dt.date(2022, 3, 11))


def test_the_3dbox_deck_is_cut_at_a_monthly_report_date(tmp_path: Path) -> None:
    """A real deck of the repository: comments, quoted months, RPTRST between DATES."""
    deck = Path(__file__).resolve().parents[3] / "examples" / "3dbox_production" / "BOX.DATA"
    text = deck.read_text()
    cut = stop_deck_at(text, dt.date(2022, 6, 1))
    folder = tmp_path / "box"
    folder.mkdir()
    for name in ("pvt.txt", "Schdl.sch"):
        (folder / name).write_text((deck.parent / name).read_text())
    dates = _dates(cut, folder)
    assert dates[-1] == "2022-06-01"
    assert dates == [d for d in _dates(text, folder) if d <= "2022-06-01"]


def test_the_deck_simulators_cut_a_members_deck_and_leave_plain_ones_unchanged(
    tmp_path: Path,
) -> None:
    """A member without ``stop_at`` renders byte for byte as before; one with it, cut."""
    from apex.core.simulation import MemberInput
    from apex.simulation.deck import render_deck, stopped_deck
    from apex.simulation.opm_flow import OPMFlowSimulator

    template = tmp_path / "deck.mako"
    template.write_text(DECK)
    simulator = OPMFlowSimulator(deck_template_file=template)
    assert simulator.stops_at_report_dates
    plain = MemberInput(3, {})
    assert simulator._render(plain) == render_deck(template, plain, {}) == DECK
    early = MemberInput(3, {}, stop_at="2022-03-01")
    assert simulator._render(early) == stop_deck_at(DECK, dt.date(2022, 3, 1))
    from apex.core.errors import SimulationError

    with pytest.raises(SimulationError, match="cannot stop at 2022-02-15"):
        stopped_deck(DECK, MemberInput(3, {}, stop_at="2022-02-15"))


def test_eclipse_states_the_capability() -> None:
    from apex.simulation.eclipse import EclipseSimulator

    assert EclipseSimulator.stops_at_report_dates


def test_with_the_templates_dates_a_tstep_after_a_schedule_include_is_cut(tmp_path: Path) -> None:
    """A deck whose wells are in an include before its TSTEPs (the review of 2026-11-04)."""
    from apex.simulation.deck import template_report_dates

    (tmp_path / "wells.sch").write_text("WELSPECS\n 'P' 'G' 1 1 1* OIL /\n/\n")
    deck = DECK.replace("TSTEP\n", "INCLUDE\n 'wells.sch' /\nTSTEP\n")
    template = tmp_path / "deck.mako"
    template.write_text(deck)
    dates = template_report_dates(template)
    assert dates is not None
    cut = stop_deck_at(deck, dt.date(2022, 3, 31), dates)
    assert "TSTEP\n 10 20 /\n\nEND\n" in cut
    # An include that holds report dates: the main text cannot place the cut.
    (tmp_path / "wells.sch").write_text("TSTEP\n 5 /\n")
    with pytest.raises(ValueError, match="includes hold report dates"):
        stop_deck_at(deck, dt.date(2022, 3, 31), template_report_dates(template))
    # A DATES record after the include makes the time known again.
    later = deck.replace("TSTEP\n 10 2*20 /\n", "DATES\n 5 MAR 2022 /\n/\nTSTEP\n 6 /\n")
    assert stop_deck_at(later, dt.date(2022, 3, 11)).endswith("END\n")


def test_skip_restart_and_times_of_day_are_refused() -> None:
    skipped = DECK.replace("DATES\n 1 'FEB'", "SKIP\nDATES\n 1 'FEB'")
    with pytest.raises(ValueError, match="SKIP before"):
        stop_deck_at(skipped, dt.date(2022, 3, 1))
    timed = DECK.replace(" 1 'FEB' 2022 /", " 1 'FEB' 2022 06:00:00 /")
    with pytest.raises(ValueError, match="time of day"):
        stop_deck_at(timed, dt.date(2022, 3, 1))
    # A time of day after the date does not matter: the cut comes first.
    assert stop_deck_at(DECK.replace(" 1 JUN 2022 /", " 1 JUN 2022 12:00 /"), dt.date(2022, 3, 1))


def test_a_later_record_on_the_stop_records_line_is_dropped() -> None:
    """The coder's review: the cut ends at the stop record's own "/"."""
    deck = DECK.replace(" 1 'FEB' 2022 /\n 1 'MAR' 2022 /\n", " 1 'FEB' 2022 / 1 'MAR' 2022 /\n")
    cut = stop_deck_at(deck, dt.date(2022, 2, 1))
    assert "MAR" not in cut
    assert cut.endswith(" 1 'FEB' 2022 /\n/\n\nEND\n")


def test_records_on_the_keywords_own_line_are_refused() -> None:
    deck = DECK.replace("TSTEP\n 10 2*20 /\n", "TSTEP 10 2*20 /\n")
    with pytest.raises(ValueError, match="on its own line"):
        stop_deck_at(deck, dt.date(2022, 3, 31))


def test_executors_refuse_a_stop_date_the_simulator_would_ignore(tmp_path: Path) -> None:
    """The coder's review: a simulator without the capability (or a wrapper that does not
    forward it) would run the whole schedule; the shared run path refuses the member."""
    import numpy as np

    from apex.core.simulation import MemberInput
    from apex.domain.observations import DataKey
    from apex.execution.local import SerialExecutor

    class Whole:
        def run(self, member: MemberInput, keys: object) -> np.ndarray:
            return np.zeros(1)

    member = MemberInput(1, {}, stop_at="2022-03-01")
    with pytest.raises(ValueError, match="cannot stop at a report date"):
        SerialExecutor().run_all(Whole(), [member], [DataKey("1", "X")])
    # Without a stop date, and with the capability, the member runs.
    assert SerialExecutor().run_all(Whole(), [MemberInput(1, {})], [DataKey("1", "X")])

    class Stops(Whole):
        stops_at_report_dates = True

    assert SerialExecutor().run_all(Stops(), [member], [DataKey("1", "X")])


def test_under_a_runner_the_cut_uses_the_local_templates_dates(tmp_path: Path) -> None:
    """The rendered deck names the host's folder for its includes (the runner's static
    folder); the cut reads the report dates from the local template, includes followed."""
    from apex.core.simulation import MemberInput
    from apex.simulation.opm_flow import OPMFlowSimulator

    (tmp_path / "wells.sch").write_text("WELSPECS\n 'P' 'G' 1 1 1* OIL /\n/\n")
    deck = DECK.replace("TSTEP\n", "INCLUDE\n '${case_dir}/wells.sch' /\nTSTEP\n")
    template = tmp_path / "deck.mako"
    template.write_text(deck)

    class Runner:
        def static_folder(self, local: Path, files: object = None) -> Path:
            return Path("/elsewhere/case")  # not readable here

    simulator = OPMFlowSimulator(deck_template_file=template)
    member = MemberInput(2, {}, runner=Runner(), stop_at="2022-03-31")  # type: ignore[arg-type]
    text = simulator._render(member)
    assert "'/elsewhere/case/wells.sch'" in text
    assert text.endswith("TSTEP\n 10 20 /\n\nEND\n")
