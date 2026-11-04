"""``stop_deck_at``: a rendered deck cut at a report date.

By hand: decks whose schedule is written out here, cut at each kind of report date, against
the text expected; the dates the cut deck reports, read back by the deck reader's own
schedule (``restart_schedule``), end at the cut.
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pytest

from apex.simulation.deck import include_resolver, restart_schedule, stop_deck_at

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
    folder); the cut reads the report dates from the local template, includes followed,
    and inlines the schedule's include from the local folder (a schedule's include is
    text, so the host's copy is not needed)."""
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
    assert "'/elsewhere/case/wells.sch'" not in text
    assert "the INCLUDE of wells.sch inlined" in text
    assert "WELSPECS\n 'P' 'G' 1 1 1* OIL /" in text
    assert text.endswith("TSTEP\n 10 20 /\n\nEND\n")


# ------------------------------------------------- schedules in includes (the user, 2026-11-04)

MAIN = """RUNSPEC
START
 1 'JAN' 2022 /
GRID
INCLUDE
 '{folder}/GRID.INC' /
SCHEDULE
WCONINJE
'I' WATER 'OPEN' RATE 100 /
/
INCLUDE
 '{folder}/HISTORY.SCH' /
END
"""

HISTORY = """WCONINJE
'I' WATER 'OPEN' RATE 110 /
/
DATES
 1 'FEB' 2022 /
/
INCLUDE
 '{folder}/MORE.SCH' /
TSTEP
 10 /
"""

MORE = """WCONINJE
'I' WATER 'OPEN' RATE 120 /
/
DATES
 1 'MAR' 2022 /
 1 'APR' 2022 /
/
"""


def _schedule_files(folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "GRID.INC").write_text("DX\n 2*10 /\n")
    (folder / "HISTORY.SCH").write_text(HISTORY.format(folder=folder))
    (folder / "MORE.SCH").write_text(MORE)
    return MAIN.format(folder=folder)


def _local(folder: Path) -> Callable[[str], Path | None]:
    def resolve(name: str) -> Path | None:
        path = Path(name) if Path(name).is_absolute() else folder / name
        return path if path.is_file() else None

    return resolve


@pytest.mark.parametrize(
    ("stop", "last"),
    [
        (dt.date(2022, 2, 1), "2022-02-01"),  # in the include
        (dt.date(2022, 3, 1), "2022-03-01"),  # in a nested include
        (dt.date(2022, 4, 11), "2022-04-11"),  # a TSTEP after the nested include
    ],
)
def test_a_schedule_in_includes_is_inlined_and_cut(
    tmp_path: Path, stop: dt.date, last: str
) -> None:
    """The schedule's includes are inlined (text inclusion, as the simulator reads them),
    then cut; includes of other sections are left alone."""
    deck = _schedule_files(tmp_path / "case")
    with pytest.raises(ValueError, match="not a report date of the deck's main text"):
        stop_deck_at(deck, stop)  # without a resolver, as before
    cut = stop_deck_at(deck, stop, resolve=_local(tmp_path / "case"))
    assert cut.endswith("END\n")
    assert f"'{tmp_path / 'case'}/GRID.INC'" in cut  # the GRID include kept
    assert "HISTORY.SCH" not in cut.replace("the INCLUDE of HISTORY.SCH inlined", "")
    assert "RATE 110" in cut
    assert ("RATE 120" in cut) == (stop > dt.date(2022, 2, 1))
    dates = _dates(cut, tmp_path)
    assert dates[-1] == last
    assert all(d <= last for d in dates)


def test_the_cut_deck_runs_the_same_schedule_up_to_the_date(tmp_path: Path) -> None:
    """Up to the cut, the inlined deck's report dates are the whole deck's, read with its
    includes followed (``restart_schedule``)."""
    deck = _schedule_files(tmp_path / "case")
    whole = tmp_path / "case" / "WHOLE.DATA"
    whole.write_text(deck)
    full = [t.date().isoformat() for t in restart_schedule(whole).report_times]
    cut = stop_deck_at(deck, dt.date(2022, 3, 1), resolve=_local(tmp_path / "case"))
    assert _dates(cut, tmp_path) == [d for d in full if d <= "2022-03-01"]


def test_an_include_not_resolved_is_left_and_refused_as_before(tmp_path: Path) -> None:
    deck = _schedule_files(tmp_path / "case").replace(f"{tmp_path / 'case'}/HISTORY.SCH",
                                                      "$SCH/HISTORY.SCH")  # fmt: skip
    with pytest.raises(ValueError, match="not a report date"):
        stop_deck_at(deck, dt.date(2022, 2, 1), resolve=include_resolver(
            tmp_path / "case" / "deck.mako", tmp_path / "case"))  # fmt: skip


def test_the_resolver_maps_a_hosts_folder_to_the_local_one(tmp_path: Path) -> None:
    """Under a runner the deck names the host's folder; the files are read here."""
    local = tmp_path / "case"
    _schedule_files(local)
    template = local / "deck.mako"
    resolve = include_resolver(template, local, Path("/remote/static/abc"))
    assert resolve("/remote/static/abc/HISTORY.SCH") == local.resolve() / "HISTORY.SCH"
    assert resolve("/remote/static/abc/MISSING.SCH") is None
    assert resolve("$ALIAS/HISTORY.SCH") is None
    # Relative paths are the deck's folder's (the member's), which has no copy here.
    assert resolve("MORE.SCH") is None
    # A folder sharing the prefix is another folder; an absolute path outside the host's
    # case folder may be another file there than here (review).
    assert resolve("/remote/static/abc2/HISTORY.SCH") is None
    assert resolve(str(local.resolve() / "HISTORY.SCH")) is None
    # Without a runner, an absolute path outside the case folder is the same file.
    here = include_resolver(template, local)
    assert here(str(local.resolve() / "HISTORY.SCH")) == local.resolve() / "HISTORY.SCH"


def test_text_after_an_include_records_slash_is_a_comment(tmp_path: Path) -> None:
    """The simulators ignore it; it goes with the record (review)."""
    deck = _schedule_files(tmp_path / "case").replace("HISTORY.SCH' /", "HISTORY.SCH' / X")
    cut = stop_deck_at(deck, dt.date(2022, 2, 1), resolve=_local(tmp_path / "case"))
    assert "RATE 110" in cut
    assert "/ X" not in cut


def test_a_schedule_include_that_is_not_utf8_is_refused_by_name(tmp_path: Path) -> None:
    deck = _schedule_files(tmp_path / "case")
    latin = ("-- br\u00f8nn\n" + HISTORY.format(folder=tmp_path / "case")).encode("latin-1")
    (tmp_path / "case" / "HISTORY.SCH").write_bytes(latin)
    with pytest.raises(ValueError, match=r"HISTORY\.SCH is not UTF-8 text"):
        stop_deck_at(deck, dt.date(2022, 2, 1), resolve=_local(tmp_path / "case"))


def test_a_non_utf8_include_without_report_keywords_is_left_to_the_simulator(
    tmp_path: Path,
) -> None:
    """Latin-1 comments in a wells-only include must not stop a cut that needs nothing of it
    (the coder's review: decks cut before the inlining)."""
    folder = tmp_path / "case"
    deck = _schedule_files(folder).replace(
        "SCHEDULE\n", f"SCHEDULE\nINCLUDE\n '{folder}/WELLS.SCH' /\n", 1
    )
    (folder / "WELLS.SCH").write_bytes("-- br\u00f8nn\nWELSPECS\n/\n".encode("latin-1"))
    cut = stop_deck_at(deck, dt.date(2022, 2, 1), resolve=_local(folder))
    assert f"'{folder}/WELLS.SCH'" in cut  # left as an INCLUDE
    assert "RATE 110" in cut  # the other include inlined


def test_an_include_that_cannot_be_read_is_a_transient_failure(tmp_path: Path) -> None:
    """An I/O error (a stale file handle, a permission flicker) may not recur: retried."""
    from apex.core.errors import TransientSimulationError
    from apex.core.simulation import MemberInput
    from apex.simulation.deck import stopped_deck

    folder = tmp_path / "case"
    deck = _schedule_files(folder)

    def unreadable(name: str) -> Path | None:
        return folder  # a folder: reading it raises IsADirectoryError, an OSError

    member = MemberInput(1, {}, stop_at="2022-02-01")
    with pytest.raises(TransientSimulationError, match="reading its schedule includes"):
        stopped_deck(deck, member, resolve=unreadable)


def test_the_resolver_follows_symbolic_links_and_dot_dots(tmp_path: Path) -> None:
    """A case folder reached through a link is the same folder (coder's review)."""
    real = tmp_path / "real"
    _schedule_files(real)
    link = tmp_path / "link"
    link.symlink_to(real)
    resolve = include_resolver(link / "deck.mako", link, link)
    assert resolve(f"{link}/HISTORY.SCH") == real / "HISTORY.SCH"
    host = include_resolver(real / "deck.mako", real, Path("/remote/x"))
    assert host("/remote/x/../x/HISTORY.SCH") == real / "HISTORY.SCH"
    assert host("/remote/x/../y/HISTORY.SCH") is None


def test_a_title_with_a_quote_does_not_hide_the_schedule(tmp_path: Path) -> None:
    deck = _schedule_files(tmp_path / "case").replace("RUNSPEC\n", "RUNSPEC\nTITLE\nBob's deck\n")
    cut = stop_deck_at(deck, dt.date(2022, 2, 1), resolve=_local(tmp_path / "case"))
    assert "RATE 110" in cut


def test_includes_after_a_date_past_the_stop_are_not_read(tmp_path: Path) -> None:
    """The cut drops them, so they are not inlined (nor read: a missing file is no error)."""
    folder = tmp_path / "case"
    deck = _schedule_files(folder).replace(
        "END\n", f"DATES\n 1 'JAN' 2023 /\n/\nINCLUDE\n '{folder}/LATER.SCH' /\nEND\n"
    )
    seen: list[str] = []

    def resolve(name: str) -> Path | None:
        seen.append(Path(name).name)
        return _local(folder)(name)

    stop_deck_at(deck, dt.date(2022, 2, 1), resolve=resolve)
    assert "LATER.SCH" not in seen
    assert "HISTORY.SCH" in seen


def test_an_include_of_itself_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    deck = _schedule_files(folder)
    (folder / "MORE.SCH").write_text("INCLUDE\n 'MORE.SCH' /\n")
    with pytest.raises(ValueError, match="nest more than ten deep"):
        stop_deck_at(deck, dt.date(2022, 2, 1), resolve=_local(folder))
