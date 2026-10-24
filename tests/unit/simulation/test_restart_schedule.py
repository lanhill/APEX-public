"""A deck's report times and restart steps, read from its text.

The expected steps follow opm-common's ``ScheduleState::rst_file`` and ``RSTConfig``,
worked by hand for each case; restarts at report step k are decided by the control in
force in OPM's schedule block k (the keywords after the k-th report time).
"""

import datetime as dt
from pathlib import Path

import pytest

from apex.simulation.deck import DeckContents, deck_contents, restart_schedule


def _deck(tmp_path: Path, solution: str, schedule: str, start: str = "1 JAN 2020") -> Path:
    path = tmp_path / "deck.mako"
    path.write_text(
        f"RUNSPEC\nSTART\n {start} /\nGRID\nSOLUTION\n{solution}\nSUMMARY\nSCHEDULE\n"
        f"{schedule}\nEND\n"
    )
    return path


def _dates(*isos: str) -> tuple[dt.datetime, ...]:
    return tuple(dt.datetime.fromisoformat(d) for d in isos)


def test_basic_2_writes_every_report_time_including_the_start(tmp_path: Path) -> None:
    deck = _deck(
        tmp_path,
        "RPTRST\n BASIC=2 /",
        "DATES\n 1 FEB 2020 /\n 1 'JLY' 2020 12:30:00 /\n/\nTSTEP\n 2*10 /",
    )
    schedule = restart_schedule(deck)
    assert schedule.report_times == _dates(
        "2020-01-01", "2020-02-01", "2020-07-01T12:30", "2020-07-11T12:30", "2020-07-21T12:30"
    )
    assert schedule.restart_steps == (0, 1, 2, 3, 4)
    assert schedule.unresolved == ()


def test_includes_are_read_in_place(tmp_path: Path) -> None:
    (tmp_path / "middle.sch").write_text("DATES\n 1 MAR 2020 /\n/\n")
    deck = _deck(
        tmp_path,
        "",
        "RPTRST\n BASIC=2 /\nDATES\n 1 FEB 2020 /\n/\nINCLUDE\n '${case_dir}/middle.sch' /\n"
        "DATES\n 1 APR 2020 /\n/",
    )
    schedule = restart_schedule(deck)
    assert [t.month for t in schedule.report_times] == [1, 2, 3, 4]
    # RPTRST before the first date also writes the start's restart, as OPM does.
    assert schedule.restart_steps == (0, 1, 2, 3)


def test_basic_3_counts_report_steps_with_freq(tmp_path: Path) -> None:
    deck = _deck(
        tmp_path,
        "",
        "DATES\n 1 FEB 2020 /\n/\nRPTRST\n BASIC=3 FREQ=2 /\n"
        "DATES\n 1 MAR 2020 /\n 1 APR 2020 /\n 1 MAY 2020 /\n/",
    )
    assert restart_schedule(deck).restart_steps == (2, 4)


def test_basic_4_writes_the_first_report_time_of_each_year(tmp_path: Path) -> None:
    deck = _deck(
        tmp_path,
        "RPTRST\n BASIC=4 /",
        "DATES\n 1 JUN 2020 /\n 1 JAN 2021 /\n 1 JUN 2021 /\n 1 MAR 2022 /\n/",
    )
    schedule = restart_schedule(deck)
    assert schedule.restart_steps == (2, 4)  # BASIC 4 in SOLUTION writes no initial restart
    assert schedule.unresolved == ()


def test_basic_5_writes_the_first_report_time_of_each_month(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "RPTRST\n BASIC=5 /", "TSTEP\n 10 10 10 20 /")
    schedule = restart_schedule(deck)
    assert schedule.report_times[-1] == dt.datetime(2020, 2, 20)
    assert schedule.restart_steps == (4,)


def test_basic_4_with_freq_is_left_to_the_read(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "RPTRST\n BASIC=4 FREQ=2 /", "DATES\n 1 JAN 2021 /\n/")
    assert "FREQ=2" in restart_schedule(deck).unresolved[0]


def test_a_control_before_a_date_governs_that_dates_restart(tmp_path: Path) -> None:
    """opm-common's must_write_rst_file(k) reads snapshot k - 1's control: RPTRST BASIC=0
    right after the first date drops the later dates' restarts, but SAVE writes one at
    the report time of its own block (snapshots[k].save())."""
    deck = _deck(
        tmp_path,
        "RPTRST\n BASIC=2 /",
        "DATES\n 1 FEB 2020 /\n/\nRPTRST\n BASIC=0 /\nDATES\n 1 MAR 2020 /\n/\n"
        "SAVE\nDATES\n 1 APR 2020 /\n 1 MAY 2020 /\n/",
    )
    assert restart_schedule(deck).restart_steps == (0, 1, 2)


def test_rptsched_restart_and_the_integer_form(tmp_path: Path) -> None:
    rptsched = _deck(tmp_path, "", "RPTSCHED\n 'RESTART=3' /\nDATES\n 1 FEB 2020 /\n/")
    assert restart_schedule(rptsched).restart_steps == (0, 1)
    integer = _deck(tmp_path, "RPTRST\n 2 0 0 /", "DATES\n 1 FEB 2020 /\n/")
    assert restart_schedule(integer).restart_steps == (0, 1)


def test_what_the_text_cannot_tell_is_reported(tmp_path: Path) -> None:
    deck = _deck(
        tmp_path,
        "RPTRST\n BASIC=2 /",
        "% for d in dates:\nDATES\n ${d} /\n/\n% endfor\nINCLUDE\n 'missing.sch' /",
    )
    unresolved = " | ".join(restart_schedule(deck).unresolved)
    assert "Mako lines around DATES" in unresolved
    assert "DATES holds template expressions" in unresolved
    assert "missing.sch" in unresolved


def test_missing_names_the_dates_without_a_restart(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "", "RPTRST\n BASIC=3 FREQ=2 /\nTSTEP\n 31 29 /")
    schedule = restart_schedule(deck)
    assert schedule.restart_steps == (2,)
    assert schedule.missing([dt.date(2020, 2, 1), dt.date(2020, 3, 1)]) == [dt.date(2020, 2, 1)]


def test_mnemonics_on_their_own_lines_are_items_not_keywords(tmp_path: Path) -> None:
    """A regression: a bare FLOWS or FIP line ended the record and dropped the control."""
    rptrst = _deck(tmp_path, "", "RPTRST\n BASIC=2\n FLOWS\n/\nDATES\n 1 FEB 2020 /\n/")
    assert restart_schedule(rptrst).restart_steps == (0, 1)
    rptsched = _deck(tmp_path, "", "RPTSCHED\n PRES RESTART=2\n FIP\n/\nDATES\n 1 FEB 2020 /\n/")
    assert restart_schedule(rptsched).restart_steps == (0, 1)


def test_no_control_means_no_restarts_without_a_note(tmp_path: Path) -> None:
    schedule = restart_schedule(_deck(tmp_path, "", "DATES\n 1 FEB 2020 /\n/"))
    assert schedule.restart_steps == ()
    assert schedule.unresolved == ()


def test_the_integer_forms_freq_is_its_sixth_item(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "", "RPTRST\n 3 0 0 0 0 2 /\nTSTEP\n 4*1 /")
    assert restart_schedule(deck).restart_steps == (2, 4)


def test_rptsol_restart_writes_the_start(tmp_path: Path) -> None:
    deck = _deck(tmp_path, "RPTSOL\n RESTART=2 /", "DATES\n 1 FEB 2020 /\n/")
    assert restart_schedule(deck).restart_steps == (0,)


def test_reading_stops_at_end_and_notes_what_it_does_not_follow(tmp_path: Path) -> None:
    deck = _deck(
        tmp_path,
        "RESTART\n 'BASE' 3 /",
        "RPTRST\n BASIC=2 /\nDATES\n 1 FEB 2020 /\n 1 FEB 2020 24:00:00 /\n/\nEND\n"
        "DATES\n 1 MAR 2020 /\n/",
    )
    schedule = restart_schedule(deck)
    assert [t.month for t in schedule.report_times] == [1, 2]
    unresolved = " | ".join(schedule.unresolved)
    assert "RESTART is not followed" in unresolved
    assert "24:00:00" in unresolved


def test_nested_includes_resolve_from_the_including_file(tmp_path: Path) -> None:
    (tmp_path / "sch").mkdir()
    (tmp_path / "sch" / "outer.sch").write_text("INCLUDE\n 'inner.sch' /\n")
    (tmp_path / "sch" / "inner.sch").write_text("DATES\n 1 FEB 2020 '06:00' /\n/\n")
    deck = _deck(tmp_path, "", "RPTRST\n BASIC=2 /\nINCLUDE\n 'sch/outer.sch' /")
    schedule = restart_schedule(deck)
    assert schedule.report_times[-1] == dt.datetime(2020, 2, 1, 6)
    assert schedule.unresolved == ()


def test_a_missing_grid_include_is_not_noted(tmp_path: Path) -> None:
    path = tmp_path / "deck.mako"
    path.write_text(
        "RUNSPEC\nSTART\n 1 JAN 2020 /\nGRID\nINCLUDE\n 'GRID.GRDECL' /\nSCHEDULE\n"
        "RPTRST\n BASIC=2 /\nDATES\n 1 FEB 2020 /\n/\n"
    )
    assert restart_schedule(path).unresolved == ()


def test_paths_aliases_resolve_includes(tmp_path: Path) -> None:
    """Brugge's decks include '$FLOC/...' through PATHS: an alias, not a template."""
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "Hist").write_text("DATES\n 1 FEB 2020 /\n 1 MAR 2020 /\n/\n")
    path = tmp_path / "deck.mako"
    path.write_text(
        "RUNSPEC\nPATHS\n'FLOC' '${case_dir}/include' /\n/\nSTART\n 1 JAN 2020 /\n"
        "SCHEDULE\nRPTRST\n BASIC=2 /\nINCLUDE\n '$FLOC/Hist' /\n"
    )
    schedule = restart_schedule(path)
    assert schedule.restart_steps == (0, 1, 2)
    assert schedule.unresolved == ()


def test_mako_around_an_include_is_noted(tmp_path: Path) -> None:
    (tmp_path / "a.sch").write_text("DATES\n 1 FEB 2020 /\n/\n")
    (tmp_path / "b.sch").write_text("DATES\n 1 MAR 2020 /\n/\n")
    deck = _deck(
        tmp_path, "", "% if x:\nINCLUDE\n 'a.sch' /\n% else:\nINCLUDE\n 'b.sch' /\n% endif"
    )
    assert "Mako lines around INCLUDE" in restart_schedule(deck).unresolved


def test_a_section_keyword_with_trailing_text_starts_its_section(tmp_path: Path) -> None:
    path = tmp_path / "deck.mako"
    path.write_text(
        "RUNSPEC\nSTART\n 1 JAN 2020 /\nSCHEDULE  ==========\n% for d in dates:\n"
        "DATES\n 1 FEB 2020 /\n/\n% endfor\n"
    )
    assert "Mako lines around DATES" in restart_schedule(path).unresolved


def test_save_is_not_a_restart_event_for_basic_4(tmp_path: Path) -> None:
    """SAVE at the start writes X0000; BASIC=4 still counts its year from the start."""
    deck = _deck(tmp_path, "RPTRST\n BASIC=4 /", "SAVE\nDATES\n 1 JUN 2020 /\n 1 JAN 2021 /\n/")
    assert restart_schedule(deck).restart_steps == (0, 2)


def test_a_template_expression_alone_on_a_line_is_noted(tmp_path: Path) -> None:
    """A regression: a bare ${...} line after a block (which may render DATES) was read as
    nothing, so a check refusing a deck without a restart at a vintage refused valid ones."""
    deck = _deck(tmp_path, "", "RPTRST\n BASIC=2 /\nDATES\n 1 JUN 2020 /\n/\n${extra_schedule}")
    (note,) = restart_schedule(deck).unresolved
    assert "is a template expression" in note


# --- what the text says about the output files ----------------------------


def _contents(tmp_path: Path, runspec: str, schedule: str) -> DeckContents:
    deck = tmp_path / "deck.mako"
    deck.write_text(
        f"RUNSPEC\n{runspec}\nGRID\nSOLUTION\nSUMMARY\nSCHEDULE\n"
        f"START\n 1 JAN 2020 /\n{schedule}\nDATES\n 1 JUL 2020 /\n/\nEND\n"
    )
    return deck_contents(deck)


def test_deck_contents_reads_phases_grid_densities_and_output_kinds(tmp_path: Path) -> None:
    contents = _contents(
        tmp_path, "DIMENS\n 3 2 4 /\nOIL\nWATER\nGAS\nUNIFOUT", "RPTRST\n BASIC=2 DEN /"
    )
    assert contents.phases == frozenset({"oil", "water", "gas"})
    assert contents.grid == (3, 2, 4)
    assert (contents.densities, contents.unified, contents.formatted) == (True, True, False)
    assert not contents.basic_one

    plain = _contents(tmp_path, "OIL\nWATER\nFMTOUT", "RPTRST\n BASIC=1 /")
    assert (plain.densities, plain.formatted, plain.basic_one, plain.grid) == (
        False, True, True, None,
    )  # fmt: skip
    # The integer form names no mnemonics; RPTSCHED RESTART=1 is BASIC=1.
    integer = _contents(tmp_path, "OIL", "RPTSCHED\n RESTART=1 /\nRPTRST\n 2 /")
    assert (integer.densities, integer.basic_one) == (None, False)  # BASIC 1, then 2
    assert _contents(tmp_path, "OIL", "RPTSCHED\n RESTART=1 /").basic_one
    # Mako in RUNSPEC hides the phases; an expression may render RPTRST's mnemonics.
    templated = _contents(
        tmp_path, "% if gas:\nGAS\n% endif\nOIL\nDIMENS\n 3 2 4 /", "RPTRST\n BASIC=2 ${extra} /"
    )
    assert (templated.phases, templated.densities, templated.grid) == (None, None, None)
    # A defaulted first item is the integer form too; BASIC=1 among others is no refusal.
    defaulted = _contents(tmp_path, "OIL", "RPTRST\n 1* 0 0 /\nRPTRST\n BASIC=1 /")
    assert (defaulted.densities, defaulted.basic_one) == (None, True)
    mixed = _contents(tmp_path, "OIL", "RPTRST\n BASIC=1 /\nRPTRST\n BASIC=2 /")
    assert not mixed.basic_one
    # ALLPROPS may hold the densities; Mako or an include in RUNSPEC hides what it says.
    assert _contents(tmp_path, "OIL", "RPTRST\n BASIC=2 ALLPROPS /").densities is None
    branch = _contents(tmp_path, "OIL\n% if text:\nFMTOUT\n% endif", "RPTRST\n BASIC=2 /")
    assert (branch.formatted, branch.unified) == (None, None)
    included = _contents(tmp_path, "OIL\nINCLUDE\n 'phases.inc' /", "RPTRST\n BASIC=2 /")
    assert included.phases is None


def test_restart_arrays_templated_in_an_include_are_unknown_not_missing(tmp_path: Path) -> None:
    """RPORV and DEN chosen by Mako inside an include cannot be read from the text: the
    deck check warns, it does not refuse (the phase-3 review)."""
    (tmp_path / "rst.inc").write_text("RPTRST\n ${rptrst} /\n")
    deck = tmp_path / "deck.mako"
    deck.write_text(
        "RUNSPEC\nOIL\nWATER\nGRID\nSOLUTION\nINCLUDE\n '${case_dir}/rst.inc' /\nSUMMARY\n"
        "SCHEDULE\nEND\n"
    )
    contents = deck_contents(deck, tmp_path)
    assert (contents.densities, contents.pore_volumes) == (None, None)
    plain = _contents(tmp_path, "OIL\nWATER", "RPTRST\n BASIC=2 DEN RPORV /")
    assert (plain.densities, plain.pore_volumes) == (True, True)


def test_the_reservoir_temperature_is_read_in_celsius(tmp_path: Path) -> None:
    """RTEMP for CO₂'s equation of state: °F in a FIELD deck;
    RTEMPVD, TEMPVD and thermal runs vary it; a templated value is unknown."""
    metric = _contents(tmp_path, "WATER\nGAS\nMETRIC", "RTEMP\n 35.5 /")
    assert (metric.temperature, metric.temperature_varies) == (35.5, False)
    field = _contents(tmp_path, "WATER\nGAS\nFIELD", "RTEMP\n 140 /")
    assert field.temperature == pytest.approx(60.0)
    none = _contents(tmp_path, "WATER\nGAS", "RPTRST\n BASIC=2 /")
    assert (none.temperature, none.temperature_varies) == (None, False)
    templated = _contents(tmp_path, "WATER\nGAS", "RTEMP\n ${t} /")
    assert templated.temperature is None
    for varying in ("RTEMPVD\n 1000 30\n 2000 60 /", "TEMPVD\n 1000 30\n 2000 60 /"):
        assert _contents(tmp_path, "WATER\nGAS", varying).temperature_varies
    # One temperature at every depth, as OPM's own CO2STORE decks set it: RTEMP.
    constant = _contents(tmp_path, "WATER\nGAS", "RTEMPVD\n 0 40\n 2000 40 /")
    assert (constant.temperature, constant.temperature_varies) == (40.0, False)
    assert _contents(tmp_path, "WATER\nGAS\nTHERMAL", "RPTRST\n BASIC=2 /").temperature_varies


def test_the_temperature_tables_are_kept_per_region_and_thermal_runs_told_apart(
    tmp_path: Path,
) -> None:
    """Each EQLNUM region's RTEMPVD/TEMPVD table, in metres and °C
    (FIELD converted), the last keyword's; a thermal run (RUNSPEC THERMAL or TEMP, or
    TEMPI) is not a varying table; a templated, odd or non-increasing table is unknown."""
    two = _contents(
        tmp_path, "WATER\nGAS\nMETRIC", "RTEMPVD\n 1000 30\n 2000 60 /\n 0 40 3000 70 /"
    )
    assert two.temperature_tables == (
        ((1000.0, 30.0), (2000.0, 60.0)),
        ((0.0, 40.0), (3000.0, 70.0)),
    )
    assert (two.temperature_varies, two.thermal) == (True, False)
    field = _contents(tmp_path, "WATER\nGAS\nFIELD", "TEMPVD\n 1000 86\n 2000 140 /")
    assert field.temperature_tables is not None
    [[(d1, t1), (d2, t2)]] = field.temperature_tables
    assert (d1, d2) == pytest.approx((304.8, 609.6))
    assert (t1, t2) == pytest.approx((30.0, 60.0))
    last = _contents(tmp_path, "WATER\nGAS", "RTEMPVD\n 0 40 2000 40 /\nRTEMPVD\n 0 50 2000 60 /")
    assert last.temperature_tables == (((0.0, 50.0), (2000.0, 60.0)),)
    for unread in (
        "RTEMPVD\n 0 40\n 2000 /",
        "RTEMPVD\n 0 ${t} 2000 40 /",
        "RTEMPVD\n 2000 40 0 50 /",
    ):
        assert _contents(tmp_path, "WATER\nGAS", unread).temperature_tables is None
    assert (
        _contents(tmp_path, "WATER\nGAS\n${units}", "RTEMPVD\n 0 40 2000 60 /").temperature_tables
        is None
    )
    assert _contents(tmp_path, "WATER\nGAS", "RTEMP\n 40 /").temperature_tables is None
    lab = _contents(tmp_path, "WATER\nGAS\nLAB", "RTEMPVD\n 1000 30\n 2000 60 /")
    assert lab.temperature_tables is None  # centimetres: not converted, so unread
    for thermal in ("WATER\nGAS\nTHERMAL", "WATER\nGAS\nTEMP"):
        run = _contents(tmp_path, thermal, "RPTRST\n BASIC=2 /")
        assert (run.thermal, run.temperature_varies) == (True, True)
    tempi = tmp_path / "tempi.mako"
    tempi.write_text(
        "RUNSPEC\nWATER\nGAS\nGRID\nSOLUTION\nTEMPI\n 4*50 /\nSUMMARY\nSCHEDULE\nEND\n"
    )
    assert deck_contents(tempi).thermal
    # TEMP outside RUNSPEC is not the thermal option.
    assert not _contents(tmp_path, "WATER\nGAS", "RPTRST\n BASIC=2 /").thermal


def test_a_templated_runspec_leaves_the_temperature_unknown(tmp_path: Path) -> None:
    """A regression (review of phase 2b): its unit system may be FIELD, so RTEMP 140 may
    be °F; read as °C, the check advised a temperature 80 °C off."""
    templated = _contents(tmp_path, "WATER\nGAS\n${units}", "RTEMP\n 140 /")
    assert templated.temperature is None
    defaulted = _contents(tmp_path, "WATER\nGAS\nMETRIC", "RTEMP\n 1* /")
    assert defaulted.temperature is None


def test_co2store_and_its_form_are_read_from_runspec(tmp_path: Path) -> None:
    gas_water = _contents(tmp_path, "WATER\nGAS\nCO2STORE", "RPTRST\n BASIC=2 DEN /")
    assert gas_water.co2store is True
    oil_gas = _contents(tmp_path, "OIL\nGAS\nCO2STORE", "RPTRST\n BASIC=2 DEN /")
    assert (oil_gas.co2store, oil_gas.phases) == (True, frozenset({"oil", "gas"}))
    assert _contents(tmp_path, "OIL\nWATER", "RPTRST\n BASIC=2 /").co2store is False
    assert _contents(tmp_path, "OIL\nGAS\n${extra}", "RPTRST\n BASIC=2 /").co2store is None


def test_the_depth_table_wins_over_rtemp_and_mako_blocks_hide_both(tmp_path: Path) -> None:
    """OPM takes RTEMPVD's temperature when RTEMP is given too, whatever their order
    (measured on OPM Flow 2026.04); a temperature inside a Mako block is unknown, in any
    section (a regression of the review of phase 2b, step 2)."""
    table = "RTEMPVD\n 0 40\n 2000 40 /"
    for schedule in (f"RTEMP\n 60 /\n{table}", f"{table}\nRTEMP\n 60 /"):
        assert _contents(tmp_path, "WATER\nGAS", schedule).temperature == 40.0
    assert _contents(tmp_path, "WATER\nGAS", "RTEMPVD\n 0 40\n 2000 /").temperature is None
    deck = tmp_path / "mako.mako"
    deck.write_text(
        "RUNSPEC\nWATER\nGAS\nGRID\nPROPS\n% if hot:\nRTEMP\n 80 /\n% else:\nRTEMP\n 40 /\n"
        "% endif\nSOLUTION\nSUMMARY\nSCHEDULE\nEND\n"
    )
    assert deck_contents(deck).temperature is None


def test_a_schedules_mako_loop_does_not_hide_the_temperature(tmp_path: Path) -> None:
    """A regression (review of phase 2b, step 2): any Mako line from SOLUTION on marked the
    whole file's records as templated, so a varying RTEMPVD was only warned of."""
    loop = "% for i in range(4):\nTSTEP\n 90 /\n% endfor"
    varying = _contents(tmp_path, "WATER\nGAS", f"RTEMPVD\n 0 40\n 2000 80 /\n{loop}")
    assert (varying.temperature, varying.temperature_varies) == (None, True)
    constant = _contents(tmp_path, "WATER\nGAS", f"RTEMPVD\n 0 40\n 2000 40 /\n{loop}")
    assert (constant.temperature, constant.temperature_varies) == (40.0, False)
    rtemp = _contents(tmp_path, "WATER\nGAS", f"RTEMP\n 37 /\n{loop}")
    assert rtemp.temperature == 37.0
    tabbed = _contents(tmp_path, "WATER\nGAS", "%if(hot):\nRTEMP\n 80 /\n%\tendif")
    assert tabbed.temperature is None


def test_a_co2store_deck_without_a_temperature_runs_at_opms_default(tmp_path: Path) -> None:
    """OPM Flow's CO2STORE without RTEMP or RTEMPVD takes 60 °F (measured; review of phase
    2b), in any unit system; a stated temperature wins; a black-oil deck, or one with
    text APEX cannot read (where RTEMP may hide), stays unknown."""
    co2 = "WATER\nGAS\nCO2STORE"
    rpt = "RPTRST\n BASIC=2 DEN /"
    assert _contents(tmp_path, co2, rpt).temperature == pytest.approx(15.5556, abs=1e-4)
    field = _contents(tmp_path, f"{co2}\nFIELD", rpt)
    assert field.temperature == pytest.approx(15.5556, abs=1e-4)
    assert _contents(tmp_path, co2, f"RTEMP\n 37 /\n{rpt}").temperature == 37.0
    for unknown in ("RTEMP\n 1* /", "RTEMP\n ${t} /", "RTEMPVD\n 0 40\n 2000 /"):
        assert _contents(tmp_path, co2, f"{unknown}\n{rpt}").temperature is None
    varying = _contents(tmp_path, co2, f"RTEMPVD\n 0 40\n 2000 80 /\n{rpt}")
    assert (varying.temperature, varying.temperature_varies) == (None, True)
    assert _contents(tmp_path, "WATER\nGAS", rpt).temperature is None  # black oil


@pytest.mark.parametrize(
    "props",
    [
        "INCLUDE\n 'missing_props.inc' /",  # not found
        "INCLUDE\n '${props_file}' /",  # a templated path
        "${props_block}",  # an expression that may render RTEMP
        "<%include file='props.mako'/>",  # a Mako include
        "INCLUDE\n 'props.inc' /",  # an included file with a Mako block
    ],
)
def test_the_default_temperature_needs_the_whole_deck_read(tmp_path: Path, props: str) -> None:
    """A regression (re-review of phase 2b): RTEMP in a PROPS include APEX could not read
    was taken as absent, and temperature_c = 40 refused as "set 15.56"."""
    (tmp_path / "props.inc").write_text("% if hot:\nRTEMP\n 40 /\n% endif\n")
    deck = tmp_path / "deck.mako"
    deck.write_text(
        f"RUNSPEC\nWATER\nGAS\nCO2STORE\nGRID\nPROPS\n{props}\nSOLUTION\nSUMMARY\nSCHEDULE\n"
        "START\n 1 JAN 2020 /\nRPTRST\n BASIC=2 DEN /\nDATES\n 1 JUL 2020 /\n/\nEND\n"
    )
    assert deck_contents(deck, tmp_path).temperature is None


def test_an_include_inside_a_mako_block_may_not_render(tmp_path: Path) -> None:
    """A regression (the review of phase 2b): an INCLUDE inside ``% if`` was read as if it
    always rendered, so an RTEMP in the included file, literal there, was taken as the
    deck's, though the rendering may skip it (and OPM then takes its default)."""
    (tmp_path / "hot.inc").write_text("RTEMP\n 40 /\n")
    deck = tmp_path / "deck.mako"
    deck.write_text(
        "RUNSPEC\nWATER\nGAS\nCO2STORE\nGRID\nPROPS\n% if hot:\nINCLUDE\n 'hot.inc' /\n% endif\n"
        "SOLUTION\nSUMMARY\nSCHEDULE\nSTART\n 1 JAN 2020 /\nRPTRST\n BASIC=2 DEN /\n"
        "DATES\n 1 JUL 2020 /\n/\nEND\n"
    )
    assert deck_contents(deck, tmp_path).temperature is None
    # The same include outside the block is read as the deck's.
    deck.write_text(deck.read_text().replace("% if hot:\n", "").replace("% endif\n", ""))
    assert deck_contents(deck, tmp_path).temperature == 40.0


def test_an_include_within_a_conditional_include_may_not_render_either(tmp_path: Path) -> None:
    """A conditional include passes its condition on to the files it includes."""
    (tmp_path / "outer.inc").write_text("INCLUDE\n 'inner.inc' /\n")
    (tmp_path / "inner.inc").write_text("RTEMP\n 40 /\n")
    deck = tmp_path / "deck.mako"
    deck.write_text(
        "RUNSPEC\nWATER\nGAS\nCO2STORE\nGRID\nPROPS\n% if hot:\nINCLUDE\n 'outer.inc' /\n"
        "% endif\nSOLUTION\nSUMMARY\nSCHEDULE\nSTART\n 1 JAN 2020 /\nRPTRST\n BASIC=2 DEN /\n"
        "DATES\n 1 JUL 2020 /\n/\nEND\n"
    )
    assert deck_contents(deck, tmp_path).temperature is None


def test_a_conditionally_included_restart_control_is_uncertain(tmp_path: Path) -> None:
    """RPTRST in a file the rendering may skip does not make the densities certain."""
    (tmp_path / "rst.inc").write_text("RPTRST\n BASIC=2 DEN /\n")
    deck = tmp_path / "deck.mako"
    deck.write_text(
        "RUNSPEC\nWATER\nGAS\nGRID\nPROPS\nSOLUTION\nSUMMARY\nSCHEDULE\nSTART\n 1 JAN 2020 /\n"
        "% if dense:\nINCLUDE\n 'rst.inc' /\n% endif\nDATES\n 1 JUL 2020 /\n/\nEND\n"
    )
    assert deck_contents(deck, tmp_path).densities is None


def test_a_templated_rptrst_does_not_leave_only_basic_one(tmp_path: Path) -> None:
    """A regression (the coder's review of phase 4): RPTSCHED RESTART=1 with an RPTRST
    BASIC=2 inside a Mako block was read as only BASIC=1, and an Eclipse deck with separate
    restarts refused, though the rendered RPTRST may write every restart."""
    templated = _contents(
        tmp_path, "WATER\nGAS", "RPTSCHED\n RESTART=1 /\n% if dense:\nRPTRST\n BASIC=2 /\n% endif"
    )
    assert templated.basic_one is None  # only BASIC=1 may hold, or not
    literal = _contents(tmp_path, "WATER\nGAS", "RPTSCHED\n RESTART=1 /")
    assert literal.basic_one is True
    # A literal control writing more than BASIC=1 already means "not only BASIC=1",
    # whatever the template renders. (A literal RPTRST would not show it: once one RPTRST
    # is in a Mako block, every RPTRST counts as uncertain, the documented over-caution.)
    mixed = _contents(
        tmp_path,
        "WATER\nGAS",
        "RPTSCHED\n RESTART=1 /\nRPTSCHED\n RESTART=2 /\n% if dense:\nRPTRST\n BASIC=1 /\n% endif",
    )
    assert mixed.basic_one is False
