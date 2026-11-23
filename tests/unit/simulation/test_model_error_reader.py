"""Model error's states through a case-state reader with wells.

The Eclipse-format reader's ``wells`` capability, seen through :class:`ReaderBlockStates`,
gives the features ``CaseFiles`` gives, byte for byte (the production path stays
``CaseFiles`` until a reader of another simulator needs the reader's); with ``metric`` a
reader's arrays are converted to the metric deck units the networks were fitted in. The
case files are written with resdata (the ``write_case`` fixture of ``tests/conftest.py``).
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from apex.core.case_state import WellsReader
from apex.core.errors import SimulationError
from apex.domain.case_state import SI_UNITS
from apex.simulation.eclipse_files import EclipseFiles
from apex.simulation.model_error import (
    METRIC_FEATURE_UNITS,
    STATIC,
    CaseFiles,
    ReaderBlockStates,
    WellSchedule,
    well_features,
)

SHAPE = (3, 2, 2)
#: Cell (3, 2, 1) (Eclipse order: index 5) is inactive.
ACTNUM = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
DATES = [dt.date(2020, 1, 1), dt.date(2020, 2, 1), dt.date(2020, 3, 1)]
N_ACTIVE = sum(ACTNUM)


def _case(write_case: Callable[..., Path], folder: Path, units: int = 1) -> Path:
    cells = np.arange(N_ACTIVE, dtype=float)
    restarts = {
        date: {
            "PRESSURE": 200.0 + 3.0 * cells + 7.0 * k,
            "SWAT": 0.2 + 0.01 * cells + 0.05 * k,
            "SGAS": 0.05 + 0.002 * cells,
            "OIL_DEN": 800.0 + cells,
            "WAT_DEN": 1000.0 + 0.5 * cells + k,
            "GAS_DEN": 150.0 + 2.0 * cells,
        }
        for k, date in enumerate(DATES)
    }
    return write_case(
        folder / "C",
        SHAPE,
        restarts,
        actnum=ACTNUM,
        units=units,
        porosity=0.2 + 0.001 * cells,
        init_arrays={
            "DX": 100.0 + cells,
            "DY": 90.0 + cells,
            "DZ": 5.0 + 0.1 * cells,
            "PERMX": 150.0 + 10.0 * cells,
            "PERMY": 140.0 + 10.0 * cells,
            "PERMZ": 15.0 + cells,
        },
        wbhp={"P1": [180.0, 175.5], "I1": [260.25, 270.0]},
    )


def _wells() -> list[WellSchedule]:
    return [
        WellSchedule(
            "P1", "producer", ("WOPR",), ((1, 1, 1), (1, 1, 2)),
            np.array([[True, False], [True, True]]), np.array([True, True]),
        ),
        # Its second block is the inactive cell: 0, PET-fork's padding, both ways.
        WellSchedule(
            "I1", "injector", ("WWIR",), ((2, 2, 2), (3, 2, 1)),
            np.array([[True, True], [True, True]]), np.array([True, False]),
        ),
    ]  # fmt: skip


def test_the_eclipse_reader_has_wells(write_case: Callable[..., Path], tmp_path: Path) -> None:
    reader = EclipseFiles(_case(write_case, tmp_path))
    assert isinstance(reader, WellsReader)
    assert "wells" in EclipseFiles.capabilities
    assert reader.block_cell((3, 2, 1)) == -1
    assert reader.block_cell((1, 1, 1)) == 0
    assert reader.block_cell(4) == 4
    assert reader.block_cell(N_ACTIVE) == -1
    assert reader.bottom_hole_pressure("I1", reader.at_step(2)) == 270.0
    assert reader.report_dates([1, 2]) == ["2020-02-01", "2020-03-01"]
    assert reader.unit("WBHP") == "bar"
    with pytest.raises(SimulationError, match="no WBHP:X"):
        reader.bottom_hole_pressure("X", reader.at_step(1))


@pytest.mark.parametrize("phases", [("oil", "water"), ("water",), ("oil", "water", "gas")])
def test_reader_features_equal_case_files_byte_for_byte(
    write_case: Callable[..., Path], tmp_path: Path, phases: tuple[str, ...]
) -> None:
    case = _case(write_case, tmp_path)
    steps = [1, 2]
    expected = well_features(CaseFiles(case), _wells(), steps, phases)
    states = ReaderBlockStates(EclipseFiles(case))
    got = well_features(states, _wells(), steps, phases)
    assert len(got) == len(expected)
    for a, b in zip(got, expected, strict=True):
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()
    assert states.report_dates(steps) == CaseFiles(case).report_dates(steps)
    assert np.any(expected[0] != 0.0)


def test_metric_converts_a_field_decks_arrays(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    """A reader of another unit system is converted to bar, m, mD, kg/m3 (the user's choice
    of 2026-11-21 for simulators that write SI); here a field deck's psia, ft, lb/ft3."""
    case = _case(write_case, tmp_path, units=2)
    raw = CaseFiles(case)
    states = ReaderBlockStates(EclipseFiles(case), metric=True)
    psia, bar = SI_UNITS["psia"][0], SI_UNITS["bar"][0]
    ft, lb = SI_UNITS["ft"][0], SI_UNITS["lb/ft3"][0]
    np.testing.assert_allclose(
        states.restart(1)["PRESSURE"], raw.restart(1)["PRESSURE"] * psia / bar
    )
    np.testing.assert_allclose(states.restart(2)["OIL_DEN"], raw.restart(2)["OIL_DEN"] * lb)
    np.testing.assert_allclose(states.static("DX"), raw.static("DX") * ft)
    np.testing.assert_array_equal(states.static("PERMX"), raw.static("PERMX"))  # mD both
    np.testing.assert_array_equal(states.restart(1)["SWAT"], raw.restart(1)["SWAT"])
    assert states.wbhp("P1", 1) == pytest.approx(180.0 * psia / bar, rel=1e-15)
    # Every feature the table names, statics and dynamics, against its own unit's factor.
    for name, target in METRIC_FEATURE_UNITS.items():
        if name == "WBHP":
            continue
        source = EclipseFiles(case).unit(name)
        assert source is not None, name
        factor = SI_UNITS[source][0] / SI_UNITS[target][0]
        if name in STATIC:
            np.testing.assert_allclose(states.static(name), raw.static(name) * factor, err_msg=name)
        else:
            np.testing.assert_allclose(
                states.restart(1)[name], raw.restart(1)[name] * factor, err_msg=name
            )
    # Through the features too: PRESS_DIFF is WBHP - PRESSURE, both in bar.
    metric_features = well_features(states, _wells(), [1, 2], ("oil", "water", "gas"))
    raw_features = well_features(raw, _wells(), [1, 2], ("oil", "water", "gas"))
    np.testing.assert_allclose(metric_features[0][:, -2:], raw_features[0][:, -2:] * psia / bar)
    # Without metric, the deck's own units, never converted.
    np.testing.assert_array_equal(
        ReaderBlockStates(EclipseFiles(case)).restart(1)["PRESSURE"], raw.restart(1)["PRESSURE"]
    )


def test_a_missing_report_step_is_refused(write_case: Callable[..., Path], tmp_path: Path) -> None:
    states = ReaderBlockStates(EclipseFiles(_case(write_case, tmp_path)))
    with pytest.raises(SimulationError, match="no state at report step 7"):
        states.restart(7)


class _NoUnits:
    """A reader that cannot state its arrays' units (``unit`` is ``None``)."""

    def __init__(self, reader: EclipseFiles) -> None:
        self.reader = reader
        self.case = reader.case

    def __getattr__(self, name: str) -> object:
        return getattr(self.reader, name)

    def unit(self, name: str) -> None:
        return None


def test_metric_refuses_an_array_of_unstated_unit(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    states = ReaderBlockStates(_NoUnits(EclipseFiles(_case(write_case, tmp_path))), metric=True)  # type: ignore[arg-type]
    with pytest.raises(SimulationError, match="C: PERMX: no conversion to SI"):
        states.static("PERMX")


def test_a_numpy_integer_is_an_active_index(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    reader = EclipseFiles(_case(write_case, tmp_path))
    assert reader.block_cell(np.int64(3)) == 3  # type: ignore[arg-type]


def test_the_eclipse_reader_gives_the_wells_vectors_as_the_summary(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    """W3: the reader's well vectors, WMCTL included, are the summary's
    report values that the lab's forecasting model reads (its ``_vector``), at each step."""
    from apex.core.case_state import WellControlsReader
    from apex.simulation.deck import report_values

    cells = np.arange(N_ACTIVE, dtype=float)
    case = write_case(
        tmp_path / "C", SHAPE, {d: {"PRESSURE": 200.0 + cells} for d in DATES}, actnum=ACTNUM,
        wbhp={"P1": [180.0, 175.5]},
        well_vectors={"WMCTL:P1": [1.0, 7.0], "WOPR:P1": [250.0, 212.5]},
    )  # fmt: skip
    reader = EclipseFiles(case)
    assert isinstance(reader, WellControlsReader)
    assert "well_controls" in EclipseFiles.capabilities
    assert reader.has_well_vector("P1", "WMCTL")
    assert not reader.has_well_vector("P1", "WWIR")
    raw = CaseFiles(case)
    for name in ("WMCTL", "WOPR", "WBHP"):
        expected = report_values(raw.summary, f"{name}:P1")
        got = [reader.well_vector("P1", name, reader.at_step(s)) for s in (1, 2)]
        assert got == [float(v) for v in expected]
    assert reader.well_vector("P1", "WMCTL", reader.at_step(2)) == 7.0
    with pytest.raises(SimulationError, match="no WWIR:P1"):
        reader.well_vector("P1", "WWIR", reader.at_step(1))
    # A step outside the summary's reports is refused, never another step's value (the
    # review: step 0, an initial restart, read the last report through index -1).
    from apex.domain.case_state import StateTime

    for step in (0, 3):
        with pytest.raises(SimulationError, match=f"no WOPR:P1 at report step {step}"):
            reader.well_vector("P1", "WOPR", StateTime(step, None, None))
