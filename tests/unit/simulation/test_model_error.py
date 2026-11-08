"""Schedule reading and the effective simulator of model error characterization."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.simulation import MemberInput
from apex.domain.observations import DataKey, ObservationSet
from apex.simulation.analytic import LinearModel
from apex.simulation.model_error import (
    RESIDUAL_PREFIX,
    ModelErrorSimulator,
    feature_names,
    read_schedule,
    well_features,
    well_schedules,
)

SCHEDULE = """\
-- a comment line
COMPDAT
'P1'\t2\t3\t1\t2\tOPEN\t2*\t0.3\t/  trailing text
'I1' 5 5 1 1 'OPEN' /
/
WCONHIST
'P1' 'STOP' 'LRAT' 3* / -- comment after a record
/
WCONINJH
'I1' 'WATER' 1* 100 /  -- a defaulted status is OPEN
/
TSTEP
1 2*10 / 21.0

COMPDAT
'P1' 2 3 1 1 OPEN/
'P1' 3 3 2 2 'SHUT' 2*/
/
WCONHIST
'P1' 'OPEN' 'LRAT' 100/
/
DATES
1 'FEB' 2020 /
/
WELOPEN
'P1' 'SHUT' 0 0 0 /  -- zeros are defaults: the well itself
'P1' 'OPEN' 2 3 1 /  -- a connection's status: not the well's
/
DATES
1 'MAR' 2020/
/
"""

WELLS = [("P1", "producer", ("WOPR", "WBHP")), ("I1", "injector", ("WBHP",))]


@pytest.fixture
def schedule(tmp_path: Path) -> Path:
    path = tmp_path / "Hist"
    path.write_text(SCHEDULE.replace("\n", "\r\n"))  # Windows line ends, as Brugge's
    return path


def test_records_expand_defaults_and_ignore_comments(schedule: Path) -> None:
    keywords = read_schedule(schedule)

    assert [k for k, _ in keywords] == [
        "COMPDAT", "WCONHIST", "WCONINJH", "TSTEP", "COMPDAT", "WCONHIST", "DATES",
        "WELOPEN", "DATES",
    ]  # fmt: skip
    compdat = keywords[0][1]
    assert compdat[0] == ["P1", "2", "3", "1", "2", "OPEN", None, None, "0.3"]
    assert keywords[1][1] == [["P1", "STOP", "LRAT", None, None, None]]
    assert keywords[3][1] == [["1", "10", "10"]]  # n*v repeats the value in TSTEP
    # A "/" attached to an item ends the record ("OPEN/", "2*/", "100/").
    assert keywords[4][1] == [
        ["P1", "2", "3", "1", "1", "OPEN"],
        ["P1", "3", "3", "2", "2", "SHUT", None, None],
    ]
    assert keywords[5][1] == [["P1", "OPEN", "LRAT", "100"]]
    assert len(keywords[6][1]) == 1
    assert len(keywords[8][1]) == 1


def test_report_steps_accumulate_connections_and_take_the_last_status(schedule: Path) -> None:
    # Steps 1-3 come from the TSTEP (1, 10, 10), steps 4 and 5 from the two DATES.
    p1, i1 = well_schedules(schedule, [1, 3, 4, 5], WELLS)

    # As in Eclipse, the second COMPDAT adds (3, 3, 2) to the well's connections (shut
    # ones count, as PET-fork's), in first-seen order.
    assert p1.blocks == ((2, 3, 1), (2, 3, 2), (3, 3, 2))
    np.testing.assert_array_equal(
        p1.perforated,
        [[True, True, False], [True, True, False], [True, True, True], [True, True, True]],
    )
    # STOP, STOP, OPEN (WCONHIST), then shut by the well-level WELOPEN.
    np.testing.assert_array_equal(p1.open, [False, False, True, False])
    assert i1.blocks == ((5, 5, 1),)
    np.testing.assert_array_equal(i1.open, [True] * 4)
    assert p1.n_features() == 12 * 3
    assert p1.n_features(("oil", "water", "gas")) == 14 * 3


def test_features_follow_the_phases_in_pet_forks_order() -> None:
    assert feature_names(["oil", "water"])[7:] == (
        "OIL_DEN", "WAT_DEN", "SWAT", "SOIL", "PRESS_DIFF",
    )  # fmt: skip
    assert feature_names(["oil", "water", "gas"])[7:] == (
        "OIL_DEN", "WAT_DEN", "SWAT", "GAS_DEN", "SGAS", "SOIL", "PRESS_DIFF",
    )  # fmt: skip
    # Without oil (the user, 2026-11-07) the layout keeps OIL_DEN and SOIL (0 in the inputs).
    assert feature_names(["water", "gas"]) == feature_names(["oil", "water", "gas"])
    assert feature_names(["gas"])[7:] == ("OIL_DEN", "GAS_DEN", "SGAS", "SOIL", "PRESS_DIFF")
    for bad in ([], ["water", "water"], ["brine"]):
        with pytest.raises(ValueError, match="one or more of"):
            feature_names(bad)


class _CaseFiles:
    """Restarts of a gas-water deck: no OIL_DEN; SOIL computed as 1 - SWAT - SGAS."""

    def __init__(self) -> None:
        self.case = Path("CASE")

    def static(self, name: str) -> NDArray[np.float64]:
        return np.arange(1.0, 51.0) * (1 + len(name))

    def restart(self, step: int) -> dict[str, NDArray[np.float64]]:
        cells = np.arange(50.0)
        swat = 0.7 + 0.001 * cells
        sgas = 0.3 - 0.001 * cells
        return {"PRESSURE": 100 + cells + step, "SWAT": swat, "SGAS": sgas,
                "WAT_DEN": 1030 + cells, "GAS_DEN": 690 + cells,
                "SOIL": 1.0 - swat - sgas}  # fmt: skip

    def wbhp(self, well: str, step: int) -> float:
        return 200.0 + step

    def active(self, block: tuple[int, int, int]) -> int:
        i, j, k = block
        return (i - 1) + 5 * (j - 1) + 25 * (k - 1)


def test_a_deck_without_oil_has_zero_oil_features(schedule: Path) -> None:
    """A gas-water (CO2 storage) deck: OIL_DEN, which it lacks, and SOIL are 0; the rest are
    read, as with oil."""
    wells = well_schedules(schedule, [3, 4], WELLS)
    files = _CaseFiles()
    names = feature_names(["water", "gas"])
    x = well_features(files, wells, [3, 4], ["water", "gas"])  # type: ignore[arg-type]
    per_block = x[1].reshape(2, len(names), -1)  # I1: one block, (5, 5, 1)
    values = dict(zip(names, per_block[0, :, 0], strict=True))
    assert values["OIL_DEN"] == 0.0
    assert values["SOIL"] == 0.0
    cell = files.active((5, 5, 1))
    assert values["WAT_DEN"] == 1030 + cell
    assert values["GAS_DEN"] == 690 + cell
    assert values["PRESS_DIFF"] == 203.0 - (100 + cell + 3)
    with pytest.raises(SimulationError, match="no OIL_DEN"):  # with oil, OIL_DEN is read
        well_features(files, wells, [3, 4], ["oil", "water", "gas"])  # type: ignore[arg-type]


def test_missing_steps_and_wells_are_refused(schedule: Path) -> None:
    with pytest.raises(ValueError, match="report steps \\[6\\] are beyond"):
        well_schedules(schedule, [6], WELLS)
    with pytest.raises(ValueError, match="P9 has no COMPDAT"):
        well_schedules(schedule, [1], [("P9", "producer", ("WBHP",))])


class _Files:
    """A file-based simulator stand-in: values from a linear model, a work-dir marker."""

    def __init__(self) -> None:
        self.cleaned: list[Path] = []
        self.seen: list[set[str]] = []

    def simulate(self, member: MemberInput) -> Path:
        self.seen.append(set(member.state))
        assert member.work_dir is not None
        return member.work_dir / "MODEL"

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.arange(len(keys), dtype=float)

    def clean(self, case: Path) -> None:
        self.cleaned.append(case)


class _Bias:
    """A model-error model adding theta[0] to every WOPR datum."""

    state_variable = "theta"
    n_params = 1

    def fit(self, observations: ObservationSet) -> None:
        pass

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        return [DataKey(k.time, RESIDUAL_PREFIX + k.datatype) for k in keys if "WOPR" in k.datatype]

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return np.array([theta[0] if "WOPR" in k.datatype else 0.0 for k in keys])


def test_the_effective_simulator_adds_residuals_and_hides_theta(tmp_path: Path) -> None:
    inner = _Files()
    simulator = ModelErrorSimulator(inner, _Bias())  # type: ignore[arg-type]
    keys = [DataKey("t1", "WOPR:P1"), DataKey("t1", "WBHP:P1")]
    extra = simulator.auxiliary_keys(keys)
    member = MemberInput(3, {"x": np.zeros(2), "theta": np.array([5.0])}, tmp_path)

    values = simulator.run(member, [*keys, *extra])

    np.testing.assert_array_equal(values, [0.0 + 5.0, 1.0, 5.0])  # g + r, g, then r itself
    assert inner.seen == [{"x"}]  # the deck never sees theta
    assert inner.cleaned == [tmp_path / "MODEL"]
    assert extra == [DataKey("t1", "MEC:WOPR:P1")]


def test_an_inner_simulator_without_files_is_refused() -> None:
    linear = LinearModel(outputs=[["t", "y"]], variables=["x"], matrix=[[1.0]])
    with pytest.raises(ValueError, match="runs leave files"):
        ModelErrorSimulator(linear, _Bias())
    simulator = ModelErrorSimulator(_Files(), _Bias())  # type: ignore[arg-type]
    with pytest.raises(SimulationError, match="no state variable 'theta'"):
        simulator.run(MemberInput(0, {"x": np.zeros(1)}, Path(".")), [DataKey("t", "WOPR:P")])


def test_a_failed_reading_releases_the_case_and_keeps_its_files(tmp_path: Path) -> None:
    """A regression (review of phase 2b): cleaning after a failed reading pruned the files
    keep_simulations = "failed" keeps for diagnosis; the inner simulator only releases."""

    class Failing(_Files):
        def __init__(self) -> None:
            super().__init__()
            self.released: list[Path] = []

        def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
            raise SimulationError("summary unreadable")

        def release(self, case: Path) -> None:
            self.released.append(case)

    inner = Failing()
    simulator = ModelErrorSimulator(inner, _Bias())  # type: ignore[arg-type]
    member = MemberInput(0, {"x": np.zeros(1), "theta": np.array([1.0])}, tmp_path)
    with pytest.raises(SimulationError, match="summary unreadable"):
        simulator.run(member, [DataKey("t", "WOPR:P")])
    assert inner.released == [tmp_path / "MODEL"]
    assert inner.cleaned == []
