"""Model error on a simulator whose wells come from its own input.

A test simulator states its wells' layout (``WellLayout``: mesh cells) and writes SI states
that its reader (``wells``) gives by date; ``pg_lstm`` without ``schedule_file`` takes the
layout from the composition, is sized by it, and reads each member's features through
``StepStates(ReaderDatedStates(...))`` in the metric deck units its networks were fitted in.
"""

import datetime as dt
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.core.plugins import case_state_readers, simulators
from apex.domain.case_state import SI_UNITS, CaseMesh, StateTime
from apex.domain.wells import WellSchedule
from apex.services import composition
from apex.simulation.model_error import well_features
from apex.workflows.model_error import PGLSTMModel

DATES = ["2020-02-01", "2020-03-01"]
N_CELLS = 4
BAR = SI_UNITS["bar"][0]
MD = SI_UNITS["mD"][0]


class _Reader:
    """Two SI states a month apart and the wells' pressures, from ``states.json``."""

    capabilities: ClassVar[frozenset[str]] = frozenset({"wells"})

    def __init__(self, case: Path) -> None:
        self.case = case
        self.data = json.loads((case.parent / "states.json").read_text())

    def regions(self) -> tuple[str, ...]:
        return ()

    def mesh(self, region: str | None = None) -> CaseMesh:
        raise NotImplementedError

    def static_names(self, region: str | None = None) -> frozenset[str]:
        return frozenset(self.data["static"])

    def static(self, name: str, region: str | None = None) -> NDArray[np.float64]:
        return np.asarray(self.data["static"][name], dtype=np.float64)

    def times(self, region: str | None = None) -> tuple[StateTime, ...]:
        # Cycle numbers, not report steps; an earlier state on a later day's date too.
        return tuple(
            StateTime(cycle, None, dt.date.fromisoformat(date))
            for cycle, date in ((3, "2020-01-15"), (7, DATES[0]), (8, DATES[0]), (12, DATES[1]))
        )

    def names(self, time: StateTime, region: str | None = None) -> frozenset[str]:
        return frozenset({"PRESSURE", "SWAT", "OIL_DEN", "WAT_DEN"})

    def at_step(self, step: int) -> StateTime:
        return next(t for t in self.times() if t.index == step)

    def state(
        self,
        time: StateTime,
        names: Sequence[str],
        region: str | None = None,
        *,
        skip_missing: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        arrays = self.data["states"][str(time.index)]
        return {n: np.asarray(arrays[n], dtype=np.float64) for n in names if n in arrays}

    def unit(self, name: str) -> str | None:
        return {"PRESSURE": "Pa", "WBHP": "Pa", "SWAT": "1", "OIL_DEN": "kg/m3",
                "WAT_DEN": "kg/m3", "PORO": "1", "DX": "m", "DY": "m", "DZ": "m",
                "PERMX": "m2", "PERMY": "m2", "PERMZ": "m2"}.get(name)  # fmt: skip

    def reached(self) -> StateTime | None:
        return None

    def block_cell(self, block: tuple[int, int, int] | int) -> int:
        assert isinstance(block, int)
        return block if 0 <= block < N_CELLS else -1

    def bottom_hole_pressure(self, well: str, time: StateTime) -> float:
        return float(self.data["wbhp"][well][str(time.index)])

    def report_dates(self, steps: Sequence[int]) -> list[str]:
        raise AssertionError("model error gives the report steps' dates itself")


class _Simulator:
    """States its wells: P on cells 1 and 3 (shut over the second period)."""

    asked: ClassVar[list[tuple[tuple[str, ...], tuple[dt.date, ...]]]] = []

    def __init__(self) -> None:
        pass

    def well_layout(
        self, wells: Sequence[tuple[str, str, Sequence[str]]], dates: Sequence[str]
    ) -> list[WellSchedule]:
        assert list(dates) == DATES
        return [
            WellSchedule(name, kind, tuple(outputs), (1, 3), np.ones((2, 2), dtype=bool),
                         np.array([True, False]))
            for name, kind, outputs in wells
        ]  # fmt: skip

    def state_output(self, names: Sequence[str], times: Sequence[object], **_: object) -> None:
        type(self).asked.append((tuple(names), tuple(times)))  # type: ignore[arg-type]

    def run(self, member: object, keys: object) -> NDArray[np.float64]:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    simulators.register("test_wells_sim")(_Simulator)
    case_state_readers.register("test_wells_sim")(_Reader)
    _Simulator.asked = []
    try:
        yield
    finally:
        simulators._factories.pop("test_wells_sim")
        case_state_readers.remove("test_wells_sim")


def _model(**options: object) -> PGLSTMModel:
    defaults: dict[str, object] = {
        "producers": ["P"],
        "report_steps": [1, 2],
        "report_dates": DATES,
    }
    return PGLSTMModel(**{**defaults, **options})  # type: ignore[arg-type]


def _config(model: PGLSTMModel) -> SimpleNamespace:
    return SimpleNamespace(
        simulator=SimpleNamespace(name="test_wells_sim", options={}),
        model_error=SimpleNamespace(name="pg_lstm"),
    )


def _write_case(folder: Path) -> Path:
    cells = np.arange(N_CELLS, dtype=float)
    states = {
        str(cycle): {
            "PRESSURE": (200.0 + 10 * k + cells) * BAR,
            "SWAT": 0.2 + 0.1 * cells + 0.01 * k,
            "OIL_DEN": 800.0 + cells,
            "WAT_DEN": 1000.0 + cells + k,
        }
        for k, cycle in enumerate((3, 7, 8, 12))
    }
    folder.mkdir(parents=True)
    (folder / "states.json").write_text(json.dumps({
        "static": {"DX": [100.0] * N_CELLS, "DY": [90.0] * N_CELLS, "DZ": [5.0] * N_CELLS,
                   "PERMX": list(150 * MD * (1 + cells)), "PERMY": [140 * MD] * N_CELLS,
                   "PERMZ": [15 * MD] * N_CELLS, "PORO": [0.25] * N_CELLS},
        "states": {c: {n: list(v) for n, v in a.items()} for c, a in states.items()},
        "wbhp": {"P": {"8": 180.0 * BAR, "12": 175.0 * BAR}},
    }))  # fmt: skip
    return folder / "case"


def test_the_layout_comes_from_the_simulator_and_sizes_the_networks(tmp_path: Path) -> None:
    model = _model()
    assert model.needs_well_layout
    composition._use_simulator_wells(_config(model), model)  # type: ignore[arg-type]
    assert not model.needs_well_layout
    assert [w.blocks for w in model.wells] == [(1, 3)]
    assert model.n_params > 0
    # The simulator was asked, before any run, to write the states the model reads.
    ((names, times),) = _Simulator.asked
    assert names == ("PRESSURE", "SWAT", "WAT_DEN", "OIL_DEN")
    assert times == tuple(dt.date.fromisoformat(d) for d in DATES)


def test_features_are_read_by_date_in_metric_deck_units(tmp_path: Path) -> None:
    model = _model()
    composition._use_simulator_wells(_config(model), model)  # type: ignore[arg-type]
    model.use_case_state_reader(_Reader)
    states = model.case_states(_write_case(tmp_path / "member"))
    (x,) = well_features(states, model.wells, model.report_steps, model.phases)
    names = ["DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ", "PORO", "OIL_DEN", "WAT_DEN", "SWAT",
             "SOIL", "PRESS_DIFF"]  # fmt: skip
    per_block = x.reshape(2, len(names), 2)  # (steps, features, blocks)
    # Step 1 is DATES[0], the later of its two states (cycle 8, k = 2); block 0 is cell 1.
    pressure = 200.0 + 10 * 2 + 1.0
    np.testing.assert_allclose(per_block[0, names.index("PRESS_DIFF"), 0], 180.0 - pressure)
    np.testing.assert_allclose(per_block[0, names.index("PERMX"), 1], 150.0 * 4)  # cell 3, mD
    np.testing.assert_allclose(per_block[1, names.index("WAT_DEN"), 0], 1000.0 + 1 + 3)
    np.testing.assert_allclose(per_block[1, names.index("SOIL"), 1], 1.0 - (0.2 + 0.3 + 0.03))


def test_a_deck_schedule_or_missing_dates_are_refused() -> None:
    with pytest.raises(ValueError, match="needs report_dates"):
        PGLSTMModel(producers=["P"], report_steps=[1, 2])
    # A model whose wells are already set (as a deck's schedule_file sets them).
    schedule = _model()
    schedule.use_well_layout(_Simulator().well_layout(schedule.well_specs, DATES))
    with pytest.raises(ConfigError, match="schedule_file is a deck's schedule"):
        composition._use_simulator_wells(_config(schedule), schedule)  # type: ignore[arg-type]


def test_a_model_scaling_its_targets_takes_the_simulators_limits(tmp_path: Path) -> None:
    """W3c: a simulator that states its wells' limits (``WellLimits``)
    gives them to a model that takes them, after the layout; others are left alone."""
    taken: list[dict[str, tuple[float, float]]] = []

    class _Limited(_Simulator):
        def well_limits(
            self, wells: Sequence[tuple[str, str, Sequence[str]]]
        ) -> dict[str, tuple[float, float]]:
            return {name: (500.0, 0.0) for name, _, _ in wells}

    class _Scaling(PGLSTMModel):
        def use_well_limits(self, limits: dict[str, tuple[float, float]]) -> None:
            assert not self.needs_well_layout  # after the layout
            taken.append(limits)

    simulators._factories.pop("test_wells_sim")
    simulators.register("test_wells_sim")(_Limited)
    model = _Scaling(producers=["P"], report_steps=[1, 2], report_dates=DATES)
    composition._use_simulator_wells(_config(model), model)  # type: ignore[arg-type]
    assert taken == [{"P": (500.0, 0.0)}]
    plain = _model()  # pg_lstm takes no limits
    composition._use_simulator_wells(_config(plain), plain)  # type: ignore[arg-type]
    assert taken == [{"P": (500.0, 0.0)}]
