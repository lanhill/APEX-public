"""Model error on a surrogate: ``p(m) + r(m, theta)`` from its states.

A linear case whose deck writes its forecasts and a pressure per report step to files
(``StatesDeck``, a case-file simulator), a model-error model whose residual reads that
pressure (``PressureOffset``: ``r = theta * mean pressure / 100`` at each report date, a
:class:`~apex.core.simulation.BlockStateModel`), and a test surrogate, the exact affine map
of the deck that also gives block states (``StateAffine``), whose pressures are the deck's
plus a planted ``DELTA``.

The independent reference is the closed form: every member's stored forecast must be
``G x + r`` with ``r`` from its own source's pressure, the deck's (``H x + P0``) for a
member the deck ran, the surrogate's (``H x + P0 + DELTA``) for one it forecast; one source
per member and forecast, never mixed. The surrogate is exact on ``G x`` (to round-off, as
in ``test_surrogate_history_matching.py``), so the tolerance is ``1e-9``.
"""

import importlib.util
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.core.options import InputFile
from apex.core.plugins import model_error_models, simulators, surrogate_models
from apex.core.simulation import BlockStates, DatedStates, MemberInput, StepStates
from apex.core.surrogates import SurrogateInputs
from apex.domain.observations import DataKey, ObservationSet
from apex.services import composition, surrogates
from apex.simulation.model_error import (
    RESIDUAL_PREFIX,
    WellSchedule,
    feature_names,
    well_features,
)
from apex.storage.surrogate_files import read_surrogate
from apex.testing import assert_block_states_conform


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_mec", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()
N_STATE = 3
DATES = ("2000-02-01", "2000-03-01", "2000-04-01", "2000-05-01")
DATATYPES = ("A", "B")
KEYS = tuple(DataKey(t, d) for t in DATES for d in DATATYPES)
RNG = np.random.default_rng(5)
G = RNG.standard_normal((len(KEYS), N_STATE))
#: Each report step's pressure in its two cells: ``H[s] x + P0``.
H = RNG.standard_normal((len(DATES), 2, N_STATE))
P0 = 200.0
#: The surrogate's pressures are the deck's plus this (a planted difference of sources).
DELTA = 7.0
TRUTH = np.array([0.4, -0.6, 0.3])
VARIANCE = 0.04
ROUND_OFF = 1e-9
ESMDA = (
    'name = "esmda"\noptions = { inflation = [2, 2] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\nmax_iterations = 4'
)
DECK = """RUNSPEC
START
 1 'JAN' 2000 /
GRID
INIT
SCHEDULE
DATES
 1 'FEB' 2000 /
/
TSTEP
 29 31 30 /
"""


def _pressure(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """The deck's pressures ``(n_steps, 2)`` of one member."""
    pressure: NDArray[np.float64] = np.einsum("scn,n->sc", H, x) + P0
    return pressure


def _residual(pressure: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """``r`` at every key from a member's pressures (closed form of ``PressureOffset``)."""
    per_step = theta * pressure.mean(axis=1) / 100.0
    return np.repeat(per_step, len(DATATYPES))


class StatesDeck:
    """``G x`` and a pressure per report step, written to the member's folder (a deck).

    ``date_shift`` (days; a class attribute, so the stack's options are unchanged) makes its
    runs report other dates than its deck's text.
    """

    needs_work_dir = True
    date_shift = 0

    def __init__(self, deck_template_file: InputFile) -> None:
        self.deck_template_file = deck_template_file

    def simulate(self, member: MemberInput) -> Path:
        assert member.work_dir is not None
        assert set(member.state) == {"x"}  # the model error's parameters stay out
        x = np.asarray(member.state["x"], dtype=np.float64)
        np.save(member.work_dir / "MODEL.npy", G @ x)
        np.save(member.work_dir / "MODEL_P.npy", _pressure(x))
        shifted = [
            (np.datetime64(d) + np.timedelta64(self.date_shift, "D")).astype(str) for d in DATES
        ]
        (member.work_dir / "MODEL_DATES.txt").write_text("\n".join(shifted))
        return member.work_dir / "MODEL"

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        values = np.load(case.with_suffix(".npy"))
        return np.array([values[KEYS.index(k)] for k in keys])

    def clean(self, case: Path) -> None:
        for path in (case.with_suffix(".npy"), Path(f"{case}_P.npy"), Path(f"{case}_DATES.txt")):
            path.unlink()

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        case = self.simulate(member)
        values = self.read(case, keys)
        self.clean(case)
        return values


class FileStates:
    """A ``StatesDeck`` run's block states (its files)."""

    def __init__(self, case: Path) -> None:
        self.case = case
        self.pressure = np.load(f"{case}_P.npy")
        self.dates = Path(f"{case}_DATES.txt").read_text().split()

    @property
    def label(self) -> str:
        return self.case.name

    def active(self, block: tuple[int, int, int]) -> int:
        return block[0] - 1

    def static(self, name: str) -> NDArray[np.float64]:
        return np.ones(2)

    def restart(self, step: int) -> dict[str, NDArray[np.float64]]:
        return {"PRESSURE": self.pressure[step - 1]}

    def wbhp(self, well: str, step: int) -> float:
        return 0.0

    def report_dates(self, steps: Sequence[int]) -> list[str]:
        return [self.dates[s - 1] for s in steps]


class PressureOffset:
    """``r = theta * mean pressure / 100`` at each report date, from block states."""

    n_params = 1
    report_steps: tuple[int, ...] = (1, 2, 3, 4)

    def __init__(self, state_variable: str = "nn_params") -> None:
        self.state_variable = state_variable

    def fit(self, observations: ObservationSet) -> None:
        pass

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        return [DataKey(k.time, RESIDUAL_PREFIX + k.datatype) for k in keys]

    def case_states(self, case: Path) -> BlockStates:
        return FileStates(case)

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return self.state_residuals(self.case_states(case), theta, keys)

    def state_residuals(
        self, states: BlockStates, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        dates = states.report_dates(self.report_steps)
        out = np.zeros(len(keys))
        for i, key in enumerate(keys):
            step = self.report_steps[dates.index(key.time)]
            out[i] = theta[0] * states.restart(step)["PRESSURE"].mean() / 100.0
        return out


class FilesOnly(PressureOffset):
    """The same model without block states: it reads a case's files only."""

    state_residuals = None  # type: ignore[assignment]


class PlantedStates:
    """A surrogate's states: the deck's pressures plus ``DELTA``, by date."""

    def __init__(self, x: NDArray[np.float64], dates: Sequence[str]) -> None:
        self.pressure = _pressure(x) + DELTA
        self.dates = list(dates)

    def active(self, block: tuple[int, int, int]) -> int:
        return block[0] - 1

    def static(self, name: str) -> NDArray[np.float64]:
        return np.ones(2)

    def dynamic(self, date: str) -> dict[str, NDArray[np.float64]]:
        return {"PRESSURE": self.pressure[DATES.index(date)]}

    def wbhp(self, well: str, date: str) -> float:
        return 0.0


class StateAffine(cases.Affine):  # type: ignore[name-defined,misc]
    """The exact affine surrogate, which also gives block states (the planted ones)."""

    def predict_block_states(self, inputs: SurrogateInputs, dates: Sequence[str]) -> DatedStates:
        x = np.asarray(inputs["x"], dtype=np.float64)[:, 0]
        return PlantedStates(x, dates)


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    simulators.register("test_states_deck")(StatesDeck)
    model_error_models.register("test_pressure_offset")(PressureOffset)
    model_error_models.register("test_files_only")(FilesOnly)
    surrogate_models.register("test_state_affine")(StateAffine)
    surrogate_models.register("test_plain_affine")(cases.Affine)
    surrogates.fitted_model.cache_clear()
    try:
        yield
    finally:
        simulators.remove("test_states_deck")
        model_error_models.remove("test_pressure_offset")
        model_error_models.remove("test_files_only")
        surrogate_models.remove("test_state_affine")
        surrogate_models.remove("test_plain_affine")


def _case(
    folder: Path, *, name: str = "hm", surrogate: str = "", model: str = "test_pressure_offset",
    deck: str = DECK, members: int = 40, method: str = "",
) -> Path:  # fmt: skip
    # The deck alone in its folder: the stack's deck digest is that folder's.
    (folder / "deck").mkdir(parents=True, exist_ok=True)
    (folder / "deck" / "MODEL.DATA").write_text(deck)
    noise = np.random.default_rng(3).standard_normal(len(KEYS)) * np.sqrt(VARIANCE)
    observed = G @ TRUTH + _residual(_pressure(TRUTH), 0.5) + noise
    rows = ["time,datatype,value,variance"] + [
        f"{k.time},{k.datatype},{float(v)!r},{VARIANCE}"
        for k, v in zip(KEYS, observed, strict=True)
    ]
    (folder / "observations.csv").write_text("\n".join(rows) + "\n")
    np.savez(folder / "truth.npz", x=TRUTH)
    path = folder / f"{name}.toml"
    path.write_text(f"""
[run]
name = "{name}"
seed = 2026

[[state.variables]]
name = "x"
size = {N_STATE}

[[state.variables]]
name = "nn_params"
size = 1

[prior]
ensemble_size = {members}
generator = {{ name = "gaussian", options = {{ mean = 0.0, std = 1.5 }} }}

[prior.variables.nn_params]
name = "gaussian"
options = {{ mean = 0.0, std = 1.0 }}

[observations]
file = "observations.csv"

[case]
truth_file = "truth.npz"

[simulator]
name = "test_states_deck"
options = {{ deck_template_file = "deck/MODEL.DATA" }}

[execution]
name = "serial"

[method]
{method or ESMDA}

[model_error]
name = "{model}"

{surrogate}
""")
    return path


def _build(folder: Path, plugin: str = "test_state_affine", box: float = 2.0) -> Path:
    """Build the surrogate on the model-error case (``base.toml``); return its manifest."""
    path = folder / f"build_{plugin}.toml"
    path.write_text(f"""
[surrogate]
name = "{plugin}"
model = "{plugin}"
seed = 7

[case]
config_file = "base.toml"

[design]
plan = "lhs"
training = 30
held_out = 10
box = {{ x = [{-box}, {box}] }}
""")
    result = surrogates.build(path)
    assert result.summary.gate is not None
    assert result.summary.gate.passed
    return folder / "runs" / plugin / "model" / "surrogate.json"


def _section(model: Path, **options: object) -> str:
    lines = ["[surrogate]", f'model_file = "{model}"']
    lines += [f"{key} = {json.dumps(value)}" for key, value in options.items()]
    return "\n".join(lines)


@pytest.fixture
def built(tmp_path: Path) -> Path:
    _case(tmp_path, name="base")
    return _build(tmp_path)


# ------------------------------------------------------------------ the build: g alone


def test_a_build_on_a_model_error_case_learns_the_deck_alone(tmp_path: Path, built: Path) -> None:
    stored = read_surrogate(built)
    assert stored.manifest.provenance.model_error_left_out
    snapshots = np.load(tmp_path / "runs" / "test_state_affine" / "snapshots.npz")
    assert not any("nn_params" in name for name in snapshots.files)
    # Exact on G x: the deck's forecasts without any residual.
    model = StateAffine.from_arrays(stored.manifest.model, stored.arrays)
    x = np.array([[0.3], [-0.2], [0.1]])
    np.testing.assert_allclose(model.predict({"x": x}, KEYS)[:, 0], G @ x[:, 0], atol=ROUND_OFF)
    state = json.loads((tmp_path / "runs" / "test_state_affine" / "state.json").read_text())
    assert state["fingerprint_parts"]["model_error"] == "left out: the deck alone"


def test_a_case_without_model_error_keeps_its_provenance(tmp_path: Path) -> None:
    """The case digest leaves [model_error] out as it always read a case without one."""
    config = composition.load_config(_case(tmp_path, name="plain"))
    plain = config.model_copy(update={"model_error": None})
    stack = composition.forecast_stack(plain, checks=False)
    without = surrogates.stack_provenance(plain, stack.simulator)
    with_mec = surrogates.stack_provenance(config, stack.simulator)
    assert without.case_digest == with_mec.case_digest
    assert not without.model_error_left_out
    assert "model_error_left_out" not in without.model_dump()


# ------------------------------------------------------------ history matching, one source


def test_each_member_takes_g_and_r_from_one_source(tmp_path: Path, built: Path) -> None:
    """Members inside the domain: the surrogate's p and r from its states; outside it (the
    box is [-2, 2], the prior's spread 1.5, so some members leave it): the deck's g and r
    from its files. Never mixed."""
    config = _case(tmp_path, surrogate=_section(built, outside="simulator", check_members=4))
    composition.start(config)
    checkpoints = tmp_path / "runs" / "hm" / "checkpoints"
    seen = set()
    for path in sorted(checkpoints.glob("iter_*.npz")):
        saved = np.load(path)
        state, predicted = saved["state"], saved["predicted"]
        for column, source in enumerate(saved["forecast_source"]):
            x, theta = state[:N_STATE, column], float(state[N_STATE, column])
            pressure = _pressure(x) + (DELTA if source == "surrogate" else 0.0)
            expected = G @ x + _residual(pressure, theta)
            np.testing.assert_allclose(predicted[:, column], expected, rtol=0, atol=ROUND_OFF)
            seen.add(str(source))
    assert seen == {"surrogate", "simulator"}


def test_the_checks_and_the_posterior_are_the_decks(tmp_path: Path, built: Path) -> None:
    """The posterior is forecast again by the deck: its r comes from the deck's files, and
    is saved for every member (the only residuals a surrogate run saves)."""
    config = _case(tmp_path, surrogate=_section(built, outside="simulator", check_members=4))
    composition.start(config)
    full = sorted((tmp_path / "runs" / "hm" / "checkpoints").glob("full_*.npz"))
    assert full
    saved = np.load(full[-1])
    assert list(saved["auxiliary_keys"]) == [f"MEC:{k.datatype}@{k.time}" for k in KEYS]
    assert len(saved["reused_ids"]) == 0  # every member runs, for its residuals
    for column in range(saved["state"].shape[1]):
        x, theta = saved["state"][:N_STATE, column], float(saved["state"][N_STATE, column])
        r = _residual(_pressure(x), theta)
        np.testing.assert_allclose(saved["predicted"][:, column], G @ x + r, atol=ROUND_OFF)
        np.testing.assert_allclose(saved["auxiliary_predicted"][:, column], r, atol=ROUND_OFF)
    iterate = np.load(tmp_path / "runs" / "hm" / "checkpoints" / "iter_0.npz")
    assert "auxiliary_predicted" not in iterate.files


def test_the_dates_are_checked_before_any_member_is_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One deck run checks the dates first, even with no checks and no fallback to it (a
    surrogate built over a box every member lies in)."""
    _case(tmp_path, name="base")
    wide = _section(_build(tmp_path, box=8.0), check_every=0, outside="stop")
    found = composition.start(_case(tmp_path, surrogate=wide, members=8))
    events = [
        json.loads(line)
        for line in (tmp_path / "runs" / "hm" / "events.jsonl").read_text().splitlines()
    ]
    assert any(e.get("event") == "surrogate_dates_checked" for e in events)
    assert found.posterior.shape[1] == 8
    # Checkpointed: a resume does not check them again.
    state = json.loads((tmp_path / "runs" / "hm" / "state.json").read_text())
    assert state["surrogate"]["dates_checked"] is True
    monkeypatch.setattr(StatesDeck, "date_shift", 1)
    with pytest.raises(ConfigError, match="report dates read from its text differ"):
        composition.start(_case(tmp_path, name="shifted", surrogate=wide, members=8))
    assert not list((tmp_path / "runs" / "shifted" / "checkpoints").glob("iter_*"))


# ------------------------------------------------------------------------------ refusals


def test_what_cannot_combine_is_refused(tmp_path: Path, built: Path) -> None:
    with pytest.raises(ConfigError, match="reads a run's files only"):
        composition.start(_case(tmp_path, name="files", surrogate=_section(built),
                                model="test_files_only"))  # fmt: skip
    with pytest.raises(ConfigError, match="refit"):
        composition.start(_case(tmp_path, name="refit",
                                surrogate=_section(built, on_check_failure="refit")))  # fmt: skip
    plain = _build(tmp_path, plugin="test_plain_affine")
    with pytest.raises(ConfigError, match="needs its model to give block states"):
        composition.start(_case(tmp_path, name="plain", surrogate=_section(plain)))


def test_the_model_error_simulator_refuses_what_it_cannot_combine() -> None:
    """Built directly: a model that extrapolates, missing step dates, a surrogate whose
    model predicts from companion runs (it gives no block states then)."""
    from apex.simulation.model_error import ModelErrorSimulator

    class Proxy:
        def __init__(self, gives: bool) -> None:
            self.gives = gives

        def gives_block_states(self) -> bool:
            return self.gives

        def block_states(self, member: MemberInput, dates: Sequence[str]) -> DatedStates:
            raise AssertionError("not reached")

        def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
            raise AssertionError("not reached")

    class Extrapolating(PressureOffset):
        extrapolates = True

        def forecast_state(self, prior: object, posterior: object, keys: object) -> dict:  # type: ignore[type-arg]
            return {}

        def member_residuals(self, *args: object) -> NDArray[np.float64]:
            return np.zeros(0)

    dates = dict(enumerate(DATES, start=1))
    ModelErrorSimulator(Proxy(True), PressureOffset(), step_dates=dates)  # accepted
    with pytest.raises(ValueError, match="extrapolates"):
        ModelErrorSimulator(Proxy(True), Extrapolating(), step_dates=dates)
    with pytest.raises(ValueError, match="report date of every report step"):
        ModelErrorSimulator(Proxy(True), PressureOffset(), step_dates={1: DATES[0]})
    with pytest.raises(ValueError, match="companion runs"):
        ModelErrorSimulator(Proxy(False), PressureOffset(), step_dates=dates)


def test_a_schedule_the_text_cannot_tell_is_refused(tmp_path: Path) -> None:
    deck = DECK.replace("TSTEP\n 29 31 30 /", "TSTEP\n ${steps} /")
    config = composition.load_config(_case(tmp_path, deck=deck))
    with pytest.raises(ConfigError, match="cannot tell them"):
        composition.model_error_step_dates(config)
    good = composition.load_config(_case(tmp_path / "good"))
    assert composition.model_error_step_dates(good) == dict(enumerate(DATES, start=1))


def test_a_deck_reporting_other_dates_is_refused(
    tmp_path: Path, built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deck's runs report other dates than its text: the first deck run stops the run."""
    monkeypatch.setattr(StatesDeck, "date_shift", 1)
    config = _case(tmp_path, surrogate=_section(built, outside="simulator", check_members=4))
    with pytest.raises(ConfigError, match="report dates read from its text differ"):
        composition.start(config)


# -------------------------------------------------------------- the features, two sources


def test_block_states_give_the_features_a_case_gives() -> None:
    """``well_features`` from a case's files and from dated states seen by step: equal."""
    rng = np.random.default_rng(1)
    names = feature_names(("oil", "water"))
    statics = {n: rng.uniform(1, 2, 4) for n in ("DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ",
                                                  "PORO")}  # fmt: skip
    dynamics = [
        {n: rng.uniform(0, 1, 4) for n in ("PRESSURE", "SOIL", "SWAT", "OIL_DEN", "WAT_DEN")}
        for _ in DATES
    ]
    bhp = rng.uniform(100, 200, len(DATES))

    class Files:
        label = "case"

        def active(self, block: tuple[int, int, int]) -> int:
            return block[0] - 1

        def static(self, name: str) -> NDArray[np.float64]:
            return statics[name]

        def restart(self, step: int) -> dict[str, NDArray[np.float64]]:
            return dynamics[step - 1]

        def wbhp(self, well: str, step: int) -> float:
            return float(bhp[step - 1])

        def report_dates(self, steps: Sequence[int]) -> list[str]:
            return [DATES[s - 1] for s in steps]

    class Dated:
        def active(self, block: tuple[int, int, int]) -> int:
            return block[0] - 1

        def static(self, name: str) -> NDArray[np.float64]:
            return statics[name]

        def dynamic(self, date: str) -> dict[str, NDArray[np.float64]]:
            return dynamics[DATES.index(date)]

        def wbhp(self, well: str, date: str) -> float:
            return float(bhp[DATES.index(date)])

    wells = [
        WellSchedule(name="P", kind="producer", outputs=("WOPR",), blocks=((1, 1, 1), (3, 1, 1)),
                     perforated=np.array([[True, False], [True, True], [True, True],
                                          [False, True]]),
                     open=np.ones(4, dtype=bool)),
    ]  # fmt: skip
    steps = [1, 2, 3, 4]
    step_dates = dict(zip(steps, DATES, strict=True))
    assert isinstance(StepStates(Dated(), step_dates), BlockStates)
    assert len(names) == well_features(Files(), wells, steps)[0].shape[1] // 2
    assert_block_states_conform(Files(), Dated(), step_dates, wells, rtol=0.0)
    # A source off by a little in one cell's saturation, or at shifted dates, is caught.
    dynamics[2]["SWAT"][0] += 1e-3
    with pytest.raises(AssertionError, match="well P's model inputs"):
        assert_block_states_conform(Files(), _Shifted(Dated(), dynamics, 2), step_dates, wells)
    dynamics[2]["SWAT"][0] -= 1e-3
    with pytest.raises(AssertionError, match="report dates differ"):
        assert_block_states_conform(Files(), Dated(), {**step_dates, 4: "2000-05-02"}, wells)


class _Shifted:
    """Dated states whose dynamic arrays at one date are copies taken before a change."""

    def __init__(
        self, source: DatedStates, dynamics: list[dict[str, NDArray[np.float64]]], index: int
    ) -> None:
        self.source = source
        self.frozen = {k: v.copy() for k, v in dynamics[index].items()}
        self.frozen["SWAT"][0] -= 1e-3  # as before the change
        self.date = DATES[index]

    def active(self, block: tuple[int, int, int]) -> int:
        return self.source.active(block)

    def static(self, name: str) -> NDArray[np.float64]:
        return self.source.static(name)

    def dynamic(self, date: str) -> dict[str, NDArray[np.float64]]:
        return self.frozen if date == self.date else self.source.dynamic(date)

    def wbhp(self, well: str, date: str) -> float:
        return self.source.wbhp(well, date)


# ---------------------------------------------------------------- resume, plan, refusals


def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path: Path, built: Path) -> None:
    """Interrupted after its first step and resumed, bit for bit as uninterrupted; the dates
    are checked once (the flag is checkpointed)."""
    section = _section(built, outside="simulator", check_members=4)
    whole = composition.start(_case(tmp_path, name="whole", surrogate=section))
    composition.start(_case(tmp_path, name="parts", surrogate=section), max_steps=1)
    resumed = composition.resume(tmp_path / "runs" / "parts")
    np.testing.assert_array_equal(resumed.posterior, whole.posterior)
    for name in ("iter_0", "iter_1", "iter_2"):
        a = np.load(tmp_path / "runs" / "whole" / "checkpoints" / f"{name}.npz")
        b = np.load(tmp_path / "runs" / "parts" / "checkpoints" / f"{name}.npz")
        for key in ("state", "predicted", "forecast_source"):
            np.testing.assert_array_equal(a[key], b[key])
    events = [
        json.loads(line)
        for line in (tmp_path / "runs" / "parts" / "events.jsonl").read_text().splitlines()
    ]
    assert sum(e.get("event") == "surrogate_dates_checked" for e in events) == 1
    # The check is a simulator run of its own label, never an ensemble forecast.
    done = [e for e in events if e.get("event") == "forecast_done"]
    checks = [e for e in done if e.get("label") == "date check (simulator)"]
    assert len(checks) == 1
    assert checks[0]["n_ok"] == 1


def test_the_plan_counts_the_date_check_runs(tmp_path: Path, built: Path) -> None:
    from apex.services.validation import validate_history_match
    from apex.workflows.history_matching import DATE_CHECK_RUNS

    config = _case(tmp_path, surrogate=_section(built, check_every=0, outside="stop"))
    found = validate_history_match(config)
    assert found.valid, found.errors
    assert found.plan is not None
    section = composition.load_config(config).surrogate
    assert section is not None
    base, _ = surrogates.planned_simulator_runs(
        section, found.plan.max_forecasts, found.plan.members
    )
    assert base is not None
    assert found.plan.max_simulator_runs == base + DATE_CHECK_RUNS
    assert any("checks the deck's report dates" in w for w in found.plan.warnings)


def test_a_failed_date_check_tries_other_members_then_refuses(
    tmp_path: Path, built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check run fails: the next members are tried; none succeeding refuses the run."""
    from apex.workflows.history_matching import DATE_CHECK_RUNS

    calls: list[int] = []
    original = StatesDeck.simulate

    def failing(self: StatesDeck, member: MemberInput) -> Path:
        assert member.work_dir is not None
        if member.work_dir.parent.name == "check_dates":
            calls.append(member.member_id)
            if len(calls) < 2:
                raise SimulationError("fails on purpose")
        return original(self, member)

    monkeypatch.setattr(StatesDeck, "simulate", failing)
    section = _section(built, check_every=0, outside="simulator")
    composition.start(_case(tmp_path, surrogate=section), max_steps=1)
    assert len(calls) == 2  # the first failed, the second checked them

    def always(self: StatesDeck, member: MemberInput) -> Path:
        assert member.work_dir is not None
        if member.work_dir.parent.name == "check_dates":
            calls.append(member.member_id)
            raise SimulationError("fails on purpose")
        return original(self, member)

    monkeypatch.setattr(StatesDeck, "simulate", always)
    calls.clear()
    with pytest.raises(ConfigError, match="no deck run succeeded to check"):
        composition.start(_case(tmp_path, name="never", surrogate=section), max_steps=1)
    assert len(calls) == DATE_CHECK_RUNS


def test_reported_steps_short_of_the_model_are_a_date_error(
    tmp_path: Path, built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deck run reporting fewer steps than the model reads: refused, not a failed member."""
    original = StatesDeck.simulate

    def short(self: StatesDeck, member: MemberInput) -> Path:
        case = original(self, member)
        Path(f"{case}_DATES.txt").write_text("\n".join(DATES[:2]))
        return case

    monkeypatch.setattr(StatesDeck, "simulate", short)

    class ShortFiles(FileStates):
        def report_dates(self, steps: Sequence[int]) -> list[str]:
            if max(steps) > len(self.dates):
                raise SimulationError(f"{self.case.name} reports 2 steps")
            return super().report_dates(steps)

    monkeypatch.setattr(PressureOffset, "case_states", lambda self, case: ShortFiles(case))
    with pytest.raises(ConfigError, match="does not report the model's report steps"):
        composition.start(_case(tmp_path, surrogate=_section(built)), max_steps=1)


def test_untrained_report_dates_are_refused(tmp_path: Path) -> None:
    """A model reading a step whose date the surrogate was not validated at: refused. The
    deck reports a fifth step (2000-06-01); the surrogate knows the four dates of its keys."""
    deck = DECK.replace("TSTEP\n 29 31 30 /", "TSTEP\n 29 31 30 31 /")

    class Longer(PressureOffset):
        report_steps = (1, 2, 3, 4, 5)

    _case(tmp_path, name="base", deck=deck)
    model = _build(tmp_path)
    model_error_models.register("test_longer_offset")(Longer)
    try:
        config = _case(tmp_path, deck=deck, surrogate=_section(model), model="test_longer_offset")
        with pytest.raises(ConfigError, match="was not validated at"):
            composition.start(config, max_steps=1)
    finally:
        model_error_models.remove("test_longer_offset")


def test_the_build_and_run_refusals(tmp_path: Path) -> None:
    from apex.domain.config import PriorGroupSection

    config = composition.load_config(_case(tmp_path))
    with pytest.raises(ConfigError, match="stored forecasts are the deck with its model error"):
        surrogates._refuse(config)
    grouped = config.model_copy(update={"prior": config.prior.model_copy(update={
        "groups": {"g": PriorGroupSection(name="gaussian", variables=["nn_params", "x"])}
    })})  # fmt: skip
    with pytest.raises(ConfigError, match="prior group 'g'"):
        surrogates._deck_alone(grouped)
    only = config.model_copy(update={"state": config.state.model_copy(update={
        "variables": [v for v in config.state.variables if v.name == "nn_params"]
    })})  # fmt: skip
    with pytest.raises(ConfigError, match="state is only"):
        surrogates._deck_alone(only)
    unknown = config.model_copy(update={
        "model_error": config.model_error.model_copy(update={"name": "no_such_model"})  # type: ignore[union-attr]
    })  # fmt: skip
    with pytest.raises(ConfigError, match=r"\[model_error\]"):
        surrogates._deck_alone(unknown)
    no_deck = config.model_copy(update={
        "simulator": config.simulator.model_copy(update={"options": {}})
    })  # fmt: skip
    with pytest.raises(ConfigError, match="no deck_template_file"):
        composition.model_error_step_dates(no_deck)


def test_a_deck_only_run_with_an_unresolved_schedule_still_runs(tmp_path: Path) -> None:
    """The deck's text is read for dates only with a surrogate: model error alone runs."""
    deck = DECK.replace("TSTEP\n 29 31 30 /", "TSTEP\n ${steps} /")
    result = composition.start(_case(tmp_path, deck=deck, members=10), max_steps=1)
    assert result.history


def test_report_dates_against_the_keys_when_no_time_range() -> None:
    """A surrogate recording no gate time range: the model's dates must be its keys' times."""
    from types import SimpleNamespace

    def model(time_range: tuple[str, str] | None) -> object:
        domain = SimpleNamespace(time_range=time_range, keys=KEYS)
        return SimpleNamespace(domain=domain, path=Path("model.json"))

    dates = dict(enumerate([*DATES, "2000-06-01"], start=1))
    inside, beyond = PressureOffset(), PressureOffset()
    beyond.report_steps = (1, 2, 3, 4, 5)
    surrogates._refuse_untrained_dates(model(None), inside, dates)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="leave those steps out of the model's report_steps"):
        surrogates._refuse_untrained_dates(model(None), beyond, dates)  # type: ignore[arg-type]
    surrogates._refuse_untrained_dates(model((DATES[0], DATES[-1])), inside, dates)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="was not validated at"):
        surrogates._refuse_untrained_dates(model((DATES[0], DATES[-1])), beyond, dates)  # type: ignore[arg-type]


def test_after_a_switch_to_the_deck_the_iterates_save_residuals(
    tmp_path: Path, built: Path
) -> None:
    """The checks fail (the planted DELTA makes the two effective models differ beyond a
    tightened gate); the run switches to the deck. Iterates before the switch save no
    residuals, those after it every member's; the posterior's forecast reuses the deck's
    forecasts with their residuals; a run resumed across the switch is the same, bit for
    bit."""
    section = _section(built, check_members=4, check_every=1, on_check_failure="simulator",
                       outside="simulator", sigma_ratio=1e-6)  # fmt: skip
    composition.start(_case(tmp_path, surrogate=section))
    run = tmp_path / "runs" / "hm"
    state = json.loads((run / "state.json").read_text())
    switched = state["surrogate"]["switched_at"]
    assert switched is not None
    after = []
    for path in sorted((run / "checkpoints").glob("iter_*.npz")):
        saved = np.load(path)
        iteration = int(path.stem.split("_")[1])
        if "simulator" not in set(saved["forecast_source"]) or "surrogate" in set(
            saved["forecast_source"]
        ):
            assert "auxiliary_predicted" not in saved.files, path.name
            continue
        if iteration <= switched:
            continue
        after.append(path.name)
        assert "auxiliary_predicted" in saved.files, path.name
        for column in range(saved["state"].shape[1]):
            x, theta = saved["state"][:N_STATE, column], float(saved["state"][N_STATE, column])
            np.testing.assert_allclose(
                saved["auxiliary_predicted"][:, column], _residual(_pressure(x), theta),
                atol=ROUND_OFF,
            )  # fmt: skip
    assert after
    # ES-MDA's posterior is the last iterate, forecast by the deck after the switch: no
    # forecast again (full_<k>), its residuals are the iterate's (the GIES test reuses them).
    fulls = sorted((run / "checkpoints").glob("full_*.npz"))
    assert not fulls
    final = np.load(sorted((run / "checkpoints").glob("iter_*.npz"))[-1])
    assert set(final["forecast_source"]) == {"simulator"}
    assert "auxiliary_predicted" in final.files
    # Resumed after the switch: the same iterates, residuals included.
    composition.start(_case(tmp_path, name="parts", surrogate=section), max_steps=switched + 1)
    composition.resume(tmp_path / "runs" / "parts")
    for path in sorted((run / "checkpoints").glob("iter_*.npz")):
        a, b = np.load(path), np.load(tmp_path / "runs" / "parts" / "checkpoints" / path.name)
        assert set(a.files) == set(b.files), path.name
        for key in ("state", "predicted", "forecast_source", "auxiliary_predicted"):
            if key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_the_posterior_reuses_the_deck_forecasts_with_their_residuals(
    tmp_path: Path, built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GIES switched to the deck at its first trial, every trial rejected (forced): its
    accepted prior iterate is forecast again by the deck with the residuals, and the
    posterior's forecast reuses those members, the residuals included."""
    from apex.algorithms.data_assimilation.gies import DampingSchedule
    from apex.algorithms.data_assimilation.protocols import DampingDecision

    def rejected(self: DampingSchedule, damping: float, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        return DampingDecision(accepted=False, converged=False, damping=damping * 2.0)

    monkeypatch.setattr(DampingSchedule, "decide", rejected)
    section = _section(built, check_members=4, check_every=1, on_check_failure="simulator",
                       outside="simulator", sigma_ratio=1e-6)  # fmt: skip
    composition.start(_case(tmp_path, surrogate=section, method=GIES))
    checkpoints = tmp_path / "runs" / "hm" / "checkpoints"
    rescored = sorted(checkpoints.glob("rescored_*.npz"))
    assert rescored
    assert "auxiliary_predicted" in np.load(rescored[-1]).files
    full = [np.load(p) for p in sorted(checkpoints.glob("full_*.npz"))]
    assert full
    assert any(len(f["reused_ids"]) for f in full)
    for f in full:
        r = [
            _residual(_pressure(f["state"][:N_STATE, c]), float(f["state"][N_STATE, c]))
            for c in range(f["state"].shape[1])
        ]
        np.testing.assert_allclose(f["auxiliary_predicted"], np.column_stack(r), atol=ROUND_OFF)
