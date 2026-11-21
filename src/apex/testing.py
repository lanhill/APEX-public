"""Test fixtures every APEX package's tests share.

A package's ``conftest.py`` opts in with ``pytest_plugins = ["apex.testing"]`` (never a
``pytest11`` entry point: the autouse fixtures would then reach every test session where
``apex`` is installed):

- isolation: the session's tests use resource pools and a services root of their own, and
  live code (``APEX_RESOURCES``, ``APEX_HOME``, ``APEX_CLI_CODE=live``);
- generated cases with exact answers: ``linear_case`` (linear-Gaussian) and ``well_case``;
- ``private_executor``, a private plugin for tests of what only the owner sees;
- ``small_deck``, a deck with a history, and ``write_pet_case``, a PET case, for the
  case-converter;
- :func:`assert_block_states_conform`: a block-state source (a surrogate's) against a
  deck's run, as a model-error model reads them;
- :class:`MeanAffineFields`, a field surrogate without torch, and
  :func:`build_field_surrogate`, a build with it, for tests of the field snapshots end to
  end on a simulator (check 3).

It needs pytest (a development dependency), so only tests import it.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.simulation import (
    BlockStates,
    DatedStates,
    MemberCallback,
    MemberFailure,
    MemberInput,
    Simulator,
)
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet
from apex.simulation.model_error import WellSchedule


@pytest.fixture(autouse=True, scope="session")
def _isolated_resource_pools(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Keep tests off the machine's resource pools, with room for parallel runs.

    Executors read ``$APEX_RESOURCES`` when they are built; worker processes inherit it.
    """
    import os

    root = tmp_path_factory.mktemp("resources")
    (root / "pools.toml").write_text("[pools]\ncpu = 64\neclipse = 3\n")
    previous = os.environ.get("APEX_RESOURCES")
    os.environ["APEX_RESOURCES"] = str(root)
    yield root
    if previous is None:
        del os.environ["APEX_RESOURCES"]
    else:
        os.environ["APEX_RESOURCES"] = previous


@pytest.fixture(autouse=True, scope="session")
def _isolated_services_root(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Keep tests off the user's services root (``$APEX_HOME``).

    Commands that build a service without ``--root`` (``apex plugins``, ``apex schema``)
    would otherwise open ``~/.apex/service``, where the user's own jobs may be running.
    """
    import os

    root = tmp_path_factory.mktemp("apex_home")
    # Command-line submissions freeze the checkout by default: tests run live
    # code, unless one sets up a repository of its own (test_submit_command.py).
    saved = {name: os.environ.get(name) for name in ("APEX_HOME", "APEX_CLI_CODE")}
    os.environ["APEX_HOME"] = str(root)
    os.environ["APEX_CLI_CODE"] = "live"
    yield root
    for name, previous in saved.items():
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous


# --- generated cases ------------------------------------------------------------------

N_STATE, N_DATA = 3, 8
PRIOR_STD = 1.0


@dataclass(frozen=True)
class LinearCase:
    """Files and exact answers of a generated case."""

    directory: Path
    config: Path
    forward: np.ndarray
    observations: np.ndarray
    variances: np.ndarray

    def exact_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance for the N(0, PRIOR_STD^2 I) prior."""
        g, c_d = self.forward, np.diag(self.variances)
        c_m = PRIOR_STD**2 * np.eye(N_STATE)
        gain = c_m @ g.T @ np.linalg.inv(g @ c_m @ g.T + c_d)
        return gain @ self.observations, c_m - gain @ g @ c_m

    def least_squares(self, perturbed: np.ndarray) -> np.ndarray:
        """Weighted least-squares fit of each column of ``perturbed``."""
        w = 1.0 / self.variances
        normal = self.forward.T @ (w[:, None] * self.forward)
        return np.linalg.solve(normal, self.forward.T @ (w[:, None] * perturbed))


CaseFactory = Callable[..., LinearCase]


@pytest.fixture
def linear_case(tmp_path: Path) -> CaseFactory:
    """Return a factory writing a case; keyword arguments fill the config template."""

    def make(
        *,
        method: str = 'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "approx", options = { energy = 1.0 } }',
        ensemble_size: int = 200,
        execution: str = 'name = "serial"',
        simulator_extra: str = "",
        failures: str = "",
        name: str = "case",
    ) -> LinearCase:
        rng = np.random.default_rng(99)
        forward = rng.standard_normal((N_DATA, N_STATE))
        variances = rng.uniform(0.05, 0.2, N_DATA)
        truth = np.array([0.5, -1.0, 0.8])
        observations = forward @ truth + np.sqrt(variances) * rng.standard_normal(N_DATA)
        directory = tmp_path / name
        directory.mkdir()
        np.savez(directory / "forward.npz", matrix=forward)
        np.savez(directory / "truth.npz", x=truth)
        rows = ["time,datatype,value,variance"] + [
            f"t{i},y,{float(observations[i])!r},{float(variances[i])!r}" for i in range(N_DATA)
        ]
        (directory / "observations.csv").write_text("\n".join(rows) + "\n")
        outputs = ", ".join(f'["t{i}", "y"]' for i in range(N_DATA))
        config = directory / "config.toml"
        config.write_text(
            f"""
[run]
name = "{name}"
seed = 2026

[[state.variables]]
name = "x"
size = {N_STATE}

[prior]
ensemble_size = {ensemble_size}
generator = {{ name = "gaussian", options = {{ mean = 0.0, std = {PRIOR_STD} }} }}

[observations]
file = "observations.csv"

[case]
truth_file = "truth.npz"

[simulator]
name = "linear"

[simulator.options]
forward_file = "forward.npz"
variables = ["x"]
outputs = [{outputs}]
{simulator_extra}

[execution]
{execution}

[method]
{method}

{failures}
"""
        )
        return LinearCase(directory, config, forward, observations, variances)

    return make


@dataclass(frozen=True)
class WellCase:
    """A generated linear case with production-style data (``WOPR:P1`` ... ``WOPR:P8``)."""

    directory: Path
    config: Path


WellCaseFactory = Callable[..., WellCase]

#: GIES without localization on 10 members for 30 unknowns overfits this case: the mean
#: CV normalized data mismatch (wells P7 and P8) of seed 1 falls to a minimum at iteration 2 and
#: rises after it, and so does the RMSE against the truth (measured with PET-fork's
#: halving after an accepted step, ``reduction = 0.5``, the default until 2026-10-13).
GIES_OVERFITTING = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-6, reduction = 0.5 }\nmax_iterations = 15"
)


@pytest.fixture
def well_case(tmp_path: Path) -> WellCaseFactory:
    """Return a factory writing a linear case with wells P1-P8, four report times each.

    ``validation`` is the text of a ``[validation]`` section (none by default).
    """

    def make(
        *,
        method: str = GIES_OVERFITTING,
        validation: str = "",
        name: str = "case",
        seed: int = 1,
        n_state: int = 30,
        ensemble_size: int = 10,
        simulator_extra: str = "",
    ) -> WellCase:
        rng = np.random.default_rng(seed)
        keys = [(f"t{t}", f"WOPR:P{w}") for w in range(1, 9) for t in range(4)]
        forward = rng.standard_normal((len(keys), n_state)) / np.sqrt(n_state)
        truth = rng.standard_normal(n_state)
        variances = np.full(len(keys), 0.09)
        observations = forward @ truth + np.sqrt(variances) * rng.standard_normal(len(keys))
        directory = tmp_path / name
        directory.mkdir()
        np.savez(directory / "forward.npz", matrix=forward)
        np.savez(directory / "truth.npz", x=truth)
        rows = ["time,datatype,value,variance"] + [
            f"{t},{dt},{float(observations[i])!r},{float(variances[i])!r}"
            for i, (t, dt) in enumerate(keys)
        ]
        (directory / "observations.csv").write_text("\n".join(rows) + "\n")
        outputs = ", ".join(f'["{t}", "{dt}"]' for t, dt in keys)
        config = directory / "config.toml"
        config.write_text(
            f"""
[run]
name = "{name}"
seed = 7

[[state.variables]]
name = "x"
size = {n_state}

[prior]
ensemble_size = {ensemble_size}
generator = {{ name = "gaussian", options = {{ mean = 0.0, std = 1.0 }} }}

[observations]
file = "observations.csv"

[case]
truth_file = "truth.npz"

[simulator]
name = "linear"

[simulator.options]
forward_file = "forward.npz"
variables = ["x"]
outputs = [{outputs}]
{simulator_extra}

[method]
{method}

{validation}
"""
        )
        return WellCase(directory, config)

    return make


# --- a private plugin -------------------------------------------------------------------


class _Relay:
    """A private executor for tests (``relay``): visible to the owner, never to guests.

    Core's tests use it where they need a private plugin, so they pass whether a private
    package is installed or not. It accepts any options and runs nothing.
    """

    visibility = "private"

    def __init__(self, **options: object) -> None:
        self.options = options

    def run_all(
        self,
        simulator: Simulator,
        members: Sequence[MemberInput],
        keys: Sequence[DataKey],
        *,
        on_result: MemberCallback | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run nothing (the tests only check where the plugin is seen)."""
        raise NotImplementedError("the test executor runs nothing")


@pytest.fixture
def private_executor() -> Iterator[str]:
    """Register a private executor, ``relay``, for one test; return its name."""
    from apex.core.plugins import executors

    executors.names()  # its entry points loaded first: the test's executor is added after
    executors.add("relay", _Relay)
    try:
        yield "relay"
    finally:
        executors.remove("relay")


SMALL_DECK_PERMX = [10.0 * (1 + i) for i in range(24)]
SMALL_DECK_PORO = [0.1 + 0.005 * i for i in range(24)]


def write_small_deck(folder: Path) -> Path:
    """Write a 4 x 3 x 2 deck with a history (the case-converter's tests).

    PERMX inline, PORO in ``props/poro.inc``, PERMY/PERMZ copied from PERMX, NTG by EQUALS,
    a producer and an injector under history control over DATES and TSTEP. Not runnable:
    it has no saturation tables.
    """
    (folder / "props").mkdir(parents=True)
    (folder / "props" / "poro.inc").write_text(
        "-- porosity\nPORO\n" + "\n".join(f"{v:.4f}" for v in SMALL_DECK_PORO) + "\n/\n"
    )
    (folder / "pvt.inc").write_text("PVTW\n 1 1 4E-5 0.5 0 /\n")
    permx = " ".join(repr(v) for v in SMALL_DECK_PERMX)
    deck = folder / "CASE.DATA"
    deck.write_text(f"""RUNSPEC
TITLE
 test case
DIMENS
 4 3 2 /
OIL
WATER
METRIC
START
 1 JAN 2020 /
GRID
DX
 24*50 /
DY
 24*50 /
DZ
 24*5 /
TOPS
 12*2000 /
PERMX
 {permx} /
COPY
 'PERMX' 'PERMY' /
 'PERMX' 'PERMZ' /
/
INCLUDE
 'props/poro.inc' /
EQUALS
 'NTG' 0.8 /
/
PROPS
INCLUDE
 'pvt.inc' /
SCHEDULE
WELSPECS
 'PROD' 'G' 4 3 1* 'OIL' /
 'INJ' 'G' 1 1 1* 'WATER' /
/
COMPDAT
 'PROD' 2* 1 2 'OPEN' /
 'INJ' 2* 1 1 'OPEN' /
/
WCONHIST
 'PROD' 'OPEN' 'ORAT' 100 5 0 1* 1* 1* 250 /
/
DATES
 1 FEB 2020 /
/
WCONHIST
 'PROD' 'OPEN' 'ORAT' 90 15 0 /
/
WCONINJH
 'INJ' 'WATER' 'OPEN' 200 300 /
/
DATES
 1 MAR 2020 /
/
TSTEP
 30 /
END
""")
    return deck


@pytest.fixture
def small_deck(tmp_path: Path) -> Path:
    """Return a small deck with a history (:func:`write_small_deck`) in its own folder."""
    return write_small_deck(tmp_path / "deck")


#: PET cases for the case-converter's tests.
PET_COMMON = """
[simulator]
runfile = "RUNFILE"
parallel = 2
sim_limit = 60.0

[dataassim]
data = "data.csv"
datavar = "var.csv"
{dataassim}
"""
PET_STORED = """
[ensemble]
ne = 4
state = ["permx"]
importstate = "prior.npz"
seed = 7
[ensemble.prior_permx]
limits = [0.0, 8.0]
"""
PET_CASES = {
    "esmda": 'scheme = "esmda"\nenergy = 98.0\n[dataassim.mda]\ntot_assim_steps = 3',
    "es": 'scheme = "es"\nenergy = 95.0',
    "lmenrml": 'scheme = "lmenrml"\nanalysis = "subspace"\niteration = [["lambda", "auto"], '
    '["lambda_max", 1e6], ["max_inner_iter", 4], ["energy", 90], ["max_iter", 12]]',
    "gnenrml": 'scheme = "gnenrml"\nanalysis = "margis"\niteration = [["gamma", 0.5], '
    '["gamma_max", 0.8], ["gamma_factor", 3], ["max_iter", 5]]',
}


def write_pet_case(folder: Path, ensemble: str, dataassim: str) -> Path:
    """Write a small PET case (the case-converter's PET tests).

    ``ensemble`` is its ``[ensemble]`` section (e.g. :data:`PET_STORED`), ``dataassim`` the
    lines of its ``[dataassim]`` (e.g. one of :data:`PET_CASES`). Returns its CONFIG.toml.
    """
    (folder / "grid").mkdir(parents=True)
    (folder / "grid" / "Grid.grdecl").write_text("-- grid\n")
    (folder / "RUNFILE.mako").write_text("INCLUDE\n'../grid/Grid.grdecl' /\n${permx[0]}\n")
    (folder / "data.csv").write_text(
        "dates,WOPR:P1,WWCT:P1\n2024-01-01,100.0,0.1\n2025-01-01,90.0,\n"
    )
    (folder / "var.csv").write_text("dates,WOPR:P1,WWCT:P1\n2024-01-01,\"['abs', 25]\","
                                    "\"['abs', 0.01]\"\n2025-01-01,\"['abs', 4]\",\n")  # fmt: skip
    np.savez(folder / "prior.npz", permx=np.zeros((6, 4)))
    np.savez(folder / "priormean.npz", PriorMean=np.arange(48.0))
    path = folder / "CONFIG.toml"
    path.write_text(ensemble + PET_COMMON.format(dataassim=dataassim))
    return path


def assert_block_states_conform(
    files: BlockStates,
    dated: DatedStates,
    step_dates: Mapping[int, str],
    wells: Sequence[WellSchedule],
    phases: Sequence[str] = ("oil", "water"),
    *,
    rtol: float = 1e-6,
    atol: float = 0.0,
) -> None:
    """Assert that ``dated`` gives a model-error model the inputs ``files`` gives.

    Every :class:`~apex.core.surrogates.BlockStateSurrogate` plugin's
    tests call it with one member's states and the same member's deck run (its
    ``CaseFiles``, or arrays read from one). Checks the report dates by step, then every
    well's input sequence (``well_features``: the static arrays, the dynamic ones, the
    pressure difference to the bottom-hole pressure, at every perforated block and step)
    read through :class:`~apex.core.simulation.StepStates`, within ``rtol``/``atol``.

    Raises
    ------
    AssertionError
        Naming the dates or the well and the largest difference.
    """
    from apex.core.simulation import StepStates
    from apex.simulation.model_error import well_features

    steps = sorted(int(s) for s in step_dates)
    found_dates = files.report_dates(steps)
    expected_dates = [step_dates[s] for s in steps]
    assert found_dates == expected_dates, (
        f"report dates differ: the run's {found_dates}, the states' {expected_dates}"
    )
    expected = well_features(files, wells, steps, phases)
    found = well_features(StepStates(dated, dict(step_dates)), wells, steps, phases)
    for well, want, got in zip(wells, expected, found, strict=True):
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol, err_msg=f"well {well.name}'s model inputs"
        )


class MeanAffineFields:
    """A field surrogate for tests: every output affine in each input variable's mean.

    Not a method: a stand-in that trains in NumPy alone (least squares, no ``torch``), so
    that a build with ``[case] fields`` runs end to end where the field surrogates' training
    extra is not installed (check 3: the field snapshots on a simulator's
    case-state reader). Its coordinates are the variables' means ``(n_variables,
    n_members)`` and its fit is exact least squares on ``[means; 1]``, so a dozen runs
    determine it; the fit is deterministic and does not use its generator. It reads its
    fields as flat arrays and works on any cell layout; it does not declare
    ``unstructured_fields``, so a build asks for a structured grid on its behalf (the case's
    reader must give one), as for the grid-based field surrogates.
    Register it under a test name (``surrogate_models.register(name)(MeanAffineFields)``).
    """

    version = "1"

    def __init__(self) -> None:
        self.keys: tuple[DataKey, ...] = ()
        self.names: tuple[str, ...] = ()
        self.field_times: tuple[str, ...] = ()
        self.weights = np.zeros((0, 0))
        self.field_weights: dict[str, NDArray[np.float64]] = {}

    def _means(self, inputs: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """``(n_variables, n_members)``: each variable's mean over its entries."""
        return np.vstack([np.asarray(inputs[n], np.float64).mean(axis=0) for n in self.names])

    def _design(self, inputs: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        means = self._means(inputs)
        return np.vstack([means, np.ones((1, means.shape[1]))])

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        """Fit every output and field value by least squares on the means; ``rng`` is unused."""
        self.names = tuple(snapshots.layout.names)
        self.keys = snapshots.keys
        self.field_times = snapshots.field_times
        design = self._design(snapshots.inputs).T  # (n_runs, n_variables + 1)
        solved = np.linalg.lstsq(design, snapshots.outputs.T, rcond=None)[0]
        self.weights = np.ascontiguousarray(solved.T)
        self.field_weights = {}
        for name, values in snapshots.fields.items():
            flat = values.reshape(-1, snapshots.n_runs)  # (cells * times, n_runs)
            solved = np.linalg.lstsq(design, flat.T, rcond=None)[0]
            self.field_weights[name] = np.ascontiguousarray(solved.T)

    def coordinates(
        self, inputs: Mapping[str, NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the means and a zero residual (the model sees nothing else)."""
        means = self._means(inputs)
        return means, np.zeros(means.shape[1])

    def predict(
        self, inputs: Mapping[str, NDArray[np.float64]], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        """Return ``(len(keys), n_members)``."""
        rows = [self.keys.index(k) for k in keys]
        return np.ascontiguousarray((self.weights @ self._design(inputs))[rows])

    def predict_fields(
        self, inputs: Mapping[str, NDArray[np.float64]], names: Sequence[str], times: Sequence[str]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{name: (n_cells, len(times), n_members)}``."""
        design = self._design(inputs)
        found = {}
        for name in names:
            cube = (self.field_weights[name] @ design).reshape(
                -1, len(self.field_times), design.shape[1]
            )
            found[name] = np.ascontiguousarray(cube[:, [self.field_times.index(t) for t in times]])
        return found

    def to_arrays(
        self,
    ) -> tuple[Mapping[str, object], Mapping[str, NDArray[np.generic]]]:
        """Return its manifest and arrays."""
        own = {
            "names": list(self.names),
            "keys": [[k.time, k.datatype, k.index] for k in self.keys],
            "field_times": list(self.field_times),
            "fields": sorted(self.field_weights),
        }
        arrays = {"weights": self.weights}
        arrays.update({f"field_{n}": w for n, w in self.field_weights.items()})
        return own, arrays

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        """Return the model :meth:`to_arrays` described."""
        model = cls()
        model.names = tuple(str(n) for n in manifest["names"])  # type: ignore[attr-defined]
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        model.field_times = tuple(str(t) for t in manifest["field_times"])  # type: ignore[attr-defined]
        model.weights = np.asarray(arrays["weights"], np.float64)
        model.field_weights = {
            str(n): np.asarray(arrays[f"field_{n}"], np.float64)
            for n in manifest["fields"]  # type: ignore[attr-defined]
        }
        return model


#: The name :func:`build_field_surrogate` registers :class:`MeanAffineFields` under.
MEAN_AFFINE_FIELDS = "test_mean_affine_fields"


@dataclass(frozen=True)
class FieldBuild:
    """What :func:`build_field_surrogate` gives: the build's folder, completion and gate."""

    path: Path
    complete: bool
    gate: Mapping[str, object]
    seconds: float


def build_field_surrogate(
    folder: Path,
    case_config: Path,
    fields: Sequence[str],
    field_times: Sequence[str],
    *,
    training: int = 12,
    held_out: int = 3,
    workers: int = 3,
    seed: int = 7,
) -> FieldBuild:
    """Build :class:`MeanAffineFields` on a case's runs with ``[case] fields`` and gate it.

    Writes ``folder/build.toml`` (design ``prior``, ``training`` + ``held_out`` runs, the
    process executor with ``workers``) and runs it in this process
    (``services.surrogates.build``, as ``apex surrogate build``). Registers the plugin under
    :data:`MEAN_AFFINE_FIELDS` for the build only. ``gate`` is the build's ``gate.json``.
    """
    import json
    import time

    from apex.core.plugins import surrogate_models
    from apex.services import surrogates

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "build.toml"
    quoted = ", ".join(f'"{name}"' for name in fields)
    times = ", ".join(f'"{t}"' for t in field_times)
    path.write_text(f"""[surrogate]
name = "fields"
model = "{MEAN_AFFINE_FIELDS}"
seed = {seed}

[case]
config_file = "{case_config.resolve()}"
fields = [{quoted}]
field_times = [{times}]

[design]
plan = "prior"
training = {training}
held_out = {held_out}

[execution]
name = "process"
options = {{ workers = {workers} }}
""")
    surrogate_models.register(MEAN_AFFINE_FIELDS)(MeanAffineFields)
    try:
        began = time.perf_counter()
        result = surrogates.build(path)
        seconds = time.perf_counter() - began
    finally:
        surrogate_models.remove(MEAN_AFFINE_FIELDS)
    gate = json.loads((result.path / "gate.json").read_text())
    return FieldBuild(result.path, result.complete, gate, seconds)
