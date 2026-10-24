"""Test fixtures every APEX package's tests share.

A package's ``conftest.py`` opts in with ``pytest_plugins = ["apex.testing"]`` (never a
``pytest11`` entry point: the autouse fixtures would then reach every test session where
``apex`` is installed):

- isolation: the session's tests use resource pools and a services root of their own, and
  live code (``APEX_RESOURCES``, ``APEX_HOME``, ``APEX_CLI_CODE=live``);
- generated cases with exact answers: ``linear_case`` (linear-Gaussian) and ``well_case``;
- ``private_executor``, a private plugin for tests of what only the owner sees;
- ``small_deck``, a deck with a history, and ``write_pet_case``, a PET case, for the
  case-converter.

It needs pytest (a development dependency), so only tests import it.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.simulation import MemberCallback, MemberFailure, MemberInput, Simulator
from apex.domain.observations import DataKey


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
