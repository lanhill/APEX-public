"""Linear cases and a test-only affine surrogate for history matching with a surrogate.

Shared by ``test_surrogate_history_matching.py`` (loaded from its file: ``tests/`` is not a
package). The surrogate is an affine least-squares map from the stacked inputs to the
outputs, registered by the tests and never shipped (validation 1); on a linear
case (the ``linear`` simulator, ``y = G x``) it reproduces the simulator to round-off.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from apex.core.surrogates import SurrogateInputs, SurrogateParts
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet

N_STATE = 3
TIMES = ("1", "2", "3", "4")
DATATYPES = ("WOPR:P1", "WBHP:P1", "WOPR:P2", "WBHP:P2")
TRUTH = np.array([0.5, -1.0, 0.8])
VARIANCE = 0.04
PRIOR_STD = 1.0
ESMDA = (
    'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9, reduction = 0.5 }\nmax_iterations = 6"
)


class Affine:
    """An affine least-squares map from the stacked inputs to the outputs (test only).

    ``predict_std`` returns ``std_of(std, k, n_keys)`` at its ``k``-th key, the same for
    every member (an uncertain surrogate's capability), so that key order matters.
    """

    version = "1"

    def __init__(self, std: float = 0.0) -> None:
        self.std = std
        self.keys: tuple[DataKey, ...] = ()
        self.names: tuple[str, ...] = ()
        self.weights = np.zeros((0, 0))

    def _stack(self, inputs: SurrogateInputs) -> NDArray[np.float64]:
        return np.concatenate([np.asarray(inputs[n], dtype=np.float64) for n in self.names])

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        self.names = snapshots.layout.names
        self.keys = snapshots.keys
        design = np.vstack([self._stack(snapshots.inputs), np.ones((1, snapshots.n_runs))])
        self.weights = np.ascontiguousarray(
            np.linalg.lstsq(design.T, snapshots.outputs.T, rcond=None)[0].T
        )

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        stacked = self._stack(inputs)
        return stacked, np.zeros(stacked.shape[1])

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        rows = [self.keys.index(k) for k in keys]
        full = self.weights[:, :-1] @ self._stack(inputs) + self.weights[:, -1:]
        return np.ascontiguousarray(full[rows])

    def predict_std(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        n = self._stack(inputs).shape[1]
        own = [std_of(self.std, self.keys.index(k), len(self.keys)) for k in keys]
        return np.tile(np.asarray(own)[:, None], (1, n))

    def to_arrays(self) -> SurrogateParts:
        own = {"names": list(self.names), "std": self.std,
               "keys": [[k.time, k.datatype, k.index] for k in self.keys]}  # fmt: skip
        return own, {"weights": self.weights}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = cls(std=float(manifest["std"]))  # type: ignore[arg-type]
        model.names = tuple(manifest["names"])  # type: ignore[arg-type]
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        model.weights = np.asarray(arrays["weights"], dtype=np.float64)
        return model


class CoarseAffine(Affine):
    """A coarse-corrected test surrogate: each key affine in its coarse value.

    On :func:`write_coarse`'s coarse stack, whose response is the fine one divided key by
    key, it corrects the coarse forecast to the fine one to round-off.
    """

    slope: NDArray[np.float64]

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        assert snapshots.coarse_outputs is not None
        coarse, fine = snapshots.coarse_outputs, snapshots.outputs
        self.slope = np.ascontiguousarray(
            [np.polyfit(coarse[k], fine[k], 1) for k in range(len(fine))]
        )

    def coarse_coordinates(
        self, coarse: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        forecast = np.asarray(coarse, dtype=np.float64)
        return forecast, np.zeros(forecast.shape[1])

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise ValueError("a coarse-corrected surrogate needs the coarse forecast")

    def predict_corrected(
        self, inputs: SurrogateInputs, coarse: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        rows = [self.keys.index(k) for k in keys]
        full = self.slope[:, :1] * np.asarray(coarse) + self.slope[:, 1:]
        return np.ascontiguousarray(full[rows])

    def to_arrays(self) -> SurrogateParts:
        own, arrays = super().to_arrays()
        return own, {**arrays, "slope": self.slope}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = super().from_arrays(manifest, arrays)
        model.slope = np.asarray(arrays["slope"], dtype=np.float64)
        return model


def write_coarse(folder: Path) -> str:
    """Write a coarse stack of the case (``y_c = G x / d``, key by key); return ``[coarse]``."""
    forward = forward_matrix()
    d = np.linspace(0.5, 2.0, forward.shape[0])
    np.savez(folder / "coarse.npz", matrix=forward / d[:, None])
    return '[coarse]\noptions = { forward_file = "coarse.npz" }'


def std_of(std: float, k: int, n_keys: int) -> float:
    """Return the test surrogate's predictive standard deviation at its ``k``-th key."""
    return std * (1.0 + k / n_keys)


def keys_in_order() -> list[tuple[str, str]]:
    """Return the case's keys ``(time, datatype)`` in the observations' (and models') order."""
    return [(t, d) for d in DATATYPES for t in TIMES]


def forward_matrix(seed: int = 3) -> NDArray[np.float64]:
    """Return the case's ``G`` ``(n_keys, N_STATE)``."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(DATATYPES) * len(TIMES), N_STATE))


def write_case(
    folder: Path,
    *,
    method: str = ESMDA,
    surrogate: str = "",
    validation: str = "",
    members: int = 100,
    prior_std: float = PRIOR_STD,
    name: str = "hm",
    execution: str = 'name = "serial"',
    extra: str = "",
    variance: float = VARIANCE,
    variances: Sequence[float] | None = None,
) -> Path:
    """Write a linear case (``y = G x``) and return its configuration ``hm.toml``.

    Its keys are ``(time, datatype)`` with step-index times (a surrogate's domain bounds
    them); the observations are ``G x_true`` plus noise of variance ``VARIANCE``, recorded
    with the error variance ``variance`` (or ``variances``, one per key).
    """
    folder.mkdir(parents=True, exist_ok=True)
    forward = forward_matrix()
    np.savez(folder / "forward.npz", matrix=forward)
    np.savez(folder / "truth.npz", x=TRUTH)
    keys = keys_in_order()
    recorded = [variance] * len(keys) if variances is None else list(variances)
    noise = np.random.default_rng(11).standard_normal(len(keys)) * np.sqrt(VARIANCE)
    observed = forward @ TRUTH + noise
    rows = ["time,datatype,value,variance"] + [
        f"{t},{d},{float(v)!r},{float(s)!r}"
        for (t, d), v, s in zip(keys, observed, recorded, strict=True)
    ]
    (folder / "observations.csv").write_text("\n".join(rows) + "\n")
    outputs = ", ".join(f'["{t}", "{d}"]' for t, d in keys)
    config = folder / f"{name}.toml"
    config.write_text(f"""
[run]
name = "{name}"
seed = 2026

[[state.variables]]
name = "x"
size = {N_STATE}

[prior]
ensemble_size = {members}
generator = {{ name = "gaussian", options = {{ mean = 0.0, std = {prior_std} }} }}

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

[execution]
{execution}

[method]
{method}

{validation}

{surrogate}

{extra}
""")
    return config


def write_build(
    folder: Path,
    plugin: str,
    *,
    case: str = "hm.toml",
    name: str = "lin",
    options: str = "{}",
    box: tuple[float, float] = (-6.0, 6.0),
    extra: str = "",
) -> Path:
    """Write a surrogate build of the case: a Latin hypercube over ``box^3``, 30 + 10 runs."""
    path = folder / f"build_{name}.toml"
    path.write_text(f"""
[surrogate]
name = "{name}"
model = "{plugin}"
options = {options}
seed = 7

[case]
config_file = "{case}"

[design]
plan = "lhs"
training = 30
held_out = 10
box = {{ x = [{box[0]}, {box[1]}] }}

{extra}
""")
    return path
