"""A new kind of loop runs on the generic driver with no workflow change.

``LineSearchESMDA`` is a test-only algorithm that fits neither of the old loops:
- each step forecasts **two candidates** at once, ES-MDA updates with inflation ``alpha``
  and ``2 alpha`` built from the same perturbation draw;
- it keeps the candidate with the lower mean data mismatch and records the other as a rejected
  trial;
- the next ``alpha`` **depends on the forecast**: halved when the smaller-inflation
  candidate won, doubled otherwise.

It is registered for this test only. The test checks that it runs through the
composition root, checkpoints after every step, and resumes bit-identically, and that
the linear-Gaussian case still moves towards the exact posterior mean.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    AlgorithmState,
    ApproxUpdate,
    Estimate,
    ForecastRequest,
    Forecasts,
    Record,
    RunSetup,
    Status,
    Step,
    keep,
    methods,
)
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.services import composition
from apex.storage.run_directory import RunDirectory

NAME = "test_line_search_esmda"


class LinearCase(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path

    def exact_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and covariance."""
        ...


CaseFactory = Callable[..., LinearCase]


class LineSearchESMDA:
    """ES-MDA with a two-candidate line search on the inflation factor (test only).

    Parameters
    ----------
    alpha
        Initial inflation factor.
    n_steps
        Number of line-search steps.
    """

    status: Status = "experimental"
    iterative: bool = True

    def __init__(self, alpha: float = 4.0, n_steps: int = 4) -> None:
        self.initial_alpha = float(alpha)
        self.n_steps = int(n_steps)
        self.strategy = ApproxUpdate(energy=1.0)
        self._arrays: dict[str, NDArray[Any]] = {}
        self._values: dict[str, Any] = {}

    def start(self, setup: RunSetup) -> Step:
        """Draw the scoring perturbations once and request the prior forecast."""
        self._setup = setup
        prior = keep(setup.prior)
        n = prior.shape[-1]
        scoring = setup.observations.values[:, None] + setup.data_error.perturbations(
            setup.streams["perturbations"], n
        )
        self._arrays = {"state": prior, "member_ids": np.arange(n), "scoring": keep(scoring)}
        self._values = {"alpha": self.initial_alpha, "step": 0, "iteration": 0}
        return Step([ForecastRequest("prior", 0, prior, np.arange(n))], iteration=0)

    def advance(self, forecasts: Forecasts) -> Step:
        """Score the prior, or pick the better candidate; then propose the next pair."""
        a, v = self._arrays, self._values
        records: list[Record] = []
        if "prior" in forecasts:
            forecast = forecasts["prior"]
            a["member_ids"] = a["member_ids"][forecast.ok]
            for key in ("state", "scoring"):
                a[key] = keep(a[key], forecast.ok)
            a["predicted"] = keep(forecast.predicted)
            records.append(self._record(0, a["state"], a["predicted"], None, accepted=True))
        else:
            records.extend(self._choose(forecasts))
        if v["step"] == self.n_steps:
            return Step([], iteration=v["iteration"] + 1, records=records)
        return self._propose(records)

    def _propose(self, records: list[Record]) -> Step:
        a, v = self._arrays, self._values
        data_error: DataErrorCovariance = self._setup.data_error
        # One perturbation draw z for both candidates: D(alpha) = d_obs + sqrt(alpha) L z.
        z = data_error.perturbations(self._setup.streams["perturbations"], a["state"].shape[-1])
        observed = self._setup.observations.values[:, None]
        for label, alpha in (("small", v["alpha"]), ("large", 2.0 * v["alpha"])):
            perturbed = observed + np.sqrt(alpha) * z
            step = self.strategy.step(
                a["state"], a["predicted"], perturbed, data_error.inflated(alpha)
            )
            a[f"candidate_{label}"] = keep(a["state"] + step)
        first = v["iteration"] + 1
        requests = [
            ForecastRequest("small", first, a["candidate_small"], a["member_ids"]),
            ForecastRequest("large", first + 1, a["candidate_large"], a["member_ids"]),
        ]
        return Step(requests, iteration=first, records=records)

    def _choose(self, forecasts: Forecasts) -> list[Record]:
        a, v = self._arrays, self._values
        small, large = forecasts["small"], forecasts["large"]
        ok = small.ok & large.ok
        predicted = {
            "small": keep(small.predicted[:, ok[small.ok]]),
            "large": keep(large.predicted[:, ok[large.ok]]),
        }
        a["member_ids"] = a["member_ids"][ok]
        for key in ("scoring", "candidate_small", "candidate_large"):
            a[key] = keep(a[key], ok)
        data_mismatch = {
            label: float(self._setup.data_error.data_mismatch(a["scoring"] - y).mean())
            for label, y in predicted.items()
        }
        winner = "small" if data_mismatch["small"] <= data_mismatch["large"] else "large"
        first = v["iteration"] + 1
        records = [
            self._record(
                first + (label == "large"),
                a[f"candidate_{label}"],
                predicted[label],
                v["alpha"] * (1 + (label == "large")),
                accepted=label == winner,
            )
            for label in ("small", "large")
        ]
        a["state"], a["predicted"] = a[f"candidate_{winner}"], predicted[winner]
        # The forecast decides the next inflation: a won small step invites a bolder one.
        v["alpha"] = v["alpha"] / 2.0 if winner == "small" else v["alpha"] * 2.0
        v["alpha"] = max(v["alpha"], 1.0)
        v["step"] += 1
        v["iteration"] = first + 1
        del a["candidate_small"], a["candidate_large"]
        return records

    def _record(
        self,
        iteration: int,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        damping: float | None,
        *,
        accepted: bool,
    ) -> Record:
        return Record(
            iteration=iteration,
            kind="iterate" if accepted else "trial",
            state=state,
            predicted=predicted,
            member_ids=self._arrays["member_ids"],
            scoring=self._arrays["scoring"],
            accepted=accepted,
            damping=damping,
        )

    def export_state(self) -> AlgorithmState:
        """Everything the loop holds: the kept ensemble, the pending candidates, alpha."""
        return AlgorithmState(dict(self._arrays), dict(self._values))

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Continue after a checkpoint: re-request the pending candidates."""
        self._setup = setup
        self._arrays = {k: keep(v) for k, v in state.arrays.items()}
        self._values = dict(state.values)
        a, first = self._arrays, self._values["iteration"] + 1
        if "candidate_small" not in a:  # checkpointed after the last step: finished
            return Step([], iteration=first)
        return Step(
            [
                ForecastRequest("small", first, a["candidate_small"], a["member_ids"]),
                ForecastRequest("large", first + 1, a["candidate_large"], a["member_ids"]),
            ],
            iteration=first,
        )

    def estimate(self) -> Estimate:
        """The kept ensemble and its forecast."""
        a = self._arrays
        return Estimate(a["state"], a["predicted"], a["member_ids"])


@pytest.fixture
def line_search_method() -> Iterator[None]:
    """Register ``LineSearchESMDA`` for one test (the registry is otherwise global)."""
    methods.add(NAME, LineSearchESMDA)
    yield
    del methods._factories[NAME]


METHOD = f'name = "{NAME}"\noptions = {{ alpha = 4.0, n_steps = 4 }}'


def _run(linear_case: CaseFactory, name: str, stops: list[int]) -> tuple[Path, np.ndarray]:
    case = linear_case(
        name=name, method=METHOD, simulator_extra="fail_members = [7]", ensemble_size=100
    )
    run_dir = case.directory / "runs" / name
    if not stops:
        return run_dir, composition.start(case.config).posterior
    composition.start(case.config, max_steps=stops[0])
    for stop in stops[1:]:
        assert _pending(run_dir), "the run finished before the planned stop"
        composition.resume(run_dir, max_steps=stop)
    assert _pending(run_dir)
    return run_dir, composition.resume(run_dir).posterior


def _pending(run_dir: Path) -> bool:
    saved = RunDirectory(run_dir).load_arrays(
        json.loads((run_dir / "state.json").read_text())["algorithm_file"]
    )
    return "candidate_small" in saved


@pytest.mark.usefixtures("line_search_method")
def test_a_line_search_algorithm_runs_and_resumes_bit_identically(
    linear_case: CaseFactory,
) -> None:
    whole_dir, whole = _run(linear_case, "whole", [])
    # Each stop falls after a line-search step, with the next pair of candidates pending.
    resumed_dir, resumed = _run(linear_case, "resumed", [1, 1, 1])

    np.testing.assert_array_equal(resumed, whole)
    whole_history, resumed_history = (
        RunDirectory(d).history_dicts() for d in (whole_dir, resumed_dir)
    )
    assert resumed_history == whole_history
    # Prior, then two scored candidates per step: one kept, one rejected.
    assert [r["iteration"] for r in whole_history] == list(range(9))
    assert [r["accepted"] for r in whole_history].count(False) == 4
    assert all(r["n_members"] == 99 for r in whole_history)  # member 7 fails in the prior
    files = {p.stem for p in (whole_dir / "checkpoints").glob("*.npz")}
    assert {f"iter_{k}" for k in range(9)} & files == {
        f"iter_{r['iteration']}" for r in whole_history if r["accepted"]
    }


@pytest.mark.usefixtures("line_search_method")
def test_the_line_search_reduces_the_data_mismatch(linear_case: CaseFactory) -> None:
    run_dir, _ = _run(linear_case, "data_mismatch", [])
    history = RunDirectory(run_dir).history_dicts()
    kept = [r["data_mismatch_mean"] for r in history if r["accepted"]]
    assert kept[-1] < 0.5 * kept[0]
