"""ERT runs APEX's forecasts in batches, without ERT.

The ``ert`` executor gives each forecast to ``ert ensemble_experiment``; here a stand-in
program (``tests/unit/integrations/fake_ert.py``) computes the linear case's forward model
from the fields APEX wrote into the run paths. The run must be bit-identical to the same
case on the ``linear`` simulator and the process executor, and so must a resumed one.
"""

import importlib
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.domain.observations import DataKey
from apex.services import composition
from apex.simulation.ert import ErtCase


class Case(Protocol):
    """The parts of the ``linear_case`` fixture's case used here."""

    directory: Path
    config: Path


# tests/ is not an importable package, so spell out the fixture's type here.
CaseFactory = Callable[..., Case]

FAKE_ERT = Path(__file__).resolve().parents[1] / "unit" / "integrations" / "fake_ert.py"


def _read_prediction(
    self: ErtCase, run_path: Path, realization: int, keys: Sequence[DataKey], iteration: int = 0
) -> NDArray[np.float64]:
    path = run_path / "pred.npy"
    if not path.exists():
        raise SimulationError(f"no results in {run_path}")
    if not keys:  # a prediction's pilot asks for none
        return np.empty(0)
    values: NDArray[np.float64] = np.load(path)
    assert values.shape == (len(keys),)
    return values


@pytest.fixture(autouse=True)
def _predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    # The stand-in writes pred.npy instead of an Eclipse summary.
    monkeypatch.setattr(ErtCase, "read", _read_prediction)


def _ert_case(
    linear_case: CaseFactory, name: str, workers: int = 3, field: str = "x", **kwargs: object
) -> Case:
    case = linear_case(name=name, **kwargs)
    program = case.directory / "ert"
    program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_ERT}" "$@"\n')
    program.chmod(0o755)
    (case.directory / "case.ert").write_text(
        "-- an ERT case: what APEX owns in a batch is dropped\n"
        "NUM_REALIZATIONS 100\nRUNPATH runs/realization-<IENS>/iter-<ITER>\n"
        "FIELD X PARAMETER x.grdecl INIT_FILES:prior/X_<IENS>.grdecl FORWARD_INIT:False\n"
        "OBS_CONFIG observations.txt\nQUEUE_OPTION LOCAL MAX_RUNNING 2\n"
        "ECLBASE CASE\nFORWARD_MODEL LINEAR_MODEL\n"
    )
    text = case.config.read_text()
    text = re.sub(
        r"\[simulator\].*?(?=\[method\])",
        '[simulator]\nname = "ert"\n\n[simulator.options]\ncase_file = "case.ert"\n'
        f'eclbase = "CASE"\nfields = {{ {field} = {{ keyword = "X", file = "x.grdecl" }} }}\n\n'
        f'[execution]\nname = "ert"\noptions = {{ workers = {workers}, ert_executable = '
        f'"{program}" }}\n\n',
        text,
        flags=re.DOTALL,
    )
    case.config.write_text(text)
    return case


GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 6"
)


@pytest.mark.parametrize("method", ["esmda", "gies"])
def test_ert_batches_are_bit_identical_to_the_process_executor(
    linear_case: CaseFactory, method: str
) -> None:
    # GIES also forecasts its mean model (member -1) and rejected trials.
    extra: dict[str, Any] = {"method": GIES} if method == "gies" else {}
    reference = linear_case(
        name="process",
        ensemble_size=24,
        execution='name = "process"\noptions = { workers = 3 }',
        **extra,
    )
    batched = _ert_case(linear_case, "ert", ensemble_size=24, **extra)

    expected = composition.start(reference.config)
    result = composition.start(batched.config)

    np.testing.assert_array_equal(result.posterior, expected.posterior)
    assert result.history == expected.history
    iteration = batched.directory / "runs" / "ert" / "simulations" / "iter_0"
    assert (iteration / "member_5" / "APEX_MEMBER").read_text() == "5\n"
    assert not list(batched.directory.glob(".apex_batch_*"))  # the batch's case is removed
    assert len(list((iteration / ".ert" / "runs").iterdir())) == 0  # moved to the members


def test_a_resumed_ert_run_is_identical_to_an_uninterrupted_one(
    linear_case: CaseFactory,
) -> None:
    whole = _ert_case(linear_case, "whole", ensemble_size=20)
    parts = _ert_case(linear_case, "parts", ensemble_size=20)

    reference = composition.start(whole.config)
    first = composition.start(parts.config, max_steps=2)
    resumed = composition.resume(parts.directory / "runs" / "parts")

    assert not first.complete
    np.testing.assert_array_equal(resumed.posterior, reference.posterior)
    assert resumed.history == reference.history


def test_realizations_without_results_are_failed_members(linear_case: CaseFactory) -> None:
    case = _ert_case(linear_case, "failing", ensemble_size=30, workers=4)
    # Batch positions 3 and 17 of the prior forecast are members 3 and 17.
    (case.directory / "FAIL_REALIZATIONS").write_text("3 17\n")

    result = composition.start(case.config)

    assert result.posterior.shape == (3, 28)
    assert {3, 17}.isdisjoint(result.member_ids.tolist())


def test_the_ert_simulator_and_executor_go_together(linear_case: CaseFactory) -> None:
    case = _ert_case(linear_case, "pairing")
    text = case.config.read_text()
    case.config.write_text(
        re.sub(r'(\[execution\]\nname = )"ert"\noptions = .*\n', r'\1"serial"\n', text)
    )

    with pytest.raises(ConfigError, match="go together"):
        composition.build_run(composition.load_config(case.config))


def test_the_batch_case_keeps_the_forward_model_and_drops_what_apex_owns(
    linear_case: CaseFactory,
) -> None:
    case = _ert_case(linear_case, "lines")
    simulator = ErtCase(
        case.directory / "case.ert", {"x": {"keyword": "X", "file": "x.grdecl"}}, "CASE"
    )

    kept = [line.split()[0] for line in simulator.forward_config() if line.split()]

    assert kept == ["--", "ECLBASE", "FORWARD_MODEL"]


PARAMETERIZED = (
    '\n[[parameterizations]]\nname = "test_affine"\noptions = { scale = 2.0, shift = 1.0 }\n'
)


def test_a_parameterized_ert_run_writes_the_properties(linear_case: CaseFactory) -> None:
    """ERT's forward model reads the property y = 2 x + 1."""
    # Registers test_affine, under the module name the process executor's workers import.
    importlib.import_module("tests.workflows.toy_parameterization")

    reference = linear_case(
        name="process", ensemble_size=20, execution='name = "process"\noptions = { workers = 3 }'
    )
    batched = _ert_case(linear_case, "ert", ensemble_size=20, field="y")
    for case in (reference, batched):
        text = case.config.read_text().replace('variables = ["x"]', 'variables = ["y"]')
        case.config.write_text(text + PARAMETERIZED)

    expected = composition.start(reference.config)
    result = composition.start(batched.config)

    np.testing.assert_array_equal(result.posterior, expected.posterior)
    assert result.history == expected.history
    member = batched.directory / "runs" / "ert" / "simulations" / "iter_0" / "member_3"
    from apex.simulation.grdecl import read_grdecl

    prior = np.load(batched.directory / "runs" / "ert" / "checkpoints" / "iter_0.npz")
    column = list(prior["member_ids"]).index(3)
    np.testing.assert_array_equal(
        read_grdecl(member / "x.grdecl", "X"), 2.0 * prior["state"][:, column] + 1.0
    )


def _read_batch_from_run_paths(
    self: ErtCase, storage: Path, keys: Sequence[DataKey], n: int, environment: object
) -> NDArray[np.float64]:
    # The batch's storage, as ERT would hold it: each realization's responses.
    runs = storage.parent / "runs"
    out = np.full((len(keys), n), np.nan)
    for i in range(n):
        path = runs / f"realization-{i}" / "pred.npy"
        if path.exists():
            out[:, i] = np.load(path)
    return out


def test_responses_read_from_the_batch_storage_are_bit_identical(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Responses of any observation type come from the batch's storage."""
    monkeypatch.setattr(ErtCase, "read_batch", _read_batch_from_run_paths)
    reference = linear_case(
        name="process", ensemble_size=24, execution='name = "process"\noptions = { workers = 3 }'
    )
    batched = _ert_case(linear_case, "ert", ensemble_size=24)
    text = batched.config.read_text().replace(
        'eclbase = "CASE"\n',
        'eclbase = "CASE"\nresponses = "storage"\nresults_program = ["unused"]\n',
    )
    batched.config.write_text(text)
    (batched.directory / "FAIL_REALIZATIONS").write_text("5\n")

    expected_failure = composition.start(batched.config)

    assert 5 not in expected_failure.member_ids.tolist()  # NaN responses: a failed member
    forward = (batched.directory / "case.ert").read_text()
    assert "OBS_CONFIG" in forward
    simulator = ErtCase(
        batched.directory / "case.ert", {"x": {"keyword": "X", "file": "x.grdecl"}}, "CASE",
        responses="storage", results_program=["unused"],
    )  # fmt: skip
    assert any(line.startswith("OBS_CONFIG") for line in simulator.forward_config())
    batched_ok = _ert_case(linear_case, "ert_ok", ensemble_size=24)
    batched_ok.config.write_text(
        batched_ok.config.read_text().replace(
            'eclbase = "CASE"\n',
            'eclbase = "CASE"\nresponses = "storage"\nresults_program = ["unused"]\n',
        )
    )
    np.testing.assert_array_equal(
        composition.start(batched_ok.config).posterior,
        composition.start(reference.config).posterior,
    )


@pytest.mark.parametrize("explicit_times", [True, False])
def test_a_b2_run_is_predicted_through_ert_batches(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch, explicit_times: bool
) -> None:
    """A prediction of an ``ert``-simulator run runs its members in ERT
    batches and reads their series from the run paths (a pilot batch without times)."""
    from apex.core.simulation import MemberSeries, SeriesRequest
    from apex.services import prediction

    case = _ert_case(linear_case, "b2", ensemble_size=12)
    composition.start(case.config)
    n_data = len(np.load(case.directory / "forward.npz")["matrix"])
    times = tuple(f"t{i}" for i in range(n_data))

    def read_series(
        self: ErtCase, run_path: Path, realization: int, request: SeriesRequest, iteration: int = 0
    ) -> MemberSeries:
        values = np.load(run_path / "pred.npy")
        return MemberSeries(times, ("y",), values[None, :])

    monkeypatch.setattr(ErtCase, "read_series", read_series, raising=False)
    outputs = "\ntimes = [" + ", ".join(f'"{t}"' for t in times) + "]" if explicit_times else ""
    path = case.directory / "future.toml"
    path.write_text(
        '[prediction]\nname = "future"\nrun_dir = "runs/b2"\nensembles = ["posterior", "prior"]\n'
        f'\n[prediction.outputs]\nvectors = ["y"]{outputs}\n'
    )

    result = prediction.predict(path)

    forward = np.load(case.directory / "forward.npz")["matrix"]
    results = composition.load_results(case.directory / "runs" / "b2")
    posterior = results.iterate(results.posterior_iteration).state["x"]
    np.testing.assert_allclose(result.forecasts["posterior"].values[0], forward @ posterior)
    assert result.forecasts["posterior"].times == times
