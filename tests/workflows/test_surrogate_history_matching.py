"""History matching with a surrogate (validation 3).

The cases are linear (the ``linear`` simulator, ``y = G x``) and the surrogate a test-only
affine least-squares map (``surrogate_cases.Affine``, registered here, never shipped). Fitted
on 40 runs of a Latin hypercube over ``[-6, 6]^3``, it reproduces the linear model to
round-off inside that box. That is the independent reference: ES-MDA with it must give the
plain run's posterior to round-off, hence the exact Kalman posterior within the ensemble's
sampling error (the plain run's own tests hold that, ``test_history_matching_e2e.py``).
Here the two runs share every random draw (the same seed and streams; the surrogate's
stream is new and draws only at checks), so their posteriors differ only by the affine fit's
round-off: a tolerance of ``1e-9`` (absolute, on values of order 1) is far above that (seen
~1e-15) and far below any sampling error (~1e-2 for 100 members).

A deliberately biased surrogate (the exact one with every intercept shifted by ``BIAS``
observation standard deviations, its manifest still recording a passed gate) fails its
checks, and each of ``stop``, ``refit`` and ``simulator`` behaves as §5 says.
"""

import importlib.util
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

from apex.core.errors import ConfigError, SurrogateDomainError, WorkflowError
from apex.core.options import InputFile
from apex.core.plugins import simulators, surrogate_models
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet
from apex.execution.local import SerialExecutor
from apex.services import composition, surrogates
from apex.services.summaries import summarize
from apex.services.validation import validate_history_match
from apex.simulation.surrogate import SurrogateSimulator
from apex.storage.run_directory import RunDirectory
from apex.storage.run_results import RunResults
from apex.storage.surrogate_directory import SurrogateDirectory, load_snapshots
from apex.storage.surrogate_files import read_cost, read_surrogate, write_surrogate
from apex.workflows.history_matching import STREAM_NAMES, RandomStreams
from apex.workflows.surrogate_forecasts import SurrogateCheckFailed


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()
PLUGIN = "test_affine_hm"
#: The biased model's shift, in observation standard deviations (the gate allows 0.5).
BIAS = 3.0
#: Two posteriors that differ only by the affine fit's round-off (module docstring).
ROUND_OFF = 1e-9


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register(PLUGIN)(cases.Affine)
    surrogates.fitted_model.cache_clear()
    try:
        yield
    finally:
        surrogate_models.remove(PLUGIN)


@pytest.fixture
def built(tmp_path: Path) -> Path:
    """Build the exact surrogate of the case in ``tmp_path``; return its manifest."""
    cases.write_case(tmp_path, name="base")
    result = surrogates.build(cases.write_build(tmp_path, PLUGIN, case="base.toml"))
    assert result.summary.gate is not None
    assert result.summary.gate.passed
    return tmp_path / "runs" / "lin" / "model" / "surrogate.json"


def _biased(model: Path) -> Path:
    """Write the exact model with every intercept shifted by ``BIAS`` sigma, gate as passed.

    Written beside the build's model (``model_biased/``), so that it names the build's
    snapshot set (``../snapshots.npz``) and can be refitted from it.
    """
    stored = read_surrogate(model)
    fitted = cases.Affine.from_arrays(stored.manifest.model, stored.arrays)
    fitted.weights = fitted.weights.copy()
    fitted.weights[:, -1] += BIAS * np.sqrt(cases.VARIANCE)
    target = model.parent.parent / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        target,
        plugin=manifest.plugin,
        model=fitted,
        domain=manifest.training_domain(),
        provenance=manifest.provenance,
        options=manifest.options,
        gate=manifest.gate,
        snapshots=manifest.snapshots,
        fit=manifest.fit,
        apex_commit=manifest.apex_commit,
    )
    return target


def _section(model: Path, **options: object) -> str:
    lines = ["[surrogate]", f'model_file = "{model}"']
    lines += [f"{key} = {json.dumps(value)}" for key, value in options.items()]
    return "\n".join(lines)


def _events(run: Path) -> list[dict[str, Any]]:
    lines = (run / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _named(run: Path, event: str) -> list[dict[str, Any]]:
    return [e for e in _events(run) if e.get("event") == event]


def _run(folder: Path, name: str) -> Path:
    return folder / "runs" / name


# ------------------------------------------------------------------- exact surrogate


def test_an_exact_surrogate_gives_the_simulators_posterior(tmp_path: Path, built: Path) -> None:
    """ES-MDA with an exact surrogate: the plain run's posterior, to round-off (§5, val. 3)."""
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built, check_members=5))
    result = composition.start(config)
    plain = composition.start(cases.write_case(tmp_path, name="plain"))
    np.testing.assert_allclose(result.posterior, plain.posterior, rtol=0, atol=ROUND_OFF)
    run = _run(tmp_path, "hm")
    results = RunResults(run, composition.build_layout(composition.load_config(config)))
    # Every forecast the method saw is the surrogate's, and says so.
    assert set(results.sources().values()) == {"surrogate"}
    for k in results.iterations:
        iterate = results.iterate(k)
        assert iterate.forecast_source == ("surrogate",) * iterate.member_ids.size
        assert iterate.surrogate_id == read_surrogate(built).identity
        assert iterate.predicted.flags.c_contiguous
    # Check iterations 3 (check_every = 3 by default, 4 steps): 5 members run by the simulator.
    checks = _named(run, "surrogate_check")
    assert [c["iteration"] for c in checks] == [3]
    assert checks[0]["passed"]
    assert len(checks[0]["members"]) == 5
    assert (run / "checkpoints" / "check_3.npz").exists()
    # The posterior is the simulator's forecast (full_4.npz), and posterior.npz holds it.
    full = results.full(4)
    assert full is not None
    assert full.forecast_source == ("simulator",) * full.member_ids.size
    forward = cases.forward_matrix()
    with np.load(run / "checkpoints" / "posterior.npz") as posterior:
        np.testing.assert_array_equal(posterior["predicted"], full.predicted)
        np.testing.assert_allclose(
            posterior["predicted"], forward @ posterior["x"], rtol=0, atol=1e-12
        )
        assert set(posterior["forecast_source"]) == {"simulator"}
    summary = json.loads((run / "summary.json").read_text())
    report = summary["surrogate"]
    assert report["posterior"]["forecast"] == "simulator"
    assert not report["posterior"]["beyond_gate"]
    for metric in ("data_mismatch", "normalized_data_mismatch"):
        assert abs(report["posterior"]["difference"][metric]) < ROUND_OFF
    assert summary["final_source"] == "simulator"
    assert summary["final_data_mismatch_mean"] == report["posterior"]["simulator"]["data_mismatch"]
    assert report["evaluations"] == 5 * 100
    assert report["simulator_runs"] == 5 + 100  # one check, the posterior's forecast
    # Decision 14: the forecasts' cost, from the events, against the build's full runs.
    timing = report["timing"]
    assert timing["surrogate"]["members"] == report["evaluations"]
    assert timing["simulator"]["members"] == report["simulator_runs"]
    assert timing["surrogate"]["worker_seconds_per_member"] is not None
    cost = read_cost(built)
    assert cost is not None
    assert timing["full_model"] == cost.model_dump(mode="json")
    assert cost.runs == len(SurrogateDirectory(built.parent.parent).journal())
    assert timing["simulator_equivalent_worker_seconds"] == pytest.approx(
        report["evaluations"] * cost.mean_seconds
    )
    # model_change and the other metrics are recorded as in a plain run.
    labels = results.metric_labels()
    assert {"data_mismatch", "normalized_data_mismatch", "rmse", "model_change"} <= set(labels)


def test_the_plan_counts_simulator_runs_apart(tmp_path: Path, built: Path) -> None:
    """`apex validate`: surrogate evaluations apart; only simulator runs count (§5)."""
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built, check_members=5))
    validation = validate_history_match(config)
    assert validation.valid, validation.errors
    plan = validation.plan
    assert plan is not None
    assert plan.surrogate_evaluations == 5 * 100
    assert plan.max_simulator_runs == 1 * 5 + 100  # one check iteration (3) of 5, the posterior
    assert plan.surrogate is not None
    assert plan.surrogate.startswith(PLUGIN)
    capped = cases.write_case(
        tmp_path, name="capped", surrogate=_section(built, max_evaluations=499)
    )
    refused = validate_history_match(capped)
    assert not refused.valid
    assert "max_evaluations" in refused.errors[0]


def test_the_evaluations_bound_stops_a_run(tmp_path: Path, built: Path) -> None:
    """A run that reaches ``max_evaluations`` stops, loudly, at its last checkpoint."""
    config = cases.write_case(
        tmp_path, name="hm", surrogate=_section(built, max_evaluations=250, check_every=0)
    )
    with pytest.raises(WorkflowError, match="max_evaluations"):
        composition.start(config)
    run = _run(tmp_path, "hm")
    assert _named(run, "surrogate_budget_exhausted")
    assert RunDirectory(run).read_state()["surrogate"]["evaluations"] == 200


# ------------------------------------------------------------------- failing checks


def test_a_failing_check_stops_the_run_before_the_update(tmp_path: Path, built: Path) -> None:
    """``stop``: loud, before the update, at the last checkpoint; resumable (§5)."""
    biased = _biased(built)
    config = cases.write_case(
        tmp_path, name="hm", surrogate=_section(biased, check_every=1, check_members=4)
    )
    with pytest.raises(SurrogateCheckFailed, match="fails its check"):
        composition.start(config)
    run = _run(tmp_path, "hm")
    [failed] = _named(run, "surrogate_check_failed")
    assert failed["iteration"] == 1
    errors = failed["errors"]["history_matching"]
    assert min(errors.values()) > 0.5  # every datatype is off by BIAS sigma
    # Stopped before the update of iteration 1: the last checkpoint is the prior's.
    history = RunDirectory(run).history_dicts()
    assert [h["iteration"] for h in history] == [0]
    assert not (run / "checkpoints" / "iter_1.npz").exists()
    # Resumable once [surrogate] says what to do: the simulator from here on.
    snapshot = json.loads((run / "config.json").read_text())
    snapshot["surrogate"]["on_check_failure"] = "simulator"
    (run / "config.json").write_text(json.dumps(snapshot))
    result = composition.resume(run)
    assert result.complete
    sources = RunResults(run, composition.build_layout(composition.load_config(config))).sources()
    assert sources == {
        0: "surrogate",
        1: "simulator",
        2: "simulator",
        3: "simulator",
        4: "simulator",
    }
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert report["switched_to_simulator_at"] == 1
    # The posterior's forecast is the simulator's already: none is repeated.
    assert report["posterior"]["forecast"] == "simulator already"
    assert not (run / "checkpoints" / "full_4.npz").exists()


def test_a_failing_check_switches_to_the_simulator(tmp_path: Path, built: Path) -> None:
    """``simulator``: the rest of the run uses the simulator, recorded (§5)."""
    biased = _biased(built)
    config = cases.write_case(
        tmp_path,
        name="hm",
        surrogate=_section(biased, check_every=2, check_members=4, on_check_failure="simulator"),
    )
    result = composition.start(config)
    plain = composition.start(cases.write_case(tmp_path, name="plain"))
    run = _run(tmp_path, "hm")
    [switched] = _named(run, "surrogate_switched")
    assert switched["iteration"] == 2
    layout = composition.build_layout(composition.load_config(config))
    results = RunResults(run, layout)
    assert results.sources() == {
        0: "surrogate",
        1: "surrogate",
        2: "simulator",
        3: "simulator",
        4: "simulator",
    }
    # Iterations 1 and 2 were updated from biased forecasts: not the plain posterior.
    assert np.abs(result.posterior - plain.posterior).max() > 1e-3
    # From iteration 2 on, the forecasts are the simulator's: G x exactly.
    forward = cases.forward_matrix()
    iterate = results.iterate(3)
    np.testing.assert_allclose(iterate.predicted, forward @ iterate.state["x"], atol=1e-12)


def test_a_failing_check_refits_and_repeats_the_forecast(tmp_path: Path, built: Path) -> None:
    """``refit``: the checked runs join the snapshots; the refit is exact again (§5)."""
    biased = _biased(built)
    config = cases.write_case(
        tmp_path,
        name="hm",
        surrogate=_section(biased, check_every=1, check_members=6, on_check_failure="refit"),
    )
    composition.start(config)
    run = _run(tmp_path, "hm")
    [refit] = _named(run, "surrogate_refit")
    assert refit["iteration"] == 1
    assert refit["passed"]
    folder = run / "surrogates" / "refit_1"
    stored = read_surrogate(folder / "model" / "surrogate.json")
    snapshots = load_snapshots(folder / "snapshots.npz")
    assert snapshots.n_runs == 40 + 6
    assert stored.manifest.gate is not None
    assert len(stored.manifest.gate.held_out) == 10 + 3  # refit_held_out = 0.5 of 6
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert [r["identity"] for r in report["refits"]] == [stored.identity]
    assert report["active_identity"] == stored.identity
    # After the refit every check passes: the refit is exact on the linear case.
    assert [c["passed"] for c in report["checks"]] == [False, True, True, True]
    assert report["evaluations"] == 5 * 100 + 100  # the repeated forecast counts
    layout = composition.build_layout(composition.load_config(config))
    results = RunResults(run, layout)
    assert results.iterate(1).surrogate_id == stored.identity
    assert results.iterate(0).surrogate_id == read_surrogate(biased).identity
    # Reproducible from the run directory alone: the refit's snapshots, split and fit state.
    model = cases.Affine()
    held = set(stored.manifest.gate.held_out)
    training = snapshots.columns([i for i, r in enumerate(snapshots.run_ids) if r not in held])
    rng = np.random.default_rng()
    assert stored.manifest.fit.stream_state is not None
    rng.bit_generator.state = dict(stored.manifest.fit.stream_state)
    model.fit(training, rng)
    np.testing.assert_array_equal(model.weights, stored.arrays["weights"])


def test_a_model_naming_a_retired_option_refits(
    tmp_path: Path, built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plugin's retired option in an older manifest: the refit creates the plugin without it.

    The loader drops ``retired_options`` before its check; the refit, which creates the plugin
    from the manifest's options, must too, and the refitted manifest no longer names it.
    """
    biased = _biased(built)
    document = json.loads(biased.read_text())
    document["options"] = {**document["options"], "old_setting": 8}
    biased.write_text(json.dumps(document))
    monkeypatch.setattr(cases.Affine, "retired_options", frozenset({"old_setting"}), raising=False)
    config = cases.write_case(
        tmp_path,
        name="hm",
        surrogate=_section(biased, check_every=1, check_members=6, on_check_failure="refit"),
    )
    composition.start(config)
    run = _run(tmp_path, "hm")
    [refit] = _named(run, "surrogate_refit")
    assert refit["passed"]
    stored = read_surrogate(run / "surrogates" / "refit_1" / "model" / "surrogate.json")
    assert "old_setting" not in stored.manifest.options


def test_refits_are_bounded(tmp_path: Path, built: Path) -> None:
    """After ``max_refits`` a failing check stops the run (§5)."""
    biased = _biased(built)
    config = cases.write_case(
        tmp_path,
        name="hm",
        surrogate=_section(biased, check_every=1, on_check_failure="refit", max_refits=0),
    )
    with pytest.raises(SurrogateCheckFailed, match="max_refits"):
        composition.start(config)


def test_resume_is_bit_identical_across_checks_and_a_refit(tmp_path: Path, built: Path) -> None:
    """Stopped after each step and resumed: every saved array equal, bit for bit (§5)."""
    biased = _biased(built)
    section = _section(biased, check_every=1, check_members=6, on_check_failure="refit")
    whole = cases.write_case(tmp_path / "whole", name="hm", method=cases.GIES, surrogate=section)
    composition.start(whole)
    parts = cases.write_case(tmp_path / "parts", name="hm", method=cases.GIES, surrogate=section)
    composition.start(parts, max_steps=1)
    for _ in range(10):
        if json.loads((_run(tmp_path / "parts", "hm") / "summary.json").read_text())["complete"]:
            break
        composition.resume(_run(tmp_path / "parts", "hm"), max_steps=1)
    a, b = _run(tmp_path / "whole", "hm"), _run(tmp_path / "parts", "hm")
    files = sorted(p.name for p in (a / "checkpoints").glob("*.npz"))
    assert files == sorted(p.name for p in (b / "checkpoints").glob("*.npz"))
    assert any(name.startswith("check_") for name in files)
    for name in files:
        with np.load(a / "checkpoints" / name) as left, np.load(b / "checkpoints" / name) as right:
            assert sorted(left.files) == sorted(right.files), name
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key], err_msg=f"{name}:{key}")
    for name in ("history.json",):
        assert (a / name).read_text() == (b / name).read_text()
    left_refit = a / "surrogates" / "refit_1" / "model" / "surrogate.npz"
    right_refit = b / "surrogates" / "refit_1" / "model" / "surrogate.npz"
    assert left_refit.read_bytes() == right_refit.read_bytes()
    state_a, state_b = RunDirectory(a).read_state(), RunDirectory(b).read_state()
    assert state_a["random_streams"] == state_b["random_streams"]
    assert state_a["surrogate"] == state_b["surrogate"]


def test_resume_refuses_a_changed_model(tmp_path: Path, built: Path) -> None:
    """A model file whose identity differs from the recorded one is refused on resume."""
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built))
    composition.start(config, max_steps=1)
    stored = read_surrogate(built)
    manifest = stored.manifest
    write_surrogate(
        built, plugin=manifest.plugin,
        model=cases.Affine.from_arrays(manifest.model, stored.arrays),
        domain=manifest.training_domain(), provenance=manifest.provenance,
        options=manifest.options, gate=manifest.gate, snapshots=manifest.snapshots,
        fit=manifest.fit, apex_commit="another commit",
    )  # fmt: skip
    surrogates.fitted_model.cache_clear()
    with pytest.raises(ConfigError, match="identity"):
        composition.resume(_run(tmp_path, "hm"))


# ------------------------------------------------------------------- the domain


def test_members_outside_the_domain_stop_the_run(tmp_path: Path, built: Path) -> None:
    """``outside = "stop"``: a ``SurrogateDomainError``, never a dropped member (§4)."""
    config = cases.write_case(tmp_path, name="hm", prior_std=3.0, surrogate=_section(built))
    with pytest.raises(SurrogateDomainError, match="outside the surrogate's domain"):
        composition.start(config)
    [outside] = _named(_run(tmp_path, "hm"), "surrogate_outside")
    assert outside["policy"] == "stop"
    assert outside["members"]


def test_members_outside_the_domain_are_simulated(tmp_path: Path, built: Path) -> None:
    """``outside = "simulator"``: those members run by the simulator, recorded as such."""
    section = _section(built, outside="simulator")
    config = cases.write_case(tmp_path, name="hm", prior_std=3.0, surrogate=section)
    result = composition.start(config)
    plain = composition.start(cases.write_case(tmp_path, name="plain", prior_std=3.0))
    np.testing.assert_allclose(result.posterior, plain.posterior, rtol=0, atol=ROUND_OFF)
    run = _run(tmp_path, "hm")
    results = RunResults(run, composition.build_layout(composition.load_config(config)))
    prior = results.iterate(0)
    assert "simulator" in prior.forecast_source
    assert "surrogate" in prior.forecast_source
    assert results.sources()[0] == "mixed"
    outside = [i for i, s in zip(prior.member_ids, prior.forecast_source, strict=True)
               if s == "simulator"]  # fmt: skip
    assert all(
        np.any(np.abs(prior.state["x"][:, list(prior.member_ids).index(m)]) > 5.0) for m in outside
    )


# ------------------------------------------------------------------- cross validation

CV = '[validation]\ncv = { datatypes = ["WOPR:P2", "WBHP:P2"] }\nstopping = { name = "select" }'


def test_cross_validation_decides_on_the_surrogate_and_reports_both(
    tmp_path: Path, built: Path
) -> None:
    """The rule's choice is the surrogate's; the chosen iterate is the simulator's (§5)."""
    section = _section(built, check_every=2, check_members=4)
    config = cases.write_case(
        tmp_path, name="hm", method=cases.GIES, validation=CV, surrogate=section
    )
    result = composition.start(config)
    plain = composition.start(
        cases.write_case(tmp_path, name="plain", method=cases.GIES, validation=CV)
    )
    assert result.selected_iteration == plain.selected_iteration
    np.testing.assert_allclose(result.posterior, plain.posterior, rtol=0, atol=ROUND_OFF)
    run = _run(tmp_path, "hm")
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    selected = result.selected_iteration
    assert selected is not None
    assert report["posterior"]["iteration"] == selected
    assert (run / "checkpoints" / f"full_{selected}.npz").exists()
    for side in ("surrogate", "simulator"):
        assert set(report["posterior"][side]) == {
            "data_mismatch",
            "normalized_data_mismatch",
            "cv_normalized_data_mismatch",
        }
    assert set(report["posterior"]["gates"]) == {"history_matching", "cv"}
    # The checks gate the cv keys apart.
    check = report["checks"][0]
    assert set(check["errors"]) == {"history_matching", "cv"}
    assert set(check["errors"]["cv"]) == {"WOPR:P2", "WBHP:P2"}


# ------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ('[model_error]\nname = "pg_lstm"', "with [model_error]"),
        (None, "fidelity levels"),
    ],
)
def test_refused_combinations(tmp_path: Path, built: Path, extra: str | None, match: str) -> None:
    """[model_error] and fidelity levels are refused with a surrogate ("Refused")."""
    method = cases.ESMDA
    if extra is None:
        method = (
            'name = "multilevel_esmda"\noptions = { inflation = [1.0], levels = '
            "[{ members = 40, weight = 0.5 }, { members = 60, weight = 0.5 }] }"
        )
        extra = ""
    config = cases.write_case(
        tmp_path, name="hm", method=method, surrogate=_section(built), extra=extra
    )
    found = validate_history_match(config)
    assert not found.valid
    assert match in " ".join(found.errors)


def test_the_adjoint_update_is_refused(tmp_path: Path, built: Path) -> None:
    """Surrogate gradients are a later decision: ``sensitivities`` is refused."""
    method = (
        'name = "esmda"\noptions = { inflation = [1.0] }\n'
        'strategy = { name = "approx", options = { sensitivities = true } }'
    )
    config = cases.write_case(tmp_path, name="hm", method=method, surrogate=_section(built))
    found = validate_history_match(config)
    assert not found.valid
    assert "adjoint" in " ".join(found.errors) or "sensitivities" in " ".join(found.errors)


def test_a_model_whose_gate_failed_is_refused(tmp_path: Path, built: Path) -> None:
    stored = read_surrogate(built)
    manifest = stored.manifest
    assert manifest.gate is not None
    failed = built.parent.parent / "model_failed" / "surrogate.json"
    write_surrogate(
        failed, plugin=manifest.plugin,
        model=cases.Affine.from_arrays(manifest.model, stored.arrays),
        domain=manifest.training_domain(), provenance=manifest.provenance,
        options=manifest.options, gate=manifest.gate.model_copy(update={"passed": False}),
        snapshots=manifest.snapshots, fit=manifest.fit,
    )  # fmt: skip
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(failed))
    found = validate_history_match(config)
    assert not found.valid
    assert "did not pass its gate" in found.errors[0]


def test_a_model_of_another_stack_is_refused(tmp_path: Path, built: Path) -> None:
    other = tmp_path / "other"
    cases.write_case(other, name="base")
    np.savez(other / "forward.npz", matrix=2 * cases.forward_matrix())
    config = cases.write_case(other, name="hm", surrogate=_section(built))
    np.savez(other / "forward.npz", matrix=2 * cases.forward_matrix())  # write_case rewrote it
    found = validate_history_match(config)
    assert not found.valid
    assert "another simulator stack" in found.errors[0]


def test_unknown_keys_are_refused(tmp_path: Path, built: Path) -> None:
    """A run observing a key the surrogate was not gated on is refused at validation."""
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built))
    observations = tmp_path / "observations.csv"
    observations.write_text(observations.read_text() + "5,WOPR:P1,1.0,0.04\n")
    found = validate_history_match(config)
    assert not found.valid
    assert "not trained and gated on" in found.errors[0]


def test_a_tightened_gate_only(tmp_path: Path, built: Path) -> None:
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built, sigma_ratio=0.9))
    found = validate_history_match(config)
    assert not found.valid
    assert "only be tightened" in found.errors[0]


# ------------------------------------------------------------------- executors, tokens


def test_a_surrogate_simulator_holds_cpu_tokens_and_never_drops_members(built: Path) -> None:
    """Surrogate forecasts hold ``cpu`` tokens; outside the domain is an error, not a failure."""

    class Recording:
        def __init__(self) -> None:
            self.held: list[dict[str, int]] = []

        def hold(self, needs: dict[str, int]) -> "Recording":
            self.held.append(dict(needs))
            return self

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_: object) -> None:
            return None

    active = surrogates.active_model(built)
    pools = Recording()
    executor = SerialExecutor(pools=pools)  # type: ignore[arg-type]
    keys = active.model.keys
    from apex.core.simulation import MemberInput

    inside = MemberInput(member_id=0, state={"x": np.array([0.1, 0.2, -0.3])})
    [out] = executor.run_all(active.simulator, [inside], keys)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, cases.forward_matrix() @ inside.state["x"], atol=1e-12)
    assert pools.held == [{"cpu": 1}]
    outside = MemberInput(member_id=1, state={"x": np.array([9.0, 0.0, 0.0])})
    with pytest.raises(SurrogateDomainError):
        executor.run_all(active.simulator, [outside], keys)
    # Pickled as a reference: loaded again (and its identity checked) where it is used.
    import pickle

    copy = pickle.loads(pickle.dumps(active.simulator))
    assert isinstance(copy, SurrogateSimulator)
    assert copy._model is None
    np.testing.assert_array_equal(copy.run(inside, keys), out)


def test_a_plain_run_keeps_its_streams() -> None:
    """``surrogate`` is appended: the other streams of every run are unchanged (§5)."""
    assert STREAM_NAMES[:5] == ("prior", "perturbations", "analysis", "localization", "chop")
    assert STREAM_NAMES[-1] == "surrogate"
    streams = RandomStreams.from_seed(2026)
    old = np.random.SeedSequence(2026).spawn(5)
    for name, child in zip(STREAM_NAMES[:5], old, strict=True):
        expected = np.random.default_rng(child).bit_generator.state
        assert streams[name].bit_generator.state == expected


# ------------------------------------------------------------------- older run directories


def test_a_version_5_run_directory_is_read_as_the_simulators(tmp_path: Path) -> None:
    """Run directories before snapshot version 6: every forecast was the simulator's (§4)."""
    config = cases.write_case(tmp_path, name="old")
    composition.start(config)
    run = _run(tmp_path, "old")
    (run / "snapshot_version").write_text("5\n")
    # A version-5 run wrote no forecast_source anywhere.
    for path in (run / "checkpoints").glob("iter_*.npz"):
        with np.load(path) as data:
            assert "forecast_source" not in data.files
    assert all("forecast_source" not in h for h in json.loads((run / "history.json").read_text()))
    results = RunResults(
        run, composition.build_layout(composition.snapshot_config(RunDirectory(run)))
    )
    iterate = results.iterate(results.final_iteration)
    assert iterate.forecast_source == ("simulator",) * iterate.member_ids.size
    assert not iterate.by_surrogate
    assert set(results.sources().values()) == {"simulator"}
    assert results.surrogate is None
    [summary] = summarize([run])
    assert summary.workflow == "history_matching"
    assert all(step.label is None for step in summary.steps)
    pytest.importorskip("matplotlib")
    figures = composition.write_report_figures(run)
    assert figures


def test_summaries_and_figures_label_surrogate_steps(tmp_path: Path, built: Path) -> None:
    """``apex summary`` and the figures say which steps were the surrogate's (§4)."""
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built, check_members=5))
    composition.start(config)
    run = _run(tmp_path, "hm")
    [summary] = summarize([run])
    assert all(step.label is not None and "surrogate" in step.label for step in summary.steps)
    simulator_rows = [m for m in summary.metrics if m.metric.startswith("simulator:")]
    assert {m.metric for m in simulator_rows} >= {"simulator:data_mismatch"}
    assert all(m.iteration == summary.selected_iteration for m in simulator_rows)
    assert any("surrogate" in note for note in summary.notes)
    # Decision 14: the forecasts' cost against the build's full runs.
    assert any(note.startswith("surrogate forecasts: 500 members") and "speed-up" in note
               for note in summary.notes)  # fmt: skip
    from typer.testing import CliRunner

    from apex.cli.main import app

    report = CliRunner().invoke(app, ["report", str(run)])
    assert report.exit_code == 0, report.output
    assert "forecast" in report.output.splitlines()[0]  # the source column
    assert "surrogate" in report.output.splitlines()[2]
    assert "posterior (iteration 4) data mismatch: simulator" in report.output
    pytest.importorskip("matplotlib")
    assert composition.write_report_figures(run)


# ------------------------------------------------------------------- variance in the errors

STD = 0.1


@pytest.mark.parametrize("validation", ["", '[validation]\ntest = { datatypes = ["WBHP:P2"] }'])
def test_variance_in_errors_adds_the_predictive_variance_to_c_d(
    tmp_path: Path, validation: str
) -> None:
    """ES-MDA with ``variance_in_errors``: the plain run with ``C_d + diag(s^2)``.

    The surrogate is exact and reports a predictive standard deviation ``s_k`` that differs
    by key (``surrogate_cases.std_of``), so the update it serves uses ``C_d + diag(s^2)`` in
    its gain and its perturbations, which is the plain run whose observation errors have
    those variances: the same draws, the same posterior, to round-off. With ``test`` data
    held out, only the history-matching rows' variances enter the update.
    """
    cases.write_case(tmp_path, name="base")
    surrogates.build(
        cases.write_build(tmp_path, PLUGIN, case="base.toml", options=f"{{ std = {STD} }}")
    )
    model = tmp_path / "runs" / "lin" / "model" / "surrogate.json"
    section = _section(model, variance_in_errors=True, check_every=0)
    n_keys = len(cases.keys_in_order())
    added = np.array([cases.std_of(STD, k, n_keys) ** 2 for k in range(n_keys)])
    result = composition.start(
        cases.write_case(tmp_path, name="hm", surrogate=section, validation=validation)
    )
    wider = composition.start(
        cases.write_case(
            tmp_path, name="plain", validation=validation, variances=cases.VARIANCE + added
        )
    )
    np.testing.assert_allclose(result.posterior, wider.posterior, rtol=0, atol=ROUND_OFF)
    narrow = composition.start(cases.write_case(tmp_path, name="narrow", validation=validation))
    assert np.abs(result.posterior - narrow.posterior).max() > 1e-6  # the variance counted
    keys = cases.keys_in_order()
    held = {k for k, (_, d) in enumerate(keys) if validation and d == "WBHP:P2"}
    with np.load(_run(tmp_path, "hm") / "checkpoints" / "iter_1.npz") as saved:
        expected = [added[k] for k in range(n_keys) if k not in held]
        np.testing.assert_allclose(saved["surrogate_variance"], expected, rtol=1e-12)


def test_variance_in_errors_is_refused_for_iterative_methods(tmp_path: Path) -> None:
    """GIES draws its perturbations once and judges trials on ``C_d``: refused (§2)."""
    cases.write_case(tmp_path, name="base")
    surrogates.build(
        cases.write_build(tmp_path, PLUGIN, case="base.toml", options=f"{{ std = {STD} }}")
    )
    model = tmp_path / "runs" / "lin" / "model" / "surrogate.json"
    config = cases.write_case(
        tmp_path, name="hm", method=cases.GIES,
        surrogate=_section(model, variance_in_errors=True),
    )  # fmt: skip
    found = validate_history_match(config)
    assert not found.valid
    assert "variance_in_errors" in found.errors[0]


def test_variance_in_errors_needs_a_predictive_variance(tmp_path: Path, built: Path) -> None:
    """A plugin without ``predict_std`` is refused (here: one that has it is accepted)."""
    config = cases.write_case(
        tmp_path, name="hm", surrogate=_section(built, variance_in_errors=True)
    )
    assert validate_history_match(config).valid, validate_history_match(config).errors
    surrogate_models.remove(PLUGIN)

    class Exact(cases.Affine):  # type: ignore[name-defined,misc]
        predict_std = None

    surrogate_models.register(PLUGIN)(Exact)
    found = validate_history_match(config)
    assert not found.valid
    assert "predict_std" in found.errors[0]


def test_erts_export_refuses_a_surrogate(tmp_path: Path, built: Path) -> None:
    from apex.integrations.ert.export import _check_exportable

    config = composition.load_config(
        cases.write_case(tmp_path, name="hm", surrogate=_section(built))
    )
    with pytest.raises(ConfigError, match="surrogate"):
        _check_exportable(config)


# ------------------------------------------------------------------- iterative fallbacks


def _in_range_bias(model: Path) -> Path:
    """Write the exact model with ``G (0.3, 0.3, 0.3)`` added: a bias a state shift explains.

    A trial judged against an accepted iterate of another source would see the bias as a
    misfit no update can remove (the reviewer's case): GIES must rescore its accepted
    iterate with the new source.
    """
    stored = read_surrogate(model)
    fitted = cases.Affine.from_arrays(stored.manifest.model, stored.arrays)
    fitted.weights = fitted.weights.copy()
    fitted.weights[:, -1] += cases.forward_matrix() @ np.full(cases.N_STATE, 0.3)
    target = model.parent.parent / "model_shifted" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        target, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
    )  # fmt: skip
    return target


LM = (
    'name = "lm_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
    "schedule = { initial = 1.0, data_mismatch_tolerance = 1e-10 }\nmax_iterations = 12"
)
GN = (
    'name = "gn_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-10 }\nmax_iterations = 12"
)
CHOP = (
    'name = "chop"\nmax_iterations = 12\n'
    'options = { inner = "gies", components = [{ name = "length_scale", mode = "data" }] }'
)
METHODS = {
    "gies": cases.GIES.replace("max_iterations = 6", "max_iterations = 12"),
    "lm_enrml": LM,
    "gn_enrml": GN,
    "chop": CHOP,
}


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("policy", ["simulator", "refit"])
def test_iterative_methods_rescore_their_accepted_iterate_after_a_change_of_source(
    tmp_path: Path, built: Path, policy: str, method: str
) -> None:
    """After a switch or a refit, trials are judged like with like.

    The shifted surrogate draws the method towards ``x - 0.3``; once the check at iteration
    3 fails, the accepted iterate is forecast again by the new source, so the next trials can
    be accepted and the run reaches the plain run's data mismatch (within 5 %: it started
    the new source from a worse iterate, and the damping path differs). Without the rescore
    every later trial is rejected against the surrogate's flattering mismatch (the review's
    case: 196 against the plain run's 25).
    """
    shifted = _in_range_bias(built)
    section = _section(shifted, check_every=3, check_members=6, on_check_failure=policy)
    config = METHODS[method]
    composition.start(cases.write_case(tmp_path, name="hm", method=config, surrogate=section))
    plain = composition.start(cases.write_case(tmp_path, name="plain", method=config))
    run = _run(tmp_path, "hm")
    [rescored] = _named(run, "surrogate_rescored")
    assert rescored["iteration"] == 3
    summary = json.loads((run / "summary.json").read_text())
    plain_final = [r.data_mismatch_mean for r in plain.history if r.accepted][-1]
    assert summary["final_data_mismatch_mean"] < 1.05 * plain_final
    history = RunDirectory(run).history_dicts()
    assert any(h["accepted"] for h in history if h["iteration"] > 3)
    assert (run / "checkpoints" / "rescored_3.npz").exists()


# ------------------------------------------------------------------- the review's minors


def _simulator_runs_in_events(run: Path) -> int:
    """Members the simulator ran, by the ``forecast_done`` events (a surrogate's excluded)."""
    return sum(
        int(e["n_ok"]) + int(e["n_failed"])
        for e in _named(run, "forecast_done")
        if e.get("label") != "surrogate"
    )


def test_a_refit_reuses_the_simulators_runs_and_counts_them_once(
    tmp_path: Path, built: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside members are run once per forecast; a rescore's go to ``accepted_<k>/``.

    With ``outside = "simulator"`` and a refit at iteration 1, the repeated forecast keeps
    the outside members' runs, the accepted iterate's outside members run in their own
    folder (not over the trial's files), and the run's count equals the runs made.
    """
    from apex.workflows.history_matching import HistoryMatchingWorkflow

    folders: list[tuple[str, str | None]] = []
    original = HistoryMatchingWorkflow._run_members

    def spy(
        self: HistoryMatchingWorkflow,
        state: np.ndarray,
        member_ids: np.ndarray,
        iteration: int,
        label: str,
        simulator: object = None,
        **options: Any,  # noqa: ANN401 (the spied method's keywords)
    ) -> Any:  # noqa: ANN401 (the spied method's outputs)
        folders.append((label, options.get("folder")))
        return original(self, state, member_ids, iteration, label, simulator, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(HistoryMatchingWorkflow, "_run_members", spy)
    section = _section(
        _biased(built), check_every=1, check_members=6, on_check_failure="refit",
        outside="simulator", max_refits=1,
    )  # fmt: skip
    method = cases.GIES.replace("max_iterations = 6", "max_iterations = 3")
    composition.start(
        cases.write_case(tmp_path, name="hm", method=method, prior_std=3.0, surrogate=section)
    )
    run = _run(tmp_path, "hm")
    repeated = [
        e for e in _named(run, "surrogate_forecast")
        if e["iteration"] == 1 and e["label"] == "ensemble"
    ]  # fmt: skip
    assert len(repeated) == 2  # the forecast, and its repetition with the refit
    assert repeated[1]["reused"] == repeated[0]["simulator"] > 0
    assert ("accepted iterate (outside)", "accepted_1") in folders
    assert all(folder in (None, "check_1", "check_2", "check_3", "accepted_1", "full_3")
               for _, folder in folders)  # fmt: skip
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert report["simulator_runs"] == _simulator_runs_in_events(run)


def test_the_planned_evaluations_bound_the_runs(tmp_path: Path, built: Path) -> None:
    """``apex validate``'s count is a run's, refits and rescores included."""
    biased = _biased(built)
    esmda = _section(biased, check_every=1, on_check_failure="refit", max_refits=1)
    config = cases.write_case(tmp_path, name="esmda", surrogate=esmda)
    plan = validate_history_match(config).plan
    assert plan is not None
    composition.start(config)
    report = json.loads((_run(tmp_path, "esmda") / "summary.json").read_text())["surrogate"]
    # ES-MDA: 5 forecasts of 100, and the refit's repeated forecast, exactly.
    assert report["evaluations"] == plan.surrogate_evaluations == 6 * 100
    gies = cases.write_case(tmp_path, name="gies", method=cases.GIES, surrogate=esmda)
    plan = validate_history_match(gies).plan
    assert plan is not None
    assert plan.surrogate_evaluations is not None
    composition.start(gies)
    report = json.loads((_run(tmp_path, "gies") / "summary.json").read_text())["surrogate"]
    assert report["rescores"]  # the refit re-forecast the accepted iterate and mean model
    assert report["evaluations"] <= plan.surrogate_evaluations
    assert plan.surrogate_evaluations == 7 * 100 + (1 + 2 * 6) + (2 * 100 + 1)


def test_the_summary_says_what_the_reduction_compares(tmp_path: Path, built: Path) -> None:
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(built, check_every=0))
    composition.start(config)
    summary = json.loads((_run(tmp_path, "hm") / "summary.json").read_text())
    assert summary["prior_source"] == "surrogate"
    assert "surrogate's prior forecast" in summary["data_mismatch_reduction_basis"]
    prior = summary["prior_data_mismatch_mean"]
    assert summary["surrogate_data_mismatch_reduction"] == pytest.approx(
        1 - summary["surrogate_final_data_mismatch_mean"] / prior
    )


def test_cross_validation_compares_like_with_like_after_a_switch(
    tmp_path: Path, built: Path
) -> None:
    """After a switch the rule starts again from the accepted iterate forecast by the
    simulator (``rescored_<k>``): it never chooses an iterate the surrogate scored before."""
    section = _section(
        _in_range_bias(built), check_every=2, check_members=6, on_check_failure="simulator"
    )
    method = cases.GIES.replace("max_iterations = 6", "max_iterations = 10")
    result = composition.start(
        cases.write_case(tmp_path, name="hm", method=method, validation=CV, surrogate=section)
    )
    run = _run(tmp_path, "hm")
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    [rescore] = report["rescores"]
    assert rescore["source"] == "simulator"
    with np.load(run / "checkpoints" / f"rescored_{rescore['iteration']}.npz") as saved:
        assert "metric:cv_normalized_data_mismatch:all" in saved.files
        assert set(saved["forecast_source"]) == {"simulator"}
    assert result.selected_iteration is not None
    assert result.selected_iteration >= rescore["accepted"]
    [summary] = summarize([run])
    assert any("cross validation (if any) compared" in note for note in summary.notes)


def test_the_posterior_reuses_the_simulators_runs(tmp_path: Path) -> None:
    """Members the simulator forecast already are not run again for the posterior."""
    cases.write_case(tmp_path, name="base")
    surrogates.build(cases.write_build(tmp_path, PLUGIN, case="base.toml", box=(-1.0, 1.0)))
    model = tmp_path / "runs" / "lin" / "model" / "surrogate.json"
    section = _section(model, check_every=0, outside="simulator")
    result = composition.start(cases.write_case(tmp_path, name="hm", surrogate=section))
    plain = composition.start(cases.write_case(tmp_path, name="plain"))
    np.testing.assert_allclose(result.posterior, plain.posterior, rtol=0, atol=ROUND_OFF)
    run = _run(tmp_path, "hm")
    with np.load(run / "checkpoints" / "iter_4.npz") as last:
        simulated = {int(m) for m, s in zip(last["member_ids"], last["forecast_source"],
                                            strict=True) if s == "simulator"}  # fmt: skip
    assert simulated  # the truth's x2 = -1 lies at the box's edge: some members outside
    with np.load(run / "checkpoints" / "full_4.npz") as full:
        assert set(full["reused_ids"].tolist()) == simulated
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert report["simulator_runs"] == _simulator_runs_in_events(run)


def test_a_plain_runs_snapshot_has_no_surrogate_key(tmp_path: Path, built: Path) -> None:
    """Frozen older code refuses unknown keys: a plain run's snapshot has none."""
    composition.start(cases.write_case(tmp_path, name="plain"), max_steps=0)
    assert "surrogate" not in json.loads((_run(tmp_path, "plain") / "config.json").read_text())
    composition.start(cases.write_case(tmp_path, name="hm", surrogate=_section(built)), max_steps=0)
    assert "surrogate" in json.loads((_run(tmp_path, "hm") / "config.json").read_text())


class _AffineWithFile(cases.Affine):  # type: ignore[name-defined,misc]
    """The affine surrogate with an option naming a file, which its fit must find."""

    def __init__(self, std: float = 0.0, scale_file: InputFile | None = None) -> None:
        super().__init__(std=std)
        self.scale_file = scale_file

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        if self.scale_file is None or not Path(self.scale_file).is_file():
            raise ValueError(f"scale_file {self.scale_file!r} does not exist")
        super().fit(snapshots, rng)


def test_a_models_file_options_are_relative_to_the_model(tmp_path: Path) -> None:
    """A model records file options relative to its folder (the user, 2026-11-02).

    Two builds of one case in different folders are the same model, and a refit in a moved
    case finds the file there, with the original's folder gone.
    """
    surrogate_models.register("test_affine_file")(_AffineWithFile)
    try:
        identities = []
        for name in ("a", "b"):
            folder = tmp_path / name
            (folder / "data").mkdir(parents=True)
            (folder / "data" / "scale.txt").write_text("1\n")
            cases.write_case(folder, name="base")
            build = cases.write_build(
                folder, "test_affine_file", case="base.toml",
                options='{ scale_file = "data/scale.txt" }',
            )  # fmt: skip
            surrogates.build(build)
            model = folder / "runs" / "lin" / "model" / "surrogate.json"
            stored = read_surrogate(model)
            assert stored.manifest.options["scale_file"] == "../../../data/scale.txt"
            identities.append(stored.identity)
        assert identities[0] == identities[1]
        moved = tmp_path / "moved"
        shutil.copytree(tmp_path / "b", moved)
        shutil.rmtree(tmp_path / "a")
        shutil.rmtree(tmp_path / "b")
        biased = _biased(moved / "runs" / "lin" / "model" / "surrogate.json")
        config = cases.write_case(
            moved, name="hm",
            surrogate=_section(biased, check_every=1, check_members=6, on_check_failure="refit"),
        )  # fmt: skip
        composition.start(config)
        refit = moved / "runs" / "hm" / "surrogates" / "refit_1" / "model"
        option = read_surrogate(refit / "surrogate.json").manifest.options["scale_file"]
        assert isinstance(option, str)
        assert not Path(option).is_absolute()
        assert (refit / option).resolve() == (moved / "data" / "scale.txt").resolve()
    finally:
        surrogate_models.remove("test_affine_file")


# ------------------------------------------------------------- coarse-corrected

COARSE_PLUGIN = "test_coarse_affine_hm"


@pytest.fixture
def coarse_built(tmp_path: Path) -> Iterator[Path]:
    """Build the exact coarse-corrected surrogate of the case; return its manifest."""
    surrogate_models.register(COARSE_PLUGIN)(cases.CoarseAffine)
    try:
        cases.write_case(tmp_path, name="base")
        extra = cases.write_coarse(tmp_path)
        result = surrogates.build(
            cases.write_build(tmp_path, COARSE_PLUGIN, case="base.toml", name="cc", extra=extra)
        )
        assert result.summary.gate is not None
        assert result.summary.gate.passed
        yield tmp_path / "runs" / "cc" / "model" / "surrogate.json"
    finally:
        surrogate_models.remove(COARSE_PLUGIN)


def _coarse_biased(model: Path) -> Path:
    """The exact coarse-corrected model with every intercept shifted by ``BIAS`` sigma."""
    stored = read_surrogate(model)
    fitted = cases.CoarseAffine.from_arrays(stored.manifest.model, stored.arrays)
    fitted.slope = fitted.slope.copy()
    fitted.slope[:, 1] += BIAS * np.sqrt(cases.VARIANCE)
    target = model.parent.parent / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        target, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
        coarse=surrogates.coarse_relative_to(
            surrogates.coarse_resolved_in(manifest.coarse, model.parent), target.parent
        ) if manifest.coarse is not None else None,
    )  # fmt: skip
    return target


def test_a_coarse_corrected_surrogate_runs_its_coarse_stack_first(
    tmp_path: Path, coarse_built: Path
) -> None:
    """Every surrogate forecast is preceded by the members' coarse runs
    on the run's executor; an exact correction gives the plain run's posterior to round-off,
    and the coarse runs are counted, planned and timed apart."""
    manifest = read_surrogate(coarse_built).manifest
    assert manifest.coarse is not None
    assert manifest.coarse.options["forward_file"] == "../../../coarse.npz"  # relative
    config = cases.write_case(
        tmp_path, name="hm", surrogate=_section(coarse_built, check_members=5)
    )
    plan = validate_history_match(config).plan
    assert plan is not None
    assert plan.max_simulator_runs == 1 * 5 + 100 + 5 * 100  # checks, posterior, coarse
    result = composition.start(config)
    plain = composition.start(cases.write_case(tmp_path, name="plain"))
    np.testing.assert_allclose(result.posterior, plain.posterior, rtol=0, atol=ROUND_OFF)
    run = _run(tmp_path, "hm")
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert report["coarse_runs"] == report["evaluations"] == 5 * 100
    coarse_done = [e for e in _named(run, "forecast_done") if e["label"].endswith("(coarse)")]
    assert len(coarse_done) == 5
    timing = surrogates_timing(run)
    assert timing["coarse"]["members"] == 5 * 100
    assert timing["simulator"]["members"] == 1 * 5 + 100


def surrogates_timing(run: Path) -> dict[str, Any]:
    from apex.workflows.surrogate_forecasts import forecast_timing

    return forecast_timing(_events(run), None)


def test_a_coarse_corrected_refit_keeps_the_checked_members_coarse_runs(
    tmp_path: Path, coarse_built: Path
) -> None:
    """A refit's snapshot set pairs the checked runs with their coarse runs, made once."""
    biased = _coarse_biased(coarse_built)
    config = cases.write_case(
        tmp_path, name="hm",
        surrogate=_section(biased, check_every=1, check_members=6, on_check_failure="refit"),
    )  # fmt: skip
    composition.start(config)
    run = _run(tmp_path, "hm")
    [refit] = _named(run, "surrogate_refit")
    assert refit["passed"]
    folder = run / "surrogates" / "refit_1"
    snapshots = load_snapshots(folder / "snapshots.npz")
    assert snapshots.coarse_outputs is not None
    assert snapshots.coarse_outputs.shape == snapshots.outputs.shape == (16, 40 + 6)
    stored = read_surrogate(folder / "model" / "surrogate.json")
    assert stored.manifest.coarse is not None
    resolved = surrogates.coarse_resolved_in(stored.manifest.coarse, folder / "model")
    assert Path(str(resolved.options["forward_file"])) == (tmp_path / "coarse.npz").resolve()
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    # The repeated forecast reuses the iteration's coarse runs: one per evaluation, less 100.
    assert report["coarse_runs"] == report["evaluations"] - 100


def test_resume_is_bit_identical_with_a_coarse_stack(tmp_path: Path, coarse_built: Path) -> None:
    section = _section(coarse_built, check_every=1, check_members=6)
    whole = cases.write_case(tmp_path / "whole", name="hm", method=cases.GIES, surrogate=section)
    composition.start(whole)
    parts = cases.write_case(tmp_path / "parts", name="hm", method=cases.GIES, surrogate=section)
    composition.start(parts, max_steps=1)
    for _ in range(10):
        if json.loads((_run(tmp_path / "parts", "hm") / "summary.json").read_text())["complete"]:
            break
        composition.resume(_run(tmp_path / "parts", "hm"), max_steps=1)
    a, b = _run(tmp_path / "whole", "hm"), _run(tmp_path / "parts", "hm")
    files = sorted(p.name for p in (a / "checkpoints").glob("*.npz"))
    assert files == sorted(p.name for p in (b / "checkpoints").glob("*.npz"))
    for name in files:
        with np.load(a / "checkpoints" / name) as left, np.load(b / "checkpoints" / name) as right:
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key], err_msg=f"{name}:{key}")
    state_a, state_b = RunDirectory(a).read_state(), RunDirectory(b).read_state()
    assert state_a["surrogate"]["evaluations"] == state_b["surrogate"]["evaluations"]


def test_a_model_of_another_coarse_stack_is_refused(tmp_path: Path, coarse_built: Path) -> None:
    config = cases.write_case(tmp_path, name="hm", surrogate=_section(coarse_built))
    np.savez(tmp_path / "coarse.npz", matrix=3 * cases.forward_matrix())
    found = validate_history_match(config)
    assert not found.valid
    assert "another coarse stack" in found.errors[0]
    assert "options_digest" in found.errors[0]


def test_a_coarse_corrected_model_without_its_stack_is_refused(
    tmp_path: Path, coarse_built: Path
) -> None:
    """Models written before step 2 record no coarse stack: refused, with the remedy."""
    stored = read_surrogate(coarse_built)
    manifest = stored.manifest
    target = coarse_built.parent.parent / "model_bare" / "surrogate.json"
    model = cases.CoarseAffine.from_arrays(manifest.model, stored.arrays)
    write_surrogate(
        target, plugin=manifest.plugin, model=model, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
    )  # fmt: skip
    found = validate_history_match(
        cases.write_case(tmp_path, name="hm", surrogate=_section(target))
    )
    assert not found.valid
    assert "records no coarse stack" in found.errors[0]


def test_a_policy_checks_the_coarse_stack_a_manifest_names(
    tmp_path: Path, coarse_built: Path
) -> None:
    """Under a [policy] the manifest's coarse stack is checked as the case's
    own simulator is: its files inside the roots."""
    from apex.domain.surrogate import CoarseSection
    from apex.services import policy
    from apex.services.settings import PolicySection

    roots = [str(tmp_path)]
    rules = PolicySection(input_roots=roots, code_roots=roots, output_roots=roots)
    config = composition.load_config(
        cases.write_case(tmp_path, name="hm", surrogate=_section(coarse_built))
    )
    assert policy.violations(config, rules, tmp_path / "root") == []
    elsewhere = tmp_path.parent / f"{tmp_path.name}-elsewhere" / "coarse.npz"
    stored = read_surrogate(coarse_built)
    manifest = stored.manifest
    target = coarse_built.parent.parent / "model_stray" / "surrogate.json"
    write_surrogate(
        target, plugin=manifest.plugin,
        model=cases.CoarseAffine.from_arrays(manifest.model, stored.arrays),
        domain=manifest.training_domain(), provenance=manifest.provenance,
        options=manifest.options, gate=manifest.gate,
        coarse=surrogates.coarse_relative_to(
            CoarseSection(options={"forward_file": str(elsewhere)}), target.parent
        ),
    )  # fmt: skip
    stray = composition.load_config(
        cases.write_case(tmp_path, name="stray", surrogate=_section(target))
    )
    problems = policy.violations(stray, rules, tmp_path / "root")
    assert any("coarse stack" in p and "input_roots" in p for p in problems), problems

    # Every confinement (the policy, tool sessions, web-UI users) sees the stack through
    # named(), only when its own rule lets it read the manifest.
    def coarse_keys(follow: Callable[[str], bool] | None) -> list[str]:
        return [
            k for _, k, _ in policy.named(stray, follow=follow) if k.startswith(policy.COARSE_STACK)
        ]

    assert coarse_keys(None) == []
    assert coarse_keys(lambda path: False) == []
    assert coarse_keys(lambda path: True) == [
        f"{policy.COARSE_STACK}simulators.linear: forward_file"
    ]

    # A model that only trained on coarse runs (trains_on_coarse) names no stack the run
    # uses: none of its files is judged (the coder's review of phase 4).
    class TrainsOnCoarse:
        trains_on_coarse = True

    surrogate_models.register("test_trains_on_coarse_policy")(TrainsOnCoarse)
    try:
        trained = coarse_built.parent.parent / "model_trained" / "surrogate.json"
        write_surrogate(
            trained, plugin="test_trains_on_coarse_policy",
            model=cases.CoarseAffine.from_arrays(manifest.model, stored.arrays),
            domain=manifest.training_domain(), provenance=manifest.provenance,
            options=manifest.options, gate=manifest.gate,
            coarse=surrogates.coarse_relative_to(
                CoarseSection(options={"forward_file": str(elsewhere)}), trained.parent
            ),
        )  # fmt: skip
        other = composition.load_config(
            cases.write_case(tmp_path, name="trained", surrogate=_section(trained))
        )
        assert not [
            k for _, k, _ in policy.named(other, follow=lambda path: True)
            if k.startswith(policy.COARSE_STACK)
        ]  # fmt: skip
    finally:
        surrogate_models.remove("test_trains_on_coarse_policy")


def test_coarse_failures_members_outside_and_retention_stay_aligned(tmp_path: Path) -> None:
    """A member whose coarse run fails is the simulator's failed member; members outside the
    domain run coarse, then by the simulator; every other member's forecast is the exact
    linear one; ``keep_simulations = "none"`` prunes the coarse folders (review, step 2)."""
    surrogate_models.register(COARSE_PLUGIN)(cases.CoarseAffine)
    try:
        cases.write_case(tmp_path, name="base")
        cases.write_coarse(tmp_path)
        extra = (
            '[coarse]\noptions = { forward_file = "coarse.npz", fail_members = [95], '
            "write_files = true }"
        )
        result = surrogates.build(
            cases.write_build(tmp_path, COARSE_PLUGIN, case="base.toml", name="cc", extra=extra)
        )
        model = tmp_path / "runs" / "cc" / "model" / "surrogate.json"
        assert result.summary.gate is not None
        assert result.summary.gate.passed
        config = cases.write_case(
            tmp_path, name="hm", prior_std=3.0,
            surrogate=_section(model, outside="simulator", check_every=0),
            extra='[output]\nkeep_simulations = "none"',
        )  # fmt: skip
        composition.start(config)
    finally:
        surrogate_models.remove(COARSE_PLUGIN)
    run = _run(tmp_path, "hm")
    results = RunResults(run, composition.build_layout(composition.load_config(config)))
    prior = results.iterate(0)
    assert 95 not in [int(m) for m in prior.member_ids]  # its coarse run failed: dropped
    assert "simulator" in prior.forecast_source
    assert "surrogate" in prior.forecast_source
    exact = cases.forward_matrix() @ prior.state["x"]
    np.testing.assert_allclose(prior.predicted, exact, rtol=0, atol=ROUND_OFF)
    assert _named(run, "surrogate_coarse_failed")
    simulations = run / "simulations"
    assert not any(simulations.glob("coarse_*/member_*")) if simulations.is_dir() else True


# -------------------------------------------------------------- early start

EARLY_PLUGIN = "test_early_linear_hm"
STOPPING = "test_linear_stops"


@pytest.fixture
def early_built(tmp_path: Path) -> Iterator[Path]:
    """Build the exact early-start surrogate of the case, on a simulator that can stop at a
    report date; return its manifest."""
    surrogate_models.register(EARLY_PLUGIN)(cases.EarlyLinear)
    simulators.register(STOPPING)(cases.StoppingLinear)
    cases.StoppingLinear.stopped.clear()
    try:
        cases.write_case(tmp_path, name="base", simulator=STOPPING)
        result = surrogates.build(cases.write_build(tmp_path, EARLY_PLUGIN, case="base.toml",
                                                    name="early"))  # fmt: skip
        assert result.summary.gate is not None
        assert result.summary.gate.passed
        assert not cases.StoppingLinear.stopped  # a build runs its design runs whole
        yield tmp_path / "runs" / "early" / "model" / "surrogate.json"
    finally:
        surrogate_models.remove(EARLY_PLUGIN)
        simulators.remove(STOPPING)


def test_an_early_start_surrogate_continues_each_members_early_run(
    tmp_path: Path, early_built: Path
) -> None:
    """Every surrogate forecast is preceded by the members' runs
    up to the model's early date (stopped there, asked only for the early keys); an exact
    continuation gives the plain run's posterior to round-off, and the early runs are
    counted, planned and timed apart."""
    config = cases.write_case(
        tmp_path, name="hm", simulator=STOPPING, surrogate=_section(early_built, check_members=5)
    )
    plan = validate_history_match(config).plan
    assert plan is not None
    assert plan.max_simulator_runs == 1 * 5 + 100 + 5 * 100  # checks, posterior, early runs
    result = composition.start(config)
    plain = composition.start(cases.write_case(tmp_path, name="plain", simulator=STOPPING))
    np.testing.assert_allclose(result.posterior, plain.posterior, rtol=0, atol=ROUND_OFF)
    early_keys = {k for k in read_model_keys(early_built) if k.time <= cases.EARLY_UNTIL}
    stopped = cases.StoppingLinear.stopped
    assert len(stopped) == 5 * 100
    assert {date for _, date, _ in stopped} == {cases.EARLY_UNTIL}
    assert all(set(keys) == early_keys for _, _, keys in stopped)
    run = _run(tmp_path, "hm")
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert report["early_until"] == cases.EARLY_UNTIL
    assert report["early_runs"] == report["evaluations"] == 5 * 100
    assert len([e for e in _named(run, "forecast_done") if e["label"].endswith("(early)")]) == 5
    timing = surrogates_timing(run)
    assert timing["early"]["members"] == 5 * 100
    assert timing["simulator"]["members"] == 1 * 5 + 100


def read_model_keys(model: Path) -> tuple[DataKey, ...]:
    stored = read_surrogate(model)
    keys: tuple[DataKey, ...] = cases.EarlyLinear.from_arrays(
        stored.manifest.model, stored.arrays
    ).keys
    return keys


def test_an_early_start_check_gates_only_the_later_keys_and_a_refit_recovers(
    tmp_path: Path, early_built: Path
) -> None:
    """A continuation shifted by 0.6 sigma after the early date fails its first check: the
    gate (sigma_ratio 0.5) scores only the later keys. Counted with the early keys, exact
    as the members' own runs, the error would be diluted to 0.6 sqrt(1/2) = 0.42 and pass.
    The refit from the checked members' full runs (a fit sets no shift) passes every later
    check."""
    stored = read_surrogate(early_built)
    fitted = cases.EarlyLinear.from_arrays(stored.manifest.model, stored.arrays)
    fitted.shift = np.array(
        [0.6 * np.sqrt(cases.VARIANCE) if k.time > cases.EARLY_UNTIL else 0.0
         for k in fitted.keys]
    )  # fmt: skip
    biased = early_built.parent.parent / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        biased, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
    )  # fmt: skip
    config = cases.write_case(
        tmp_path, name="hm", simulator=STOPPING,
        surrogate=_section(biased, check_every=1, check_members=6, on_check_failure="refit"),
    )  # fmt: skip
    composition.start(config)
    run = _run(tmp_path, "hm")
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert [c["passed"] for c in report["checks"]] == [False, True, True, True]
    [refit] = _named(run, "surrogate_refit")
    assert refit["passed"]
    snapshots = load_snapshots(run / "surrogates" / "refit_1" / "snapshots.npz")
    assert snapshots.n_runs == 40 + 6  # the checked members' full runs joined
    # The repeated forecast reuses the iteration's early runs: one per evaluation, less 100.
    assert report["early_runs"] == report["evaluations"] - 100


def test_resume_is_bit_identical_with_early_runs(tmp_path: Path, early_built: Path) -> None:
    section = _section(early_built, check_every=1, check_members=6)
    kwargs: dict[str, Any] = {"name": "hm", "method": cases.GIES, "surrogate": section,
                              "simulator": STOPPING}  # fmt: skip
    composition.start(cases.write_case(tmp_path / "whole", **kwargs))
    composition.start(cases.write_case(tmp_path / "parts", **kwargs), max_steps=1)
    for _ in range(10):
        if json.loads((_run(tmp_path / "parts", "hm") / "summary.json").read_text())["complete"]:
            break
        composition.resume(_run(tmp_path / "parts", "hm"), max_steps=1)
    a, b = _run(tmp_path / "whole", "hm"), _run(tmp_path / "parts", "hm")
    files = sorted(p.name for p in (a / "checkpoints").glob("*.npz"))
    assert files == sorted(p.name for p in (b / "checkpoints").glob("*.npz"))
    for name in files:
        with np.load(a / "checkpoints" / name) as left, np.load(b / "checkpoints" / name) as right:
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key], err_msg=f"{name}:{key}")
    state_a, state_b = RunDirectory(a).read_state(), RunDirectory(b).read_state()
    assert state_a["surrogate"]["evaluations"] == state_b["surrogate"]["evaluations"]
    assert state_a["surrogate"]["early_runs"] == state_b["surrogate"]["early_runs"] > 0


def test_early_start_refusals(tmp_path: Path, early_built: Path) -> None:
    """A run whose simulator cannot stop at a date, or whose every observation lies up to
    the early date, is refused before any run."""
    from types import SimpleNamespace

    from apex.services.surrogates import active_model, refuse_model

    model = active_model(early_built)
    early = [k for k in model.model.keys if k.time <= cases.EARLY_UNTIL]
    with pytest.raises(ConfigError, match="every observation of this run lies up to"):
        refuse_model(
            cast(Any, SimpleNamespace(surrogate=SimpleNamespace(on_check_failure="stop"))),
            model, cast(Any, SimpleNamespace(keys=early)), cast(Any, None),
        )  # fmt: skip
    # The same case on the plain linear model, which cannot stop: the build's stack differs
    # only by the simulator's name, so a model built on it is needed.
    cases.write_case(tmp_path / "plain", name="base")
    shutil.copy(tmp_path / "build_early.toml", tmp_path / "plain" / "build_early.toml")
    built = surrogates.build(tmp_path / "plain" / "build_early.toml")
    assert built.summary.gate is not None
    assert built.summary.gate.passed
    config = cases.write_case(
        tmp_path / "plain", name="hm",
        surrogate=_section(built.path / "model" / "surrogate.json"),
    )  # fmt: skip
    with pytest.raises(ConfigError, match="cannot stop at a report date"):
        composition.start(config)


def _stopping_case(
    folder: Path, name: str, options: str, *, prior_std: float = cases.PRIOR_STD,
    surrogate: str = "", extra: str = "",
) -> Path:  # fmt: skip
    """``cases.write_case`` on the stopping simulator, with more of its options."""
    config: Path = cases.write_case(folder, name=name, simulator=STOPPING, prior_std=prior_std,
                              surrogate=surrogate, extra=extra)  # fmt: skip
    text = config.read_text().replace('variables = ["x"]', f'variables = ["x"]\n{options}', 1)
    config.write_text(text)
    return config


def test_early_failures_members_outside_and_retention_stay_aligned(tmp_path: Path) -> None:
    """A member whose early run fails is the simulator's failed member; members outside the
    domain make their early run, then run whole (never stopped); the rest are the exact
    linear forecast; ``keep_simulations = "none"`` prunes the early folders (review, 2b)."""
    surrogate_models.register(EARLY_PLUGIN)(cases.EarlyLinear)
    simulators.register(STOPPING)(cases.StoppingLinear)
    cases.StoppingLinear.stopped.clear()
    options = "fail_members = [95]\nwrite_files = true"
    try:
        _stopping_case(tmp_path, "base", options)
        built = surrogates.build(cases.write_build(tmp_path, EARLY_PLUGIN, case="base.toml",
                                                   name="early"))  # fmt: skip
        assert built.summary.gate is not None
        assert built.summary.gate.passed
        model = built.path / "model" / "surrogate.json"
        config = _stopping_case(
            tmp_path, "hm", options, prior_std=3.0,
            surrogate=_section(model, outside="simulator", check_every=0),
            extra='[output]\nkeep_simulations = "none"',
        )  # fmt: skip
        composition.start(config)
    finally:
        surrogate_models.remove(EARLY_PLUGIN)
        simulators.remove(STOPPING)
    run = _run(tmp_path, "hm")
    results = RunResults(run, composition.build_layout(composition.load_config(config)))
    prior = results.iterate(0)
    assert 95 not in [int(m) for m in prior.member_ids]  # its early run failed: dropped
    assert "simulator" in prior.forecast_source
    assert "surrogate" in prior.forecast_source
    exact = cases.forward_matrix() @ prior.state["x"]
    np.testing.assert_allclose(prior.predicted, exact, rtol=0, atol=ROUND_OFF)
    assert _named(run, "surrogate_early_failed")
    outside = {int(m) for m, s in zip(prior.member_ids, prior.forecast_source, strict=True)
               if s == "simulator"}  # fmt: skip
    stopped = {member for member, _, _ in cases.StoppingLinear.stopped}
    assert outside <= stopped  # they made their early run first, then ran whole
    simulations = run / "simulations"
    assert not any(simulations.glob("early_*/member_*")) if simulations.is_dir() else True


def test_an_early_start_model_whose_later_data_are_only_test_data_is_refused(
    tmp_path: Path, early_built: Path
) -> None:
    """Its checks would gate nothing: test data are scored, never gated (review, 2b)."""
    datatypes = ", ".join(f'"{d}"' for d in cases.DATATYPES)
    validation = f'[validation]\ntest = {{ datatypes = [{datatypes}], times = ["3", "4"] }}'
    config = cases.write_case(tmp_path, name="hm", simulator=STOPPING, validation=validation,
                              surrogate=_section(early_built))  # fmt: skip
    with pytest.raises(ConfigError, match="no assimilated or cv observation lies after"):
        composition.start(config)


def test_an_iterative_run_refits_an_early_start_model_and_rescores_its_accepted_iterate(
    tmp_path: Path, early_built: Path
) -> None:
    """GIES with a continuation biased after the early date: the refit's change of source
    forecasts the accepted iterate again, its members' early runs in ``accepted_<k>_early``
    (review, 2b)."""
    stored = read_surrogate(early_built)
    fitted = cases.EarlyLinear.from_arrays(stored.manifest.model, stored.arrays)
    fitted.shift = np.array(
        [BIAS * np.sqrt(cases.VARIANCE) if k.time > cases.EARLY_UNTIL else 0.0
         for k in fitted.keys]
    )  # fmt: skip
    biased = early_built.parent.parent / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        biased, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
    )  # fmt: skip
    config = cases.write_case(
        tmp_path, name="hm", simulator=STOPPING, method=cases.GIES,
        surrogate=_section(biased, check_every=1, check_members=6, on_check_failure="refit"),
    )  # fmt: skip
    composition.start(config)
    run = _run(tmp_path, "hm")
    [refit] = _named(run, "surrogate_refit")
    assert refit["passed"]
    report = json.loads((run / "summary.json").read_text())["surrogate"]
    assert report["rescores"], "the refit should rescore the accepted iterate"
    labels = {e["label"] for e in _named(run, "forecast_done")}
    assert any(label.endswith("(early)") for label in labels)
    assert report["early_runs"] >= report["evaluations"] - 2 * 100 - 1


def test_early_runs_need_an_executor_that_carries_the_stop_date_and_a_report_date(
    tmp_path: Path,
) -> None:
    """The ssh executor's remote mode rebuilds members on its hosts, losing a stop date, and
    an early date that is not one of the deck's report dates would fail every early run:
    both refused before any run (coder's review); and a simulator that cannot stop."""
    from types import SimpleNamespace

    from apex.services.composition import refuse_early_runs

    deck = tmp_path / "deck.mako"
    deck.write_text("RUNSPEC\nSTART\n 1 'JAN' 2022 /\nSCHEDULE\nDATES\n 1 'FEB' 2022 /\n/\n")
    model = Path("m.json")
    simulator = cast(
        Any, SimpleNamespace(template_path=deck, case_dir=None, stops_at_report_dates=True)
    )
    refuse_early_runs(model, "2022-02-01", SimpleNamespace(), simulator, "ssh")
    with pytest.raises(ConfigError, match="does not carry a member's stop date"):
        refuse_early_runs(
            model, "2022-02-01", SimpleNamespace(carries_stop_at=False), simulator, "ssh"
        )
    with pytest.raises(ConfigError, match=r"not a report date of the deck deck\.mako"):
        refuse_early_runs(model, "2022-01-15", SimpleNamespace(), simulator, "ssh")
    with pytest.raises(ConfigError, match="cannot stop at a report date"):
        refuse_early_runs(model, "2022-02-01", SimpleNamespace(), cast(Any, SimpleNamespace()),
                          "ssh")  # fmt: skip
