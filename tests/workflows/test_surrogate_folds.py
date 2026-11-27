"""A surrogate build validated by folds (``[design] validation = "folds"``).

Every design run trains; the fitting procedure is gated by K-fold cross-validation (each
fold predicted by the plugin fitted on the other folds' runs, one gate over every run's
out-of-fold prediction), and the shipped model is fitted on every run.

The case is linear (the ``linear`` simulator, ``y = G x``, observation variance 0.04) and
the surrogates test-only: a "mean" model that predicts the training runs' mean output plus
an offset drawn from its generator, whose out-of-fold predictions and gate are computed
here by hand (an independent reference: the K-fold split, the per-fold generators and the
gate's RMS of (prediction - simulator) / sigma per datatype), and the exact affine
least-squares map of ``surrogate_cases.py``.
"""

import importlib.util
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar, Self

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError, WorkflowError
from apex.core.plugins import surrogate_models
from apex.core.surrogates import FoldAwareSurrogate, SurrogateInputs, SurrogateParts
from apex.domain.observations import DataKey
from apex.domain.surrogate import FOLDS_NOTE, SnapshotSet
from apex.services import composition, surrogates
from apex.services.summaries import summarize
from apex.storage.surrogate_directory import load_snapshots
from apex.storage.surrogate_files import read_surrogate
from apex.workflows.surrogate import fold_split


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_folds", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()

MEAN = "test_mean_folds"
AFFINE = "test_affine_folds"
SEED = 7


class _Mean:
    """The training runs' mean output plus ``jitter`` times one draw of its generator.

    Its coordinate is the inputs' projection on the training inputs' mean direction ``u``,
    and its residual what lies off it, ``|x - (u . x) u|``: a reduced-input model's, whose
    in-sample residuals understate an unseen member's.
    """

    version = "1"
    #: The number of runs of every fit, in order (all instances): what a resume refits.
    fits: ClassVar[list[int]] = []

    def __init__(self, jitter: float = 0.0) -> None:
        self.jitter = jitter
        self.keys: tuple[DataKey, ...] = ()
        self.names: tuple[str, ...] = ()
        self.mean = np.zeros(0)
        self.direction = np.zeros(0)

    def _stack(self, inputs: SurrogateInputs) -> NDArray[np.float64]:
        return np.concatenate([np.asarray(inputs[n], dtype=np.float64) for n in self.names])

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        _Mean.fits.append(snapshots.n_runs)
        self.names = snapshots.layout.names
        self.keys = snapshots.keys
        offset = self.jitter * float(rng.standard_normal())
        self.mean = np.ascontiguousarray(np.mean(snapshots.outputs, axis=-1) + offset)
        self.direction = _direction(self._stack(snapshots.inputs))

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        stacked = self._stack(inputs)
        along = self.direction @ stacked
        off = stacked - np.outer(self.direction, along)
        return along[None, :], np.linalg.norm(off, axis=0)

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        n = self._stack(inputs).shape[1]
        rows = [self.keys.index(k) for k in keys]
        return np.ascontiguousarray(np.tile(self.mean[rows][:, None], (1, n)))

    def to_arrays(self) -> SurrogateParts:
        own = {"names": list(self.names), "jitter": self.jitter,
               "keys": [[k.time, k.datatype, k.index] for k in self.keys]}  # fmt: skip
        return own, {"mean": self.mean, "direction": self.direction}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = cls(jitter=float(manifest["jitter"]))  # type: ignore[arg-type]
        model.names = tuple(manifest["names"])  # type: ignore[arg-type]
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        model.mean = np.asarray(arrays["mean"], dtype=np.float64)
        model.direction = np.asarray(arrays["direction"], dtype=np.float64)
        return model


def _direction(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    mean = np.mean(inputs, axis=-1)
    return np.ascontiguousarray(mean / np.linalg.norm(mean))


def _residuals(direction: NDArray[np.float64], inputs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.linalg.norm(inputs - np.outer(direction, direction @ inputs), axis=0)


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register(MEAN)(_Mean)
    surrogate_models.register(AFFINE)(cases.Affine)
    _Mean.fits.clear()
    try:
        yield
    finally:
        surrogate_models.remove(MEAN)
        surrogate_models.remove(AFFINE)


FOLDS = 'plan = "prior"\ntraining = 12\nvalidation = "folds"\nfolds = 4'


def write_build(
    folder: Path,
    *,
    plugin: str = MEAN,
    name: str = "folds",
    design: str = FOLDS,
    options: str = "{ jitter = 0.5 }",
) -> Path:
    """Write a build of the linear case ``hm.toml`` in ``folder``."""
    path = folder / f"build_{name}.toml"
    path.write_text(f"""
[surrogate]
name = "{name}"
model = "{plugin}"
options = {options}
seed = {SEED}

[case]
config_file = "hm.toml"

[design]
{design}
""")
    return path


def _sigma_ratios(
    predicted: NDArray[np.float64], simulator: NDArray[np.float64], keys: Sequence[DataKey]
) -> dict[str, float]:
    """The gate's measure by hand: per datatype, RMS of (prediction - simulator) / sigma."""
    normalized = (predicted - simulator) / np.sqrt(cases.VARIANCE)
    return {
        datatype: float(np.sqrt(np.mean(np.square(normalized[rows]))))
        for datatype in {k.datatype for k in keys}
        for rows in [[i for i, k in enumerate(keys) if k.datatype == datatype]]
    }


def _gate_errors(document: Mapping[str, object]) -> dict[str, float]:
    entries = document["datatypes"]
    assert isinstance(entries, list)
    return {str(d["datatype"]): float(d["error"]) for d in entries}


def test_the_gate_pools_every_runs_out_of_fold_prediction(tmp_path: Path) -> None:
    """Against a hand computation: the split, each fold's generator, the pooled gate."""
    cases.write_case(tmp_path)
    config = surrogates.load_surrogate_config(write_build(tmp_path))
    plan = surrogates.plan_build(config)
    assert plan.folds == 4
    assert plan.held_out == 0
    assert plan.simulator_runs == 12
    assert any(FOLDS_NOTE in w for w in plan.warnings)

    result = surrogates.build_config(config)
    assert result.complete
    snapshots = load_snapshots(result.path / "snapshots.npz")
    ids = list(snapshots.run_ids)
    assert sorted(ids) == list(range(12))  # every design run trains, none held out

    # The split by hand: the runs in id order, permuted by the split's seed (default the
    # build's), cut into 4 near-equal parts.
    ordered = sorted(ids)
    parts = np.array_split(np.random.default_rng(SEED).permutation(len(ordered)), 4)
    split = [sorted(ordered[int(i)] for i in part) for part in parts]
    # Each fold's model: the other folds' mean, offset by 0.5 x its own generator's draw.
    predicted = np.zeros_like(snapshots.outputs)
    by_fold = []
    for k, fold in enumerate(split):
        held = [ids.index(r) for r in fold]
        rest = [p for p in range(len(ids)) if p not in held]
        # Fold k's generator: the k-th child of the build's fourth stream ("folds").
        stream = np.random.SeedSequence(SEED).spawn(4)[3].spawn(k + 1)[k]
        offset = 0.5 * float(np.random.default_rng(stream).standard_normal())
        mean = np.mean(snapshots.outputs[:, rest], axis=-1) + offset
        predicted[:, held] = mean[:, None]
        by_fold.append(
            _sigma_ratios(predicted[:, held], snapshots.outputs[:, held], snapshots.keys)
        )
    pooled = _sigma_ratios(predicted, snapshots.outputs, snapshots.keys)

    gate = json.loads((result.path / "gate.json").read_text())
    assert gate["held_out"] == ids
    for datatype, error in _gate_errors(gate).items():
        assert error == pytest.approx(pooled[datatype], rel=1e-12)
    record = json.loads((result.path / "folds.json").read_text())
    assert record["folds"] == 4
    assert record["fold_seed"] == SEED
    assert record["note"] == FOLDS_NOTE
    assert [f["runs"] for f in record["per_fold"]] == split
    for fold, expected in zip(record["per_fold"], by_fold, strict=True):
        for datatype, error in _gate_errors(fold["gate"]).items():
            assert error == pytest.approx(expected[datatype], rel=1e-12)
    assert sorted((result.path / "folds").iterdir()) == [
        result.path / "folds" / f"fold_{k}.npz" for k in range(4)
    ]

    # Recorded with the model, its summary and the summarizer, and said plainly.
    manifest = read_surrogate(result.path / "model" / "surrogate.json").manifest
    assert manifest.validation is not None
    assert manifest.validation.note == FOLDS_NOTE
    assert manifest.gate is not None
    assert manifest.gate.held_out == ids
    assert result.summary.validation == manifest.validation
    # The domain's residual bound: the shipped model's own residuals floored by every run's
    # out-of-fold residual (under the model of the fold that left it out).
    inputs = snapshots.inputs["x"]
    floor = max(
        float(np.max(_residuals(
            _direction(inputs[:, [p for p in range(12) if ids[p] not in fold]]),
            inputs[:, [ids.index(r) for r in fold]],
        )))
        for fold in split
    )  # fmt: skip
    own = float(np.max(_residuals(_direction(inputs), inputs)))
    assert floor > own  # out of sample, the runs lie further off the model's direction
    assert manifest.domain.residual_bound == pytest.approx(max(own, floor), rel=1e-12)
    assert result.summary.held_out_runs == 0
    [summary] = summarize([result.path])
    assert FOLDS_NOTE in summary.notes
    assert [s.label for s in summary.steps] == ["training", "out-of-fold"]
    assert summary.value("folds", "all", 1) is not None
    entry = summary.value("fold_2_gate_error", "WOPR:P1", 1)
    assert entry is not None
    assert entry.mean == pytest.approx(by_fold[2]["WOPR:P1"], rel=1e-12)


def test_the_shipped_model_is_a_fit_on_every_run(tmp_path: Path) -> None:
    cases.write_case(tmp_path)
    result = surrogates.build(write_build(tmp_path))
    snapshots = load_snapshots(result.path / "snapshots.npz")
    fresh = _Mean(jitter=0.5)
    fresh.fit(snapshots, surrogates.build_streams(SEED)["surrogate"])
    stored = read_surrogate(result.path / "model" / "surrogate.json")
    with np.load(result.path / "model" / "surrogate.npz") as data:
        assert np.array_equal(data["mean"], fresh.mean)
    assert stored.manifest.fit.seed == SEED
    # Its domain is every run's inputs (none held out).
    assert stored.manifest.domain.n_points == 12


class _FoldAwareMean(_Mean):
    """``_Mean`` that records the folds it is told (core's ``FoldAwareSurrogate``)."""

    told: ClassVar[list[tuple[int, int | None]]] = []

    def set_validation_fold(self, fold: int | None) -> None:
        """Record ``(runs fitted so far, fold)``, so the shipped fit's place is known."""
        _FoldAwareMean.told.append((len(_Mean.fits), fold))


def test_a_fold_aware_model_is_told_its_fold_and_the_shipped_one_is_not(tmp_path: Path) -> None:
    """Each fold's model is told its fold before its fit; the shipped model never is.

    A plugin whose draws come from fixed integers rather than its generator would otherwise
    fit every fold alike.
    """
    cases.write_case(tmp_path)
    name = "fold_aware_mean"
    surrogate_models.register(name)(_FoldAwareMean)
    _FoldAwareMean.told.clear()
    try:
        surrogates.build(write_build(tmp_path, plugin=name))
    finally:
        surrogate_models.remove(name)
    assert isinstance(_FoldAwareMean(), FoldAwareSurrogate)
    # Folds 0..3, each told before its own fit (the k-th told after k fits); 4 + 1 fits.
    assert _FoldAwareMean.told == [(k, k) for k in range(4)]
    assert len(_Mean.fits) == 5
    assert _Mean.fits[-1] == 12  # the shipped fit, untold


def test_a_build_stopped_between_folds_resumes_without_refitting_them(tmp_path: Path) -> None:
    cases.write_case(tmp_path)
    whole = surrogates.build(write_build(tmp_path, name="whole"))
    assert _Mean.fits == [9, 9, 9, 9, 12]  # four folds of 3, then the shipped model
    path = write_build(tmp_path, name="parts")
    folds = tmp_path / "runs" / "parts" / "folds"

    def stopped() -> bool:
        return folds.is_dir() and len(list(folds.glob("fold_*.npz"))) >= 2

    _Mean.fits.clear()
    first = surrogates.build_config(surrogates.load_surrogate_config(path), stopped=stopped)
    assert not first.complete
    assert first.summary.status == "stopped"
    assert sorted(p.name for p in folds.iterdir()) == ["fold_0.npz", "fold_1.npz"]
    assert not (first.path / "model" / "surrogate.json").exists()

    _Mean.fits.clear()
    finished = surrogates.resume_build(first.path, stopped=lambda: False)
    assert finished.complete
    assert _Mean.fits == [9, 9, 12]  # folds 2 and 3, then the shipped model
    for name in ("model/surrogate.json", "model/surrogate.npz", "gate.json", "folds.json"):
        assert (finished.path / name).read_bytes() == (whole.path / name).read_bytes(), name

    # A complete build returns at once; a changed K refits on the runs made, no new run.
    _Mean.fits.clear()
    assert surrogates.build(path).summary.identity == finished.summary.identity
    assert _Mean.fits == []
    changed = path.read_text().replace("folds = 4", "folds = 3")
    path.write_text(changed)
    refitted = surrogates.build(path)
    assert refitted.complete
    assert refitted.summary.simulator_runs == 12
    assert _Mean.fits == [8, 8, 8, 12]
    assert sorted(p.name for p in folds.iterdir()) == ["fold_0.npz", "fold_1.npz", "fold_2.npz"]
    assert refitted.summary.validation is not None
    assert refitted.summary.validation.folds == 3


def test_a_split_of_a_runs_members_is_the_usual_k_fold_split(tmp_path: Path) -> None:
    """Plan ``run``: the members are the run's columns, so with every member a training run
    the folds are ``array_split(default_rng(fold_seed).permutation(n), K)``, each sorted."""
    config = cases.write_case(tmp_path, members=20)
    composition.start(config)
    run_dir = tmp_path / "runs" / "hm"
    design = (
        f'plan = "run"\nrun_dir = "{run_dir}"\niterations = [0]\ntraining = 20\n'
        'validation = "folds"\nfolds = 5\nfold_seed = 2026'
    )
    result = surrogates.build(write_build(tmp_path, plugin=AFFINE, design=design, options="{}"))
    assert result.complete
    assert result.summary.simulator_runs == 0
    permutation = np.random.default_rng(2026).permutation(20)
    expected = [sorted(int(i) for i in part) for part in np.array_split(permutation, 5)]
    assert result.summary.validation is not None
    assert [f.runs for f in result.summary.validation.per_fold] == expected
    assert result.summary.gate is not None
    assert result.summary.gate.passed  # an exact surrogate of a linear case


def test_a_model_validated_by_folds_is_checked_on_new_runs(tmp_path: Path) -> None:
    cases.write_case(tmp_path)
    design = 'plan = "prior"\ntraining = 10\nvalidation = "folds"'
    result = surrogates.build(write_build(tmp_path, plugin=AFFINE, design=design, options="{}"))
    assert result.summary.validation is not None
    assert result.summary.validation.folds == 5  # the default K
    model = result.path / "model" / "surrogate.json"
    checked = CliRunner().invoke(
        app,
        ["surrogate", "check", str(model), "--against", str(tmp_path / "hm.toml"), "--runs",
         "4", "--seed", "3", "--min-inside", "0", "--no-progress"],
    )  # fmt: skip
    assert checked.exit_code == 0, checked.output
    assert "check passed" in checked.output
    last = surrogates.last_check(model)
    assert last is not None
    assert last.passed
    assert not last.trained_on
    shown = CliRunner().invoke(
        app, ["surrogate", "build", str(tmp_path / "build_folds.toml"), "--no-progress"]
    )
    assert shown.exit_code == 0, shown.output
    assert "5-fold cross-validation" in shown.output
    assert "not itself gated" in shown.output


def test_a_build_gated_on_held_out_runs_is_unchanged(tmp_path: Path) -> None:
    """The default mode's files and fingerprints carry nothing of the folds: its
    configuration, manifest and summary have no new keys, and an explicit default is the
    same build."""
    cases.write_case(tmp_path)
    plain = 'plan = "prior"\ntraining = 8\nheld_out = 3'
    result = surrogates.build(write_build(tmp_path, name="plain", design=plain))
    assert not (result.path / "folds").exists()
    assert not (result.path / "folds.json").exists()
    for name in ("config.json", "model/surrogate.json", "summary.json"):
        text = (result.path / name).read_text()
        for key in ('"validation"', '"folds"', '"fold_seed"'):
            assert key not in text, (name, key)
    explicit = surrogates.load_surrogate_config(
        write_build(tmp_path, name="plain", design=plain + '\nvalidation = "held_out"')
    )
    config = surrogates.load_surrogate_config(tmp_path / "build_plain.toml")
    assert explicit.model_dump_json() == config.model_dump_json()
    case = surrogates._build_case(config)
    assert surrogates.fingerprint_parts(explicit, case) == surrogates.fingerprint_parts(
        config, case
    )
    assert surrogates._fit_fingerprint(explicit) == surrogates._fit_fingerprint(config)
    assert surrogates.build_config(explicit).summary.identity == result.summary.identity


def test_what_a_validation_by_folds_refuses(tmp_path: Path) -> None:
    cases.write_case(tmp_path)

    def load(design: str) -> object:
        return surrogates.load_surrogate_config(write_build(tmp_path, name="r", design=design))

    with pytest.raises(ConfigError, match="held_out must be 0"):
        load(FOLDS + "\nheld_out = 2")
    with pytest.raises(ConfigError, match="5 folds need at least 5 training runs"):
        load('plan = "prior"\ntraining = 4\nvalidation = "folds"')
    with pytest.raises(ConfigError, match="greater than or equal to 2"):
        load('plan = "prior"\ntraining = 4\nvalidation = "folds"\nfolds = 1')
    with pytest.raises(ConfigError, match="folds is for validation = 'folds'"):
        load('plan = "prior"\ntraining = 8\nheld_out = 2\nfolds = 4')
    with pytest.raises(ConfigError, match="held_out must be at least 1"):
        load('plan = "prior"\ntraining = 8')
    with pytest.raises(ConfigError, match="for plans prior, lhs, file and run"):
        load('plan = "adaptive_test"\ntraining = 8\nvalidation = "folds"\nfolds = 4')
    # Fewer successful runs than folds (failures are decided only by the runs).
    with pytest.raises(WorkflowError, match="4 folds need at least 4 successful runs, got 3"):
        fold_split([0, 1, 2], 4, 0)
    with pytest.raises(WorkflowError, match="at least 2 folds"):
        fold_split([0, 1, 2], 1, 0)


def test_the_split_partitions_the_runs_and_depends_on_its_seed_only() -> None:
    ids = [3, 9, 4, 10, 1, 7, 2]
    split = fold_split(ids, 3, 11)
    assert sorted(r for fold in split for r in fold) == sorted(ids)
    assert [len(f) for f in split] == [3, 2, 2]
    assert split == fold_split(list(reversed(ids)), 3, 11)  # the runs' order is not used
    assert split != fold_split(ids, 3, 12)


def test_the_design_configuration_rejects_mixed_modes_directly() -> None:
    from apex.domain.surrogate import DesignSection

    with pytest.raises(ValidationError, match="fold_seed is for validation = 'folds'"):
        DesignSection(training=4, held_out=1, fold_seed=3)
    section = DesignSection(training=6, validation="folds")
    assert section.n_folds == 5
    assert section.held_out == 0


def test_a_refit_in_history_matching_holds_out_new_runs_only(tmp_path: Path) -> None:
    """A model validated by folds has no held-out runs of its own: a refit
    trains on all of its build's runs and the checked members it does not hold out, and is
    gated on those it holds out (``refit_held_out`` = 0.5 of 6)."""
    from apex.storage.surrogate_files import write_surrogate

    cases.write_case(tmp_path, name="base")
    design = 'plan = "lhs"\ntraining = 30\nbox = { x = [-6.0, 6.0] }\nvalidation = "folds"'
    path = tmp_path / "build_lin.toml"
    path.write_text(
        write_build(tmp_path, plugin=AFFINE, name="lin", design=design, options="{}")
        .read_text()
        .replace("hm.toml", "base.toml")
    )
    built = surrogates.build(path)
    assert built.summary.gate is not None
    assert built.summary.gate.passed
    model = built.path / "model" / "surrogate.json"
    # Biased by 3 sigma (gate kept as passed), so that the first check fails and refits.
    stored = read_surrogate(model)
    fitted = cases.Affine.from_arrays(stored.manifest.model, stored.arrays)
    fitted.weights = fitted.weights.copy()
    fitted.weights[:, -1] += 3.0 * np.sqrt(cases.VARIANCE)
    biased = built.path / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        biased, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
        validation=manifest.validation,
    )  # fmt: skip
    section = (
        f'[surrogate]\nmodel_file = "{biased}"\ncheck_every = 1\ncheck_members = 6\n'
        'on_check_failure = "refit"'
    )
    composition.start(cases.write_case(tmp_path, name="hm", surrogate=section))
    folder = tmp_path / "runs" / "hm" / "surrogates" / "refit_1"
    refit = read_surrogate(folder / "model" / "surrogate.json").manifest
    assert refit.gate is not None
    assert refit.gate.passed
    assert len(refit.gate.held_out) == 3
    assert all(r >= 30 for r in refit.gate.held_out)  # new runs only
    assert load_snapshots(folder / "snapshots.npz").n_runs == 30 + 6


def test_the_agents_say_what_a_folds_build_gated(tmp_path: Path) -> None:
    """The monitor's facts, rules and the report state the pooled out-of-fold gate of the
    procedure, never a gate on held-out runs (rules only: no model)."""
    from apex.agents.surrogate import (
        SurrogateReportFacts,
        report_template,
        surrogate_facts,
        surrogate_rules,
    )
    from apex.services.settings import MonitorSection

    cases.write_case(tmp_path)
    result = surrogates.build(write_build(tmp_path))
    facts = surrogate_facts(result.path, None)
    assert facts.folds == 4
    assert facts.validation_note == FOLDS_NOTE
    assert facts.planned_runs == 12
    rules = surrogate_rules(facts, MonitorSection())
    report = report_template(SurrogateReportFacts(facts=facts, monitor=rules))
    gate = next(s for s in report.sections if s.heading == "Gate")
    assert "out-of-fold prediction (4 folds)" in gate.text
    assert FOLDS_NOTE in gate.text
    assert "held-out" not in gate.text
    failed = facts.model_copy(update={"gate_passed": False, "gate_failures": ["x"]})
    findings = surrogate_rules(failed, MonitorSection()).findings
    [finding] = [f for f in findings if f.code == "gate_failed"]
    assert "pooled out-of-fold gate of its fitting procedure (4 folds)" in finding.evidence


def test_the_rest_schema_keeps_the_summarys_and_plans_fields() -> None:
    """``validation`` and ``folds`` are left out when absent by a field's ``exclude_if``, so
    the JSON schemas (REST, ``/openapi.json``) still list every field."""
    from apex.domain.surrogate import SurrogateBuildPlan, SurrogateBuildSummary

    for mode in ("validation", "serialization"):
        summary = SurrogateBuildSummary.model_json_schema(mode=mode)["properties"]
        assert {"validation", "gate", "training_runs"} <= set(summary)
        assert "folds" in SurrogateBuildPlan.model_json_schema(mode=mode)["properties"]


def test_a_one_member_refit_of_a_folds_model_is_refused_before_any_run(tmp_path: Path) -> None:
    cases.write_case(tmp_path)
    built = surrogates.build(write_build(tmp_path, plugin=AFFINE, options="{}"))
    model = built.path / "model" / "surrogate.json"
    plan_lines = CliRunner().invoke(app, ["validate", str(tmp_path / "build_folds.toml")])
    assert "12 training runs, validated by 4 folds" in plan_lines.output
    section = (
        f'[surrogate]\nmodel_file = "{model}"\ncheck_every = 1\ncheck_members = 1\n'
        'on_check_failure = "refit"'
    )
    config = cases.write_case(tmp_path, name="hm1", surrogate=section)
    with pytest.raises(ConfigError, match="validated by folds"):
        composition.start(config)
    assert not (tmp_path / "runs" / "hm1" / "checkpoints").exists()
