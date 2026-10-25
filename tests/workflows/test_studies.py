"""Studies: expansion, budget, approval, jobs, predictions and comparisons."""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.stats import wilcoxon

from apex.core.errors import ConfigError
from apex.domain.study import StudySpec
from apex.services import composition
from apex.services.api import ApexService
from apex.services.study import expand, load_study, replicate_seeds

TIMEOUT = 300.0


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray


CaseFactory = Callable[..., Case]


def _study(case: Case, *, replicates: int = 2, extra: str = "", variants: str = "") -> Path:
    path = case.directory / "study.toml"
    path.write_text(
        f"""
[study]
name = "inflation"
seed = 11
replicates = {replicates}
metrics = ["normalized_data_mismatch:all", "rmse:all"]
{extra}

[study.cases]
linear = "config.toml"

[[study.variants]]
name = "four_steps"

[[study.variants]]
name = "two_steps"
patch = {{ method = {{ options = {{ inflation = [2, 2] }} }} }}
{variants}
"""
    )
    return path


def test_a_study_expands_to_paired_cells(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    spec = load_study(_study(case, replicates=3))

    cells = expand(spec, tmp_path, case.directory)

    assert [(c.variant, c.replicate) for c in cells] == [
        (v, r) for v in ("four_steps", "two_steps") for r in range(3)
    ]
    seeds = replicate_seeds(2026, 11, 3)
    assert seeds[0] == 2026  # replicate 0 keeps the case's seed
    assert len(set(seeds)) == 3
    assert [c.config.run.seed for c in cells] == seeds * 2  # the same seeds in every variant
    assert cells[3].config.method.options["inflation"] == [2, 2]
    assert cells[0].config.run.output_dir == tmp_path / "studies/inflation/linear/four_steps/r0"


def test_comparing_data_mismatch_warns_in_plain_words(
    linear_case: CaseFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """Release tester: the warning read like a changelog (a date and an ADR number)."""
    case = linear_case(ensemble_size=20)
    path = _study(case)
    path.write_text(
        path.read_text().replace('"normalized_data_mismatch:all"', '"data_mismatch:all"')
    )

    with caplog.at_level("WARNING", logger="apex.services.study"):
        load_study(path)

    (message,) = [r.getMessage() for r in caplog.records if "data_mismatch" in r.getMessage()]
    assert "the objective the methods minimise" in message
    assert "normalized_data_mismatch" in message
    assert "ADR" not in message
    assert "2026" not in message


def test_budget_and_approval_are_checked_before_anything_runs(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    over = _study(case, extra="budget = { max_simulator_runs = 50 }")
    service = ApexService(tmp_path / "service")

    validation = service.validate_study(over)
    assert not validation.valid
    assert "over its budget of 50" in validation.errors[0]
    assert len(validation.plans) == 4
    with pytest.raises(ConfigError, match="over its budget"):
        service.submit_study(over)

    guarded = ApexService(tmp_path / "guarded", approval_threshold_runs=100)
    path = _study(case)
    with pytest.raises(ConfigError, match="need approval"):
        guarded.submit_study(path, approve=lambda plan: False)
    assert not (tmp_path / "guarded" / "jobs").exists() or not list(
        (tmp_path / "guarded" / "jobs").glob("*/job.json")
    )


def test_a_study_metric_its_cells_do_not_record_is_refused(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (#192): a study compared facies_brier on a case without facies scores,
    and would have finished with no values to compare."""
    case = linear_case(ensemble_size=20)
    path = _study(case)
    path.write_text(path.read_text().replace('"rmse:all"', '"facies_brier:all"'))
    service = ApexService(tmp_path / "service")

    validation = service.validate_study(path)

    assert not validation.valid
    assert validation.errors == [validation.errors[0]]  # once, not once per cell
    assert "study metric 'facies_brier' would have no values" in validation.errors[0]
    assert "rmse" in validation.errors[0]  # what the cells do record
    assert service.validate_study(_study(case)).valid  # rmse is recorded


def test_a_study_metric_with_an_unknown_label_or_name_is_refused(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (2026-10-24): ``rmse:xx``, a typo for the state variable ``x``,
    validated, ran, and was left out of the comparison without a word."""
    case = linear_case(ensemble_size=20)
    path = _study(case)
    path.write_text(path.read_text().replace(
        '"rmse:all"', '"rmse:xx", "rmse:x", "normalized_data_mismatch:y", '
        '"normalized_data_mismatch:z", "rmsee:all", "rmse:x:skill", '
        '"constraint_cost:all"'))  # fmt: skip
    service = ApexService(tmp_path / "service")

    validation = service.validate_study(path)

    assert not validation.valid
    refused = [e for e in validation.errors if "would have no values" not in e]
    assert refused == [
        "study metric 'rmse:xx': no 'xx' in the cases; give 'all' or one of their state "
        "variables: x",
        "study metric 'normalized_data_mismatch:z': no 'z' in the cases; give 'all' or one "
        "of their data types: y",
        refused[2],
        "study metric 'constraint_cost:all': no 'all' in the cases; give one of their labels: "
        "cost, violations",
    ]  # rmse:x:skill: a facies score's parts follow the variable
    assert refused[2].startswith("study metric 'rmsee:all': no metric 'rmsee'; ")
    assert "rmse" in refused[2].split("the metrics are ")[1].split(", ")


def test_a_comparison_names_a_metric_no_cell_recorded(tmp_path: Path) -> None:
    """A study planned before labels were checked says which metric has no values."""
    from apex.services.study import comparison_from_values

    notes = comparison_from_values(
        tmp_path, "s", cases=["c"], metrics=["rmse:x", "rmse:xx"], methods=[None],
        variants=["v"], replicates=1, values={("c", "rmse:x", None, "v"): {0: 1.0}},
        missing=[], skipped=[],
    ).notes  # fmt: skip
    assert "rmse:xx: no values in any finished cell; check the metric's name and label" in notes
    assert not any(note.startswith("rmse:x:") for note in notes)


def test_a_study_runs_its_cells_and_compares_its_variants(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    service = ApexService(tmp_path / "service", max_concurrent_jobs=3)

    handle = service.submit_study(_study(case, replicates=3))
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.3)

    assert status.state == "complete", status
    assert len(status.children) == 6
    assert set(status.children.values()) == {"complete"}
    comparison = service.compare("inflation")
    assert comparison.missing == []
    assert comparison.reference == "four_steps"
    mismatch = next(m for m in comparison.metrics if m.metric == "normalized_data_mismatch:all")
    four, two = mismatch.variants
    assert (four.n, two.n) == (3, 3)
    # Independent check: the runs' own histories.
    record = json.loads((service.root / "studies/inflation/study.json").read_text())
    values: dict[str, list[float]] = {"four_steps": [], "two_steps": []}
    for cell in record["cells"]:
        history = json.loads((Path(cell["run_dir"]) / "history.json").read_text())
        values[cell["variant"]].append(history[-1]["metrics"]["normalized_data_mismatch"]["all"])
    assert four.values == pytest.approx(values["four_steps"])
    difference = np.array(values["two_steps"]) - np.array(values["four_steps"])
    assert mismatch.differences[0].mean_difference == pytest.approx(difference.mean())
    assert mismatch.differences[0].p_value == pytest.approx(wilcoxon(difference).pvalue)
    assert any("cannot reach p < 0.05" in note for note in comparison.notes)
    rows = list(csv.DictReader((service.root / "studies/inflation/comparison.csv").open()))
    assert len(rows) == 2 * 2  # two metrics x two variants


def test_one_replicate_reuses_existing_runs_and_predicts(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    service = ApexService(tmp_path / "service")
    # The base configuration ran before, outside the study.
    before = service.submit_run(case.config)
    assert service.wait(before.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    rng = np.random.default_rng(3)
    np.savez(
        case.directory / "forward_future.npz",
        matrix=np.vstack([case.forward, rng.standard_normal((2, case.forward.shape[1]))]),
    )
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(10))
    predictions = (
        "\n[[study.predictions]]\n"
        'prediction = { name = "future", ensembles = ["posterior", "prior"], '
        'outputs = { vectors = ["y"] } }\n'
        'simulator = { options = { forward_file = "forward_future.npz", '
        f"outputs = [{outputs}] }} }}\n"
    )

    handle = service.submit_study(_study(case, replicates=1, variants=predictions))
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.3)

    assert status.state == "complete", status
    assert before.run_ids[0] in handle.run_ids  # the unchanged variant reused the run
    comparison = service.compare("inflation")
    assert comparison.notes == [
        "one replicate: values and differences only, no spread over seeds, no test"
    ]
    difference = comparison.metrics[0].differences[0]
    assert (difference.n, difference.standard_error, difference.p_value) == (1, None, None)
    record = json.loads((service.root / "studies/inflation/study.json").read_text())
    for cell in record["cells"]:
        summary = service.prediction_summary(Path(cell["run_dir"]) / "predictions" / "future")
        assert summary.times[-1] == "t9"
        assert summary.first_prediction_time == 8
    # The study's own configurations validate as ordinary runs.
    assert composition.load_config(case.config).run.seed == 2026


def test_cli_validates_runs_and_compares_a_study(tmp_path: Path, linear_case: CaseFactory) -> None:
    from typer.testing import CliRunner

    from apex.cli.main import app

    case = linear_case(ensemble_size=20)
    study = _study(case)
    root = ["--root", str(tmp_path / "service")]
    runner = CliRunner()

    validated = runner.invoke(app, ["validate", str(study), *root])
    assert validated.exit_code == 0, validated.output
    # Two seeds x (four steps: 5 forecasts + two steps: 3) x 20 members.
    assert "study: 4 runs, at most 320 simulator runs" in validated.stdout
    submitted = runner.invoke(app, ["study", "run", str(study), "--yes", "--wait", *root])
    assert submitted.exit_code == 0, submitted.output
    assert "study complete" in submitted.stdout
    status = runner.invoke(app, ["study", "status", "inflation", *root])
    assert "4 complete" in status.stdout
    compared = runner.invoke(app, ["study", "compare", "inflation", *root])
    assert compared.exit_code == 0, compared.output
    assert "two_steps - four_steps" in compared.stdout
    assert "Wilcoxon" in compared.stdout

    plugins = runner.invoke(app, ["plugins", "--json"])
    assert any(p["name"] == "gies" for p in json.loads(plugins.stdout))
    schema = runner.invoke(app, ["schema", "study"])
    assert "variants" in json.loads(schema.stdout)["$defs"]["StudySection"]["properties"]
    run_dir = tmp_path / "some_run"
    stopped = runner.invoke(app, ["stop", str(run_dir)])
    assert stopped.exit_code == 0
    assert (run_dir / "stop_requested").exists()


def test_the_budget_counts_predictions_and_patches_cannot_set_seeds(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    from apex.domain.study import VariantSection
    from apex.services.study import prediction_runs

    case = linear_case(ensemble_size=20)
    spec = load_study(_study(case, replicates=1))
    section = {
        "name": "p",
        "ensembles": ["posterior", "prior", "truth"],
        "include_mean_model": True,
        "outputs": {"vectors": ["y"]},
    }
    template = {"prediction": section}
    with_predictions = spec.model_copy(
        update={"study": spec.study.model_copy(update={"predictions": [template]})}
    )
    # Two ensembles of 20 members, the truth, and a mean model per ensemble.
    assert prediction_runs(with_predictions, 20) == 2 * 20 + 1 + 2
    service = ApexService(tmp_path / "service")
    from apex.services.study import plan_study

    plan = plan_study(service, with_predictions, case.directory)
    assert plan.max_simulator_runs == (100 + 43) + (60 + 43)

    with pytest.raises(ValueError, match=r"patches run\.seed"):
        VariantSection(name="v", patch={"run": {"seed": 3}})


@pytest.mark.parametrize(
    ("where", "name"),
    [
        ("study", "."),
        ("study", ".."),
        ("variant", "."),
        ("variant", ".."),
        ("case", ".."),
        ("case", "/abs"),
        ("case", "a/b"),
        ("case", ""),
    ],
)
def test_names_of_studies_cases_and_variants_are_one_folder(where: str, name: str) -> None:
    # Each is a path component of the cells (studies/<study>/<case>/<variant>): "." or ".."
    # would point at the service root's folders, and "/abs" would replace them.
    study = {"name": "s", "cases": {"c": "c.toml"}, "variants": [{"name": "v"}]}
    if where == "study":
        study["name"] = name
    elif where == "variant":
        study["variants"] = [{"name": name}]
    else:
        study["cases"] = {name: "c.toml"}
    with pytest.raises(ValidationError, match=f"{where} name"):
        StudySpec.model_validate({"study": study})


def test_case_names_are_ascii_like_the_other_names() -> None:
    with pytest.raises(ValidationError, match="case name"):
        StudySpec.model_validate(
            {"study": {"name": "s", "cases": {"m\u00e9": "c.toml"}, "variants": [{"name": "v"}]}}
        )


def test_comparing_a_study_stored_under_an_older_rule_is_a_clear_refusal(
    tmp_path: Path,
) -> None:
    # Code review of the names patch: compare re-validated the stored spec and let a raw
    # ValidationError out (an HTTP 500 through the API).
    service = ApexService(tmp_path / "root")
    folder = service.root / "studies" / "old"
    folder.mkdir(parents=True)
    spec = {"study": {"name": "old", "cases": {"c": "c.toml"}, "variants": [{"name": "."}]}}
    (folder / "study.json").write_text(json.dumps({"spec": spec, "cells": [], "job": "j"}))
    with pytest.raises(ConfigError, match=r"not valid now: spec\.study\.variants\.0\.name: "):
        service.compare("old")


def test_comparing_a_damaged_study_record_is_a_clear_refusal(tmp_path: Path) -> None:
    service = ApexService(tmp_path / "root")
    folder = service.root / "studies" / "broken"
    folder.mkdir(parents=True)
    (folder / "study.json").write_text(json.dumps({"spec": 3, "cells": [], "job": "j"}))
    with pytest.raises(ConfigError, match="its record is damaged"):
        service.compare("broken")


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ("[1, 2]", "its record is damaged"),
        ('{"spec": {}, "cells": 3, "job": "j"}', "its record is damaged"),
        ('{"cells": [], "job": "j"}', "its record is damaged"),
        ("{not json", "its record cannot be read"),
        ('{"spec": SPEC, "cells": [CELL, {"case": "c"}], "job": "j"}', r"cells\.1\.variant: "),
        ('{"spec": SPEC, "cells": [3], "job": "j"}', r"cells\.0: "),
    ],
)
def test_a_damaged_record_names_what_is_wrong(tmp_path: Path, record: str, message: str) -> None:
    """A client's review (2026-10-01): errors had no cell index, and some gave a KeyError or
    TypeError (an HTTP 500)."""
    service = ApexService(tmp_path / "root")
    folder = service.root / "studies" / "broken"
    folder.mkdir(parents=True)
    spec = {"study": {"name": "broken", "cases": {"c": "c.toml"}, "variants": [{"name": "v"}]}}
    cell = {"case": "c", "variant": "v", "replicate": 0, "seed": 1, "run_id": "r",
            "run_dir": str(tmp_path), "job": "j"}  # fmt: skip
    text = record.replace("SPEC", json.dumps(spec)).replace("CELL", json.dumps(cell))
    (folder / "study.json").write_text(text)

    with pytest.raises(ConfigError, match=message):
        service.compare("broken")


def test_dots_inside_a_name_are_fine() -> None:
    spec = StudySpec.model_validate(
        {"study": {"name": "s.1", "cases": {"m5.a": "c.toml"}, "variants": [{"name": "v..2"}]}}
    )
    assert spec.study.name == "s.1"


# ---------------------------------------------------------------------- methods

#: A GIES case with GIES-only settings, as 5Spots/configs/gies.toml.
GIES_CASE = (
    'name = "gies"\noptions = { max_trials_per_iteration = 4 }\n'
    'strategy = { name = "rlmmac", options = { energy = 0.99 } }\n'
    "schedule = { initial = 1.0, factor = 2.0, data_mismatch_tolerance = 1e-4 }\n"
    "max_iterations = 2"
)


def _methods_study(case: Case, *, replicates: int = 2) -> Path:
    path = case.directory / "methods.toml"
    path.write_text(
        f"""
[study]
name = "algorithms"
seed = 5
replicates = {replicates}
metrics = ["normalized_data_mismatch:all"]

[study.cases]
linear = "config.toml"

[study.methods.esmda]
name = "esmda"
options = {{ inflation = [2, 2] }}
strategy = {{ name = "approx" }}

[study.methods.gies]
name = "gies"
options = {{ max_trials_per_iteration = 4 }}
strategy = {{ name = "rlmmac", options = {{ energy = 0.99 }} }}
max_iterations = 2

[[study.variants]]
name = "plain"

[[study.variants]]
name = "adaptive"
patch = {{ method = {{ localization = {{ name = "adaptive" }} }} }}
"""
    )
    return path


def test_a_method_entry_replaces_the_cases_algorithm(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The 2026-09-30 fault: ES-MDA as a patch of a GIES case inherited GIES's options,
    RLM-MAC strategy and schedule. A method entry keeps none of them."""
    case = linear_case(method=GIES_CASE, ensemble_size=20)
    spec = load_study(_methods_study(case, replicates=2))

    cells = expand(spec, tmp_path, case.directory)

    assert [(c.method, c.variant, c.replicate) for c in cells] == [
        (m, v, r) for m in ("esmda", "gies") for v in ("plain", "adaptive") for r in range(2)
    ]
    esmda = cells[0].config.method
    assert esmda.name == "esmda"
    assert esmda.options == {"inflation": [2, 2]}
    assert esmda.strategy is not None
    assert esmda.strategy.name == "approx"
    assert esmda.schedule is None
    assert esmda.max_iterations == 10  # [method]'s default, not the case's 2
    gies = cells[4].config.method
    assert gies.strategy is not None
    assert gies.strategy.name == "rlmmac"
    assert gies.schedule is None  # the entry has none: not inherited from the case either
    assert cells[2].config.method.localization is not None  # the variant on top
    assert cells[2].config.method.localization.name == "adaptive"
    # Every method and variant shares the replicate seeds (and so the prior draws).
    assert len({c.config.run.seed for c in cells if c.replicate == 1}) == 1
    assert cells[2].config.run.output_dir == (
        tmp_path / "studies/algorithms/linear/esmda/adaptive/r0"
    )
    assert cells[2].config.run.name == "algorithms-linear-esmda-adaptive-r0"


def test_the_auxiliary_fields_and_the_algorithm_fields_cover_method() -> None:
    """Every [method] field must be classified, so a new one is not inherited by accident."""
    from apex.domain.config import ALGORITHM_FIELDS, AUXILIARY_FIELDS, MethodSection
    from apex.domain.study import MethodEntry

    assert ALGORITHM_FIELDS.isdisjoint(AUXILIARY_FIELDS)
    assert set(MethodSection.model_fields) == ALGORITHM_FIELDS | AUXILIARY_FIELDS
    # ``engine`` says where the method runs; it is not a field of [method].
    assert set(MethodEntry.model_fields) == ALGORITHM_FIELDS | {"engine"}


def test_a_variant_may_not_switch_the_algorithm() -> None:
    study = {
        "name": "s",
        "cases": {"c": "c.toml"},
        "variants": [{"name": "esmda", "patch": {"method": {"name": "esmda"}}}],
    }
    with pytest.raises(ValidationError, match=r"study\.methods"):
        StudySpec.model_validate({"study": study})
    # Tuning the same algorithm stays a variant.
    study["variants"] = [{"name": "t8", "patch": {"method": {"options": {"x": 8}}}}]
    StudySpec.model_validate({"study": study})


def test_method_names_are_one_folder() -> None:
    study = {"name": "s", "cases": {"c": "c.toml"}, "methods": {"..": {"name": "esmda"}},
             "variants": [{"name": "v"}]}  # fmt: skip
    with pytest.raises(ValidationError, match="method name"):
        StudySpec.model_validate({"study": study})


def test_a_study_compares_methods_and_variants(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(method=GIES_CASE, ensemble_size=20)
    service = ApexService(tmp_path / "service", max_concurrent_jobs=4)

    handle = service.submit_study(_methods_study(case, replicates=2))
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.3)

    assert status.state == "complete", status
    assert len(status.children) == 8
    comparison = service.compare("algorithms")
    assert (comparison.reference, comparison.reference_method) == ("plain", "esmda")
    (mismatch,) = comparison.metrics
    assert [v.label() for v in mismatch.variants] == [
        "esmda/plain", "esmda/adaptive", "gies/plain", "gies/adaptive"
    ]  # fmt: skip
    pairs = [(d.label(), d.reference_label()) for d in mismatch.differences]
    assert pairs == [
        ("esmda/adaptive", "esmda/plain"),  # the localization within each method
        ("gies/adaptive", "gies/plain"),
        ("gies/plain", "esmda/plain"),  # the algorithm with each variant
        ("gies/adaptive", "esmda/adaptive"),
    ]
    # Independent check from the runs' histories and the cells' annotations.
    record = json.loads((service.root / "studies/algorithms/study.json").read_text())
    final: dict[tuple[str, str], list[float]] = {}
    for cell in sorted(record["cells"], key=lambda c: c["replicate"]):
        history = json.loads((Path(cell["run_dir"]) / "history.json").read_text())
        value = history[-1]["metrics"]["normalized_data_mismatch"]["all"]
        final.setdefault((cell["method"], cell["variant"]), []).append(value)
    expected = np.array(final[("gies", "plain")]) - np.array(final[("esmda", "plain")])
    assert mismatch.differences[2].mean_difference == pytest.approx(expected.mean())
    # The two algorithms started from the same prior ensemble in each replicate.
    priors = {}
    for cell in record["cells"]:
        if cell["variant"] == "plain" and cell["replicate"] == 1:
            with np.load(Path(cell["run_dir"]) / "checkpoints/iter_0.npz") as data:
                priors[cell["method"]] = data["state"].copy()
    np.testing.assert_array_equal(priors["esmda"], priors["gies"])
    assert any("4 paired tests" in note for note in comparison.notes)
    rows = list(csv.DictReader((service.root / "studies/algorithms/comparison.csv").open()))
    assert {(r["method"], r["variant"]) for r in rows} == set(final)


# ---------------------------------------------------------------------- simulator, paths


def test_a_study_moves_a_deck_case_to_another_simulator_with_the_shared_options() -> None:
    from apex.services.study import switch_simulator

    eclipse = {
        "name": "eclipse",
        "options": {"deck_template_file": "/m/deck.mako", "version": "e100", "licenses": 3,
                    "timeout_s": 3600, "keep_files": False, "executable": "/opt/eclrun"},
    }  # fmt: skip

    moved = switch_simulator(eclipse, "opm_flow")

    assert moved["name"] == "opm_flow"
    assert moved["options"] == {
        "deck_template_file": "/m/deck.mako",
        "timeout_s": 3600,
        "keep_files": False,
    }  # fmt: skip  (Eclipse's version, licences and program are its own)
    copied = {"name": "eclipse", "options": {"deck_template_file": "/m/deck.mako",
                                             "case_dir": "/copy"}}  # fmt: skip
    assert switch_simulator(copied, "opm_flow")["options"]["case_dir"] == "/copy"  # its data
    with pytest.raises(ConfigError, match="only the deck simulators"):
        switch_simulator({"name": "linear", "options": {}}, "opm_flow")


def test_a_case_that_runs_no_deck_cannot_change_simulator(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    path = _study(case, extra='simulator = "opm_flow"')

    with pytest.raises(ConfigError, match="cannot replace 'linear'"):
        expand(load_study(path), tmp_path, case.directory)


def test_cases_may_be_named_as_the_workspace_shows_them(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A draft in UI/owner/drafts/<chat>/ names its case 'case/config.toml'."""
    case = linear_case(ensemble_size=20)
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    drafts = tmp_path / "UI" / "owner" / "drafts" / "chat1"
    drafts.mkdir(parents=True)
    study = drafts / "study.toml"
    study.write_text(
        '[study]\nname = "placed"\n[study.cases]\nlinear = "case/config.toml"\n'
        '[[study.variants]]\nname = "baseline"\n'
    )

    validation = ApexService(root).validate_study(study)

    assert validation.valid, validation.errors
    assert len(validation.plans) == 1
    beside = drafts / "case"  # a file beside the study still wins
    beside.mkdir()
    (beside / "config.toml").write_text(case.config.read_text())
    (beside / "observations.csv").write_text((case.directory / "observations.csv").read_text())
    for name in ("forward.npz", "truth.npz"):
        (beside / name).write_bytes((case.directory / name).read_bytes())
    spec, _ = ApexService(root).checked_study(study)
    assert spec.study.cases["linear"] == beside / "config.toml"


def test_an_eclipse_case_runs_on_opm_flow_in_a_study(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    (case.directory / "deck.mako").write_text("RUNSPEC\n")
    text = case.config.read_text()
    start, end = text.index("[simulator]"), text.index("[execution]")
    case.config.write_text(
        text[:start] + '[simulator]\nname = "eclipse"\n\n[simulator.options]\n'
        'deck_template_file = "deck.mako"\nversion = "e100"\nlicenses = 3\ntimeout_s = 600\n\n'
        + text[end:]
    )

    cells = expand(
        load_study(_study(case, replicates=1, extra='simulator = "opm_flow"')),
        tmp_path,
        case.directory,
    )

    simulator = cells[0].config.simulator
    assert simulator.name == "opm_flow"
    assert simulator.options == {
        "deck_template_file": str(case.directory / "deck.mako"), "timeout_s": 600,
    }  # fmt: skip
