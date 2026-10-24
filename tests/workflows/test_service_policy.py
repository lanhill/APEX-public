"""The service's [policy]: what a submission may read, execute and write.

A service runs outside its clients' sandboxes; without a policy a client could make it
write anywhere, read anything, or execute code (a Mako template, a program). These tests
use the generated linear-Gaussian case, which needs no simulator. Several reproduce
attacks found in the code review of this ADR: each must now be refused *before* the
attack's effect (a marker file its program would write, a secret it would read).
"""

import json
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

from apex.core.errors import ConfigError
from apex.domain.config import HistoryMatchConfig
from apex.domain.prediction import PredictionConfig
from apex.domain.study import StudySpec
from apex.services import composition, policy
from apex.services.api import ApexService
from apex.services.settings import PolicySection, load_settings
from apex.services.tools import ToolSession, call


class Case(Protocol):
    """What these tests use of the generated case (``apex.testing``)."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def rules(tmp_path: Path, **overrides: list[str]) -> PolicySection:
    """A complete policy: inputs anywhere under tmp_path, code in trusted/, outputs in out/."""
    lists = {
        "input_roots": [str(tmp_path)],
        "code_roots": [str(tmp_path / "trusted")],
        "output_roots": [str(tmp_path / "out")],
        **overrides,
    }
    return PolicySection(**lists)


def service(tmp_path: Path, **overrides: list[str]) -> ApexService:
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    section = rules(tmp_path, **overrides)
    lines = [
        f"{name} = [{', '.join(f'{chr(34)}{r}{chr(34)}' for r in getattr(section, name))}]"
        for name in ("input_roots", "code_roots", "output_roots")
    ]
    (root / "service.toml").write_text("[policy]\n" + "\n".join(lines) + "\n")
    return ApexService(root)


def resolved(case: Case, **run: object) -> HistoryMatchConfig:
    config = composition.load_config(case.config)
    return config.model_copy(update={"run": config.run.model_copy(update=run)})


def with_simulator(config: HistoryMatchConfig, name: str, **options: object) -> HistoryMatchConfig:
    simulator = config.simulator.model_copy(update={"name": name, "options": options})
    return config.model_copy(update={"simulator": simulator})


def outside(tmp_path: Path, name: str) -> Path:
    """A folder outside every root of ``rules(tmp_path)``."""
    return tmp_path.parent / f"{tmp_path.name}-{name}"


def evil_program(tmp_path: Path) -> tuple[Path, Path]:
    """A script a client could plant (in a folder it may write); it leaves a marker if run."""
    marker = outside(tmp_path, "PWNED")
    script = tmp_path / "zone" / "evil.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/bin/sh\necho ran >> {marker}\necho 1.0\n")
    script.chmod(0o755)
    return script, marker


def evil_config(case: Case, tmp_path: Path, script: Path, *, output: bool) -> dict[str, object]:
    config = tomllib.loads(case.config.read_text())
    template = tmp_path / "trusted" / "deck.mako"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("x")
    config["simulator"] = {
        "name": "opm_flow",
        "options": {"deck_template_file": str(template), "executable": str(script)},
    }
    if output:
        config["run"]["output_dir"] = str(tmp_path / "out" / "run")
    return config


# ---------------------------------------------------------------------- the rules


def test_no_policy_means_no_restriction(tmp_path: Path, linear_case: CaseFactory) -> None:
    config = resolved(linear_case(), output_dir=tmp_path / "anywhere")
    assert policy.violations(config, PolicySection(), tmp_path / "root") == []


def test_a_partial_policy_is_refused(tmp_path: Path) -> None:
    # Review: with code_roots empty, templates, programs and arguments went unchecked.
    (tmp_path / "service.toml").write_text(
        f'[policy]\ninput_roots = ["{tmp_path}"]\noutput_roots = ["{tmp_path}"]\n'
    )
    with pytest.raises(ConfigError, match="set all three"):
        load_settings(tmp_path)


def test_policy_roots_must_be_absolute_and_plain(tmp_path: Path) -> None:
    for root, message in (("relative/dir", "absolute"), ("/a/../b", "must not contain")):
        (tmp_path / "service.toml").write_text(f'[policy]\noutput_roots = ["{root}"]\n')
        with pytest.raises(ConfigError, match=message):
            load_settings(tmp_path)


def test_inputs_outside_the_roots_are_refused(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case()
    section = rules(tmp_path, input_roots=[str(tmp_path / "elsewhere")])
    problems = policy.violations(resolved(case), section, tmp_path / "root")
    assert any("outside [policy] input_roots" in p for p in problems)


def test_outputs_must_lie_under_the_output_roots(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case()
    section = rules(tmp_path, output_roots=[str(tmp_path / "cases" / "*" / "runs" / "planner")])
    root = tmp_path / "root"
    zone = tmp_path / "cases" / "c1" / "runs"
    assert (
        policy.violations(resolved(case, output_dir=zone / "planner" / "exp"), section, root) == []
    )
    assert policy.violations(resolved(case, output_dir=zone / "gies"), section, root)
    # Of the service root, only runs/ and studies/ (review: the root itself and jobs/ passed).
    assert policy.violations(resolved(case, output_dir=root / "runs" / "x"), section, root) == []
    assert policy.violations(resolved(case, output_dir=root), section, root)
    assert policy.violations(resolved(case, output_dir=root / "jobs" / "j1"), section, root)


def test_a_symlink_cannot_lead_out_of_a_root(tmp_path: Path, linear_case: CaseFactory) -> None:
    zone = tmp_path / "out"
    zone.mkdir()
    (zone / "escape").symlink_to(outside(tmp_path, "elsewhere"))
    config = resolved(linear_case(), output_dir=zone / "escape" / "run")
    assert policy.violations(config, rules(tmp_path), tmp_path / "root")


def test_templates_programs_and_arguments(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    section = rules(tmp_path)
    root = tmp_path / "root"
    base = resolved(linear_case(), output_dir=tmp_path / "out" / "r")
    ok = with_simulator(base, "opm_flow", deck_template_file=str(trusted / "deck.mako"))
    assert policy.violations(ok, section, root) == []
    stray = with_simulator(base, "opm_flow", deck_template_file=str(tmp_path / "x.mako"))
    assert "templates are code" in policy.violations(stray, section, root)[0]

    def program(executable: str) -> list[str]:
        config = with_simulator(
            ok, "opm_flow", deck_template_file=str(trusted / "d.mako"), executable=executable
        )
        return policy.violations(config, section, root)

    assert program(str(trusted / "flow.sh")) == []  # a path under the code roots
    assert program("python3")  # not in [policy] executables
    assert program(str(tmp_path / "evil.sh"))  # a path elsewhere
    # A name in executables must resolve (on PATH) under the code roots or a system folder:
    # a client-writable folder on PATH does not make its `flow` trusted (review).
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "flow").write_text("#!/bin/sh\n")
    (fake / "flow").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
    assert program("flow")
    # Arguments could make an allowed program run a script.
    args = with_simulator(
        ok, "eclipse", deck_template_file=str(trusted / "d.mako"), eclrun_args=["-c", "x"]
    )
    assert "extra program arguments" in policy.violations(args, section, root)[0]


# ---------------------------------------------------------------------- the review's attacks


def test_no_program_runs_before_the_check(tmp_path: Path, linear_case: CaseFactory) -> None:
    # Review B1: without an output_dir, resolve() computed the run id, which runs the
    # simulator's `executable --version`, before the policy was checked.
    case = linear_case()
    script, marker = evil_program(tmp_path)
    svc = service(tmp_path)
    for output in (False, True):
        config = evil_config(case, tmp_path, script, output=output)
        with pytest.raises(policy.PolicyViolation):
            svc.submit_run(config, base_dir=case.directory)
        assert not svc.validate(config, base_dir=case.directory).valid
    assert not marker.exists()  # the planted program never ran


def test_files_are_not_opened_outside_the_roots(tmp_path: Path) -> None:
    # Review M1: config_file was read before any check, and its content came back.
    secret = outside(tmp_path, "secret") / "creds.toml"
    secret.parent.mkdir()
    secret.write_text('api_key = "sk-TOPSECRET"\n')
    svc = service(tmp_path)
    validation = svc.validate(secret)
    assert not validation.valid
    assert "outside [policy] input_roots" in validation.errors[0]
    assert "TOPSECRET" not in " ".join(validation.errors)
    with pytest.raises(policy.PolicyViolation) as refused:
        svc.submit_run(secret)
    assert "TOPSECRET" not in str(refused.value)


@pytest.mark.parametrize(
    ("tool", "argument"),
    [
        ("validate", "config_file"),
        ("submit_run", "config_file"),
        ("submit_study", "study_file"),
        ("submit_prediction", "prediction_file"),
    ],
)
def test_the_tools_do_not_open_files_outside_the_roots(
    tmp_path: Path, tool: str, argument: str
) -> None:
    # The tools (and so MCP, served at /mcp) parsed a file before the policy: its values
    # came back in the validation errors, and "not found" told whether a file exists.
    secret = outside(tmp_path, "secret") / "creds.toml"
    secret.parent.mkdir()
    secret.write_text('[run]\nname = "x"\napi_key = "sk-TOPSECRET"\n[case]\nkind = "real"\n')
    session = ToolSession(service(tmp_path), client="test")
    arguments = {argument: str(secret), "approval_token": "t"}
    if tool == "validate":
        del arguments["approval_token"]
    refused = call(tool, arguments, session)
    missing = call(tool, {**arguments, argument: str(secret.parent / "missing.toml")}, session)
    for result in (refused, missing):
        assert not result["ok"]
        assert "outside [policy] input_roots" in result["error"]
    assert "TOPSECRET" not in json.dumps(refused)
    assert "not found" not in json.dumps(missing)  # no oracle for files outside the roots


@pytest.mark.parametrize("tool", ["validate", "submit_study"])
@pytest.mark.parametrize("present", [True, False])
def test_the_tools_do_not_open_a_study_case_outside_the_roots(
    tmp_path: Path, tool: str, present: bool
) -> None:
    # Code review of the tools' fix: the study file was checked, but not the cases it names.
    case = outside(tmp_path, "secret") / "creds.toml"
    case.parent.mkdir()
    if present:
        case.write_text('[run]\nname = "x"\napi_key = "sk-TOPSECRET"\n[case]\nkind = "real"\n')
    study = tmp_path / "study.toml"
    study.write_text(
        f'[study]\nname = "s"\ncases = {{ c = "{case}" }}\nvariants = [{{ name = "v" }}]\n'
    )
    session = ToolSession(service(tmp_path), client="test")
    arguments = {"config_file": str(study)} if tool == "validate" else {
        "study_file": str(study), "approval_token": "t"
    }  # fmt: skip
    result = call(tool, arguments, session)
    assert not result["ok"]
    assert "outside [policy] input_roots" in result["error"]
    assert "TOPSECRET" not in json.dumps(result)
    assert "No such file" not in json.dumps(result)


@pytest.mark.parametrize("name", ["../../x", "..", "a/b", "/etc", ""])
def test_a_study_name_is_one_path_segment(tmp_path: Path, name: str) -> None:
    # Code review: compare_study (a tool, and an MCP resource) took "../../x" and opened
    # <anywhere>/study.json; its error named the service root.
    planted = tmp_path / "x"
    planted.mkdir()
    (planted / "study.json").write_text('{"job": "j"}')
    svc = service(tmp_path)
    (svc.root / "studies").mkdir()
    with pytest.raises(ConfigError) as refused:
        svc.compare(name)
    assert str(svc.root) not in str(refused.value)
    result = call("compare_study", {"study": name}, ToolSession(svc, client="test"))
    assert not result["ok"]


def test_the_tools_open_files_inside_the_roots(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case()
    session = ToolSession(service(tmp_path, output_roots=[str(tmp_path)]), client="test")
    result = call("validate", {"config_file": str(case.config)}, session)
    assert result["ok"], result
    assert "approval_token" in result["result"]


def test_a_study_case_file_outside_the_roots_is_not_opened(tmp_path: Path) -> None:
    case_file = outside(tmp_path, "study") / "case.toml"
    case_file.parent.mkdir()
    case_file.write_text("secret = 1\n")
    study = {"study": {"name": "s", "cases": {"c": str(case_file)}, "variants": [{"name": "v"}]}}
    with pytest.raises(policy.PolicyViolation):
        service(tmp_path).validate_study(StudySpec.model_validate(study), base_dir=tmp_path)


def _planted_run(tmp_path: Path, case: Case, where: Path, script: Path) -> Path:
    """A run directory with a snapshot naming the client's own template and program."""
    config = with_simulator(
        composition.load_config(case.config),
        "opm_flow",
        deck_template_file=str(tmp_path / "zone" / "x.mako"),
        executable=str(script),
    )
    where.mkdir(parents=True)
    (where / "config.json").write_text(config.model_dump_json())
    (where / "snapshot_version").write_text("2\n")
    return where


def test_predictions_use_only_runs_the_service_wrote(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    # Review B2: a prediction ran the simulator of a snapshot the client planted.
    case = linear_case()
    script, marker = evil_program(tmp_path)
    svc = service(tmp_path)

    def predict(run_dir: Path) -> None:
        config = PredictionConfig.model_validate(
            {
                "prediction": {
                    "name": "p",
                    "run_dir": str(run_dir),
                    "output_dir": str(tmp_path / "out" / "pred"),
                    "outputs": {"vectors": ["y"]},
                }
            }
        )
        svc.submit_prediction(config)

    # A run folder outside the output roots is not a run the service wrote.
    with pytest.raises(policy.PolicyViolation, match="a run used as a source"):
        predict(_planted_run(tmp_path, case, tmp_path / "zone" / "fake", script))
    # Even inside them, what would actually run (snapshot + override) is checked.
    with pytest.raises(policy.PolicyViolation):
        predict(_planted_run(tmp_path, case, tmp_path / "out" / "fake", script))
    assert not marker.exists()


def test_the_worker_rechecks_at_the_time_of_use(tmp_path: Path, linear_case: CaseFactory) -> None:
    # Review B3 and M2: a snapshot changed, or a path swapped for a symlink, after submission.
    case = linear_case()
    script, _ = evil_program(tmp_path)
    section = rules(tmp_path)
    root = tmp_path / "root"
    run_dir = _planted_run(tmp_path, case, tmp_path / "out" / "r1", script)
    with pytest.raises(policy.PolicyViolation):
        policy.check_job("resume", {"run_dir": str(run_dir)}, section, root)
    good = resolved(case, output_dir=tmp_path / "out" / "r2")
    policy.check_job("run", {"config": good.model_dump_json()}, section, root)  # passes
    (tmp_path / "out" / "r2").symlink_to(outside(tmp_path, "elsewhere"))  # swapped in later
    with pytest.raises(policy.PolicyViolation):
        policy.check_job("run", {"config": good.model_dump_json()}, section, root)


def test_the_service_refuses_and_validate_explains(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case()
    svc = service(tmp_path)
    forbidden = outside(tmp_path, "forbidden")
    config = resolved(case, output_dir=forbidden / "run")
    validation = svc.validate(config)
    assert not validation.valid
    assert "outside [policy] output_roots" in validation.errors[0]
    with pytest.raises(ConfigError, match="refused by the service's \\[policy\\]"):
        svc.submit_run(config)
    assert not forbidden.exists()  # nothing was written
    assert svc.validate(resolved(case, output_dir=tmp_path / "out" / "run")).valid


def test_a_relative_study_case_is_opened_where_it_was_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review, delta round 1 (M1a): the check used <base_dir>/case, but the case was then
    # opened relative to the service's working directory, and its content leaked.
    elsewhere = outside(tmp_path, "cwd")
    elsewhere.mkdir()
    (elsewhere / "creds.toml").write_text('api_key = "sk-TOPSECRET"\n')
    monkeypatch.chdir(elsewhere)
    base = tmp_path / "studies"
    base.mkdir()
    study = {"study": {"name": "s", "cases": {"c": "creds.toml"}, "variants": [{"name": "v"}]}}
    svc = service(tmp_path)
    try:
        validation = svc.validate_study(StudySpec.model_validate(study), base_dir=base)
        text = " ".join(validation.errors)
    except ConfigError as error:
        text = str(error)
    assert "TOPSECRET" not in text


def test_a_studys_predictions_are_checked_before_any_cell_is_queued(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A client's review (2026-10-01): a refused prediction was found only after the cells were
    queued (and rolled back). It now stops the study when it is planned."""
    case = linear_case()
    svc = service(tmp_path)
    text = case.config.read_text().replace(
        'name = "case"', f'name = "case"\noutput_dir = "{tmp_path / "out" / "run"}"'
    )
    case.config.write_text(text)
    study = case.directory / "study.toml"
    study.write_text(
        '[study]\nname = "s"\n[study.cases]\nc = "config.toml"\n'
        '[[study.variants]]\nname = "v"\n'
        "[[study.predictions]]\n"
        f'prediction = {{ name = "p", output_dir = "{outside(tmp_path, "pred")}", '
        'outputs = { vectors = ["y"] } }\n'
    )

    validation = svc.validate_study(study)

    assert not validation.valid
    assert any("prediction 'p'" in e and "[policy]" in e for e in validation.errors)
    with pytest.raises(ConfigError):
        svc.submit_study(study)
    assert svc.jobs.all_records() == []  # nothing was queued


# ---------------------------------------------------------------------- declared options


def _holds_text(annotation: object) -> bool:
    """Whether an option's type can hold a string (string Literals are text by type)."""
    import typing

    if annotation in (str, Path, typing.Any, object):
        return True
    import numpy.typing

    if annotation is numpy.typing.ArrayLike:
        # Numeric arrays: the alias admits strings, the options never take them. An option
        # that names a file must be its own ``*_file`` option (e.g. mean_file), never this.
        return False
    if isinstance(annotation, typing.TypeAliasType):  # ``type X = ...``: look through it
        return _holds_text(annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return False
    if origin is typing.Annotated:
        return _holds_text(typing.get_args(annotation)[0])
    return any(
        _holds_text(a) for a in typing.get_args(annotation)
        if a is not type(None) and not isinstance(a, int | float | str | bool)
    )  # fmt: skip


def test_every_plugin_declares_what_its_string_options_name() -> None:
    """The guard for every plugin to come: a string option without a role would be refused
    under a policy, so a new plugin must declare it (core.options), and an input or code
    option must be named *_file, *_dir or *_path, as the composition resolves paths by
    name (relative to the configuration)."""
    # Plugins that register themselves outside the composition (the ERT interface's
    # prior generator), so that the check never depends on the test order.
    import apex.integrations.ert.forward_init  # noqa: F401
    from apex.core.options import role_of
    from apex.services.composition import PATH_OPTION_SUFFIXES
    from apex.services.plugins import REGISTRIES, option_model

    missing, unresolved = [], []
    for registry, entries in REGISTRIES.items():
        for name in entries.names():
            for field, info in option_model(registry, name).model_fields.items():
                annotation = info.rebuild_annotation()
                role = role_of(annotation)
                if _holds_text(annotation) and role is None:
                    missing.append(f"{registry}.{name}.{field}")
                if (
                    role is not None
                    and role.kind in ("input", "code")
                    and not field.endswith(PATH_OPTION_SUFFIXES)
                ):
                    unresolved.append(f"{registry}.{name}.{field}")
    assert missing == []
    assert unresolved == []


def test_an_undeclared_string_option_is_refused(tmp_path: Path, linear_case: CaseFactory) -> None:
    from apex.core.plugins import simulators

    class Shell:
        """A plugin whose option names a program, but not by any rule's suffix."""

        def __init__(self, command_line: str = "") -> None:
            self.command_line = command_line

    simulators.register("test_shell")(Shell)
    try:
        base = resolved(linear_case(), output_dir=tmp_path / "out" / "r")
        config = with_simulator(base, "test_shell", command_line="sh -c 'rm -rf ~'")
        problems = policy.violations(config, rules(tmp_path), tmp_path / "root")
    finally:
        simulators._factories.pop("test_shell")
    assert any("does not declare what this option names" in p for p in problems)
    assert any("simulators.test_shell: command_line" in p for p in problems)


def test_a_levels_own_deck_template_is_code(tmp_path: Path, linear_case: CaseFactory) -> None:
    trusted = tmp_path / "trusted"
    section = rules(tmp_path)
    base = resolved(linear_case(), output_dir=tmp_path / "out" / "r")

    def levels(**level: object) -> list[str]:
        config = with_simulator(
            base, "opm_flow", deck_template_file=str(trusted / "d.mako"),
            levels=[{"first": 0, "size": 5, **level}],
        )  # fmt: skip
        return policy.violations(config, section, tmp_path / "root")

    assert levels(deck_template_file=str(trusted / "coarse.mako")) == []
    assert "templates are code" in levels(deck_template_file=str(tmp_path / "c.mako"))[0]
    assert "does not declare" in levels(script="x")[0]  # an unknown key holding a string


def test_composed_options_are_checked_and_accepted(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """actnum_file and its kind (read by the composition, not passed to the plugin) were
    refused as unknown by the service's validation, though `apex run` accepted them."""
    import numpy as np

    from apex.domain.config import MetricSection

    case = linear_case(ensemble_size=10)
    np.savez(case.directory / "actnum.npz", actnum=np.array([1, 0, 1]))
    with case.config.open("a") as handle:
        handle.write('\n[[metrics]]\nname = "rmse"\noptions = { actnum_file = "actnum.npz" }\n')

    validation = ApexService(tmp_path / "plain").validate(case.config)

    assert validation.valid, validation.errors
    base = resolved(case, output_dir=tmp_path / "out" / "r")
    far = outside(tmp_path, "actnum.npz")
    config = base.model_copy(update={"metrics": [
        MetricSection(name="rmse", options={"actnum_file": str(far)})]})  # fmt: skip
    problems = policy.violations(config, rules(tmp_path), tmp_path / "root")
    assert any("actnum_file" in p and "input_roots" in p for p in problems)


def test_a_predictions_simulator_changes_are_judged_by_the_runs_simulator(
    tmp_path: Path,
) -> None:
    """Review of 2026-10-05: without a simulator name, a prediction's options were judged by
    what every simulator agrees on, so an option two simulators declare differently was
    refused as undeclared; the run's own simulator (its saved configuration) decides."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"simulator": {"name": "opm_flow"}}))
    config = PredictionConfig.model_validate({
        "prediction": {"name": "p", "run_dir": str(run), "outputs": {"vectors": ["y"]}},
        "simulator": {"options": {"executable": "flow"}},
    })  # fmt: skip
    keys = [key for _, key, _ in policy.named(config)]
    assert "simulators.opm_flow: executable" in keys
    assert not any(kind == "undeclared" for kind, _, _ in policy.named(config))
    (run / "config.json").unlink()  # unreadable: as before, every simulator must agree
    assert any(key.startswith("simulators.?") for _, key, _ in policy.named(config))


@pytest.mark.parametrize(
    ("registry", "name", "options"),
    [
        ("optimizers", "pso", {"topology": "ring", "bounds_handling": "clip", "init": "lhs"}),
        ("optimizers", "ga", {"constraint_handling": "epf", "epf": {"r": 100.0}}),
        ("optimizers", "enopt", {"epf": {"r": 1.0}}),
    ],
)
def test_options_typed_by_type_aliases_are_declared(
    registry: str, name: str, options: dict[str, object]
) -> None:
    """Review of 2026-10-06: ``type X = Literal[...]`` (and ``type EpfOptions = Annotated[...]``)
    hid the declaration, so a policy or a web-UI user was refused these options as
    undeclared."""
    kinds = [kind for kind, _, _ in policy._plugin_items(registry, name, options)]
    assert "undeclared" not in kinds
