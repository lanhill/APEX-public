"""Surrogate builds and checks as service jobs.

The case is linear and the surrogate the test-only affine map of ``surrogate_cases``
(registered here, never shipped), as in the workflow's tests. The jobs' workers run in
threads of this process (as ``test_engines.py``'s), so they see the test plugin; everything
else is the service's own path: placement, approval, the job, its stop and resume, the
summary, the check, the pinned code's refusal and the policy.

Not exercised here, by that choice: a real worker process (``worker_spawn`` and its
environment), the process executor inside a worker, and recovering a dead worker; the
service's tests of runs and optimizations cover those paths, which are kind-independent.
"""

import importlib.util
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from apex.cli.worker import run_job
from apex.core.errors import ConfigError
from apex.core.jobs import JobRecord
from apex.core.plugins import surrogate_models
from apex.services import policy, surrogates
from apex.services.api import ApexService
from apex.services.approval import ApprovalRequired
from apex.services.code import CodeError
from apex.services.jobs import LocalJobRunner
from apex.services.settings import PolicySection
from apex.storage.surrogate_directory import JOURNAL

PLUGIN = "test_affine_service"
TIMEOUT = 120.0


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_service", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()


@pytest.fixture(autouse=True)
def _registered(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    surrogate_models.register(PLUGIN)(cases.Affine)

    def spawn(self: LocalJobRunner, record: JobRecord) -> int:
        def work() -> None:
            run_job(LocalJobRunner(self.root, role="worker"), record["id"])

        threading.Thread(target=work, daemon=True).start()
        return os.getpid()

    monkeypatch.setattr(LocalJobRunner, "_spawn", spawn)
    try:
        yield
    finally:
        surrogate_models.remove(PLUGIN)


@pytest.fixture
def service(tmp_path: Path) -> ApexService:
    return ApexService(tmp_path / "service", max_concurrent_jobs=2)


def _build(folder: Path, **changes: str) -> Path:
    cases.write_case(folder, name="base")
    built: Path = cases.write_build(folder, PLUGIN, case="base.toml", **changes)
    return built


def test_a_build_job_is_placed_by_its_content_and_found_again(
    service: ApexService, tmp_path: Path
) -> None:
    config = _build(tmp_path)
    validation = service.validate_surrogate_build(config)
    assert validation.valid, validation.errors
    assert validation.surrogate is not None
    assert validation.surrogate.simulator_runs == 40

    handle = service.submit_surrogate_build(config)
    assert handle.kind == "surrogate_build"
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = service.surrogates()
    path = Path(row["path"])
    assert path.parent == service.root / "surrogates" / "lin"
    assert path.name == handle.run_ids[0]
    summary = service.surrogate_summary(path)
    assert summary.status == "complete"
    assert summary.gate is not None
    assert summary.gate.passed
    journal = (path / JOURNAL).read_text()

    again = service.submit_surrogate_build(config)
    assert again.run_ids == handle.run_ids
    assert service.wait(again.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    assert (path / JOURNAL).read_text() == journal  # nothing run again


def test_a_stopped_build_job_resumes_to_the_uninterrupted_model(
    service: ApexService, tmp_path: Path
) -> None:
    whole = surrogates.build(_build(tmp_path / "whole"))
    part = service.submit_surrogate_build(_build(tmp_path / "part"), max_runs=7)
    assert service.wait(part.id, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    resumed = service.resume(part.id)
    assert resumed.kind == "surrogate_build_resume"
    assert service.wait(resumed.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [path] = {Path(row["path"]) for row in service.surrogates()}
    for name in ("surrogate.json", "surrogate.npz"):
        assert (path / "model" / name).read_bytes() == (whole.path / "model" / name).read_bytes()


def test_a_large_build_needs_approval(service: ApexService, tmp_path: Path) -> None:
    cases.write_case(tmp_path, name="base")
    big = cases.write_build(tmp_path, PLUGIN, case="base.toml")
    big.write_text(big.read_text().replace("training = 30", "training = 30000"))
    with pytest.raises(ApprovalRequired, match="simulator runs"):
        service.submit_surrogate_build(big)


def test_a_check_job_gates_a_model_on_new_runs(service: ApexService, tmp_path: Path) -> None:
    built = surrogates.build(_build(tmp_path))
    model = built.path / "model" / "surrogate.json"
    handle = service.submit_surrogate_check(model, tmp_path / "base.toml", runs=5, seed=3)
    assert handle.kind == "surrogate_check"
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    last = surrogates.last_check(model)
    assert last is not None
    assert last.status == "complete"
    with pytest.raises(ConfigError, match="at least one run"):
        service.submit_surrogate_check(model, tmp_path / "base.toml", runs=0)


def test_a_job_pinned_to_code_without_the_kind_is_refused(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 5: refused when submitted, not in an old worker."""
    monkeypatch.setattr(
        ApexService,
        "pin_code",
        lambda self, code, run_dir=None: {"commit": "0" * 12, "python": "x"},
    )

    def refuse(commit: str, kind: str) -> None:
        raise CodeError(f"commit {commit} cannot run {kind} jobs (its worker predates them)")

    monkeypatch.setattr(service.code, "check_job_kind", refuse)
    with pytest.raises(ConfigError, match="cannot run surrogate_build jobs"):
        service.submit_surrogate_build(_build(tmp_path), code="0" * 12)
    assert service.surrogates() == []


def test_the_policy_judges_a_build_its_case_and_a_models_manifest(tmp_path: Path) -> None:
    """Decision 6: a build's files and plugins, its case's configuration
    (which it runs), and what a model's manifest names."""
    allowed = tmp_path / "allowed"
    rules = PolicySection(
        input_roots=[str(allowed)], code_roots=[str(allowed)], output_roots=[str(allowed)]
    )
    root = tmp_path / "root"
    inside = surrogates.load_surrogate_config(_build(allowed))
    assert policy.violations(inside, rules, root) == []

    outside = surrogates.load_surrogate_config(_build(tmp_path / "elsewhere"))
    problems = policy.violations(outside, rules, root)
    assert any("config_file" in p and "input_roots" in p for p in problems), problems

    # The case is judged as the configuration it is: its forward file outside the roots.
    stray = allowed / "base.toml"
    stray.write_text(
        stray.read_text().replace('"forward.npz"', f'"{tmp_path / "elsewhere" / "forward.npz"}"')
    )
    problems = policy.violations(inside, rules, root)
    assert any(p.startswith("[case]") and "input_roots" in p for p in problems), problems

    # A model's manifest: its plugin's file options and its snapshots, by their places.
    built = surrogates.build(_build(tmp_path / "model_case"))
    model = built.path / "model" / "surrogate.json"
    found = {key: value for _, key, value in policy.manifest_items(model)}
    assert found[f"{policy.MANIFEST}snapshots"] == str(built.path / "snapshots.npz")
    with pytest.raises(policy.PolicyViolation, match="input_roots"):
        policy.check_surrogate_check(
            {"model_file": str(model), "against": str(tmp_path / "model_case" / "base.toml")},
            rules, root,
        )  # fmt: skip
    # At the time of use, the worker judges the job's kind as at submission.
    with pytest.raises(policy.PolicyViolation, match="input_roots"):
        policy.check_job("surrogate_build", {"config": outside.model_dump_json()}, rules, root)


def test_the_tools_validate_submit_and_summarize_a_build(service: ApexService) -> None:
    from apex.services.tools import ToolSession, call

    # In the workspace: tool clients name files only there (release tester N3).
    folder = service.workspace().roots[0]
    path = _build(folder)
    session = ToolSession(service, client="llm-test")
    checked = call("validate", {"config_file": str(path)}, session)["result"]
    assert checked["valid"], checked
    assert checked["plan"]["simulator_runs"] == 40
    refused = call("submit_surrogate", {"surrogate_file": str(path), "approval_token": "nope"},
                   session)  # fmt: skip
    assert "error" in refused
    submitted = call(
        "submit_surrogate",
        {"surrogate_file": str(path), "approval_token": checked["approval_token"]},
        session,
    )["result"]
    assert submitted["build_id"] == checked["plan"]["build_id"]
    assert service.wait(submitted["job_id"], timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = service.surrogates()
    summary = call("surrogate_summary", {"path": row["path"]}, session)["result"]
    assert summary["status"] == "complete"
    assert summary["gate"]["passed"]


def test_the_rest_api_builds_lists_and_summarizes(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from apex.api.app import create_app
    from apex.services.auth import api_token, ui_token

    root = tmp_path / "service"
    root.mkdir()
    (tmp_path / "alpha").mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    service = ApexService(root)
    http = TestClient(create_app(root))
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    path = _build(tmp_path / "alpha")

    answer = http.post("/v1/surrogates", json={"surrogate_file": str(path)}, headers=owner)
    assert answer.status_code == 202, answer.text
    job = answer.json()["id"]
    assert service.wait(job, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = http.get("/v1/surrogates", headers=owner).json()
    assert row["job"] == job
    summary = http.get("/v1/surrogates/summary", params={"path": row["path"]}, headers=owner)
    assert summary.status_code == 200, summary.text
    assert summary.json()["status"] == "complete"

    # A guest sees none of the owner's builds, and may not use a plugin the list hides.
    (root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["alpha"]\n[plugins]\nsimulators = ["linear"]\n'
        'executors = ["*"]\n[actions]\nsubmit = true\n'
    )
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='g')}"}
    assert http.get("/v1/surrogates", headers=guest).json() == []
    hidden = http.get("/v1/surrogates/summary", params={"path": row["path"]}, headers=guest)
    assert hidden.status_code == 404
    refused = http.post("/v1/surrogates", json={"surrogate_file": str(path), "client": "ui"},
                        headers=guest)  # fmt: skip
    assert refused.status_code == 403
    assert "surrogate_models" in refused.text


def test_a_build_validates_as_a_build_by_file_document_and_text(service: ApexService) -> None:
    """The service's generic validation (and the REST route) tell a build by its [design]."""
    import tomllib

    folder = service.workspace().roots[0]
    path = _build(folder)
    text = path.read_text()
    for validation in (service.validate_text(text, folder),
                       service.validate_surrogate_build(tomllib.loads(text), folder)):  # fmt: skip
        assert validation.valid, validation.errors
        assert validation.surrogate is not None
        assert validation.surrogate.simulator_runs == 40


# ------------------------------------------------------------------ the review's points


def test_an_approval_names_the_builds_content_not_its_folder(service: ApexService) -> None:
    """A token for a small build is not spent on an edited, larger one of the same name."""
    from apex.services.tools import ToolSession, call

    folder = service.workspace().roots[0]
    path = _build(folder)
    # An explicit folder: the folder's name is the same before and after the edit.
    path.write_text(path.read_text().replace('name = "lin"', 'name = "lin"\noutput_dir = "fixed"'))
    session = ToolSession(service, client="llm-test")
    checked = call("validate", {"config_file": str(path)}, session)["result"]
    assert checked["valid"], checked
    path.write_text(path.read_text().replace("training = 30", "training = 31"))
    spent = call(
        "submit_surrogate",
        {"surrogate_file": str(path), "approval_token": checked["approval_token"]},
        session,
    )
    assert "error" in spent
    assert service.surrogates() == []


def test_a_resumed_build_keeps_its_code_pin(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume naming no code (the web UI's, Client.resume) keeps the pin."""
    commit = "abcdef123456"
    monkeypatch.setattr(ApexService, "pin_code",
                        lambda self, code, run_dir=None: None if code is None
                        else {"commit": str(code), "python": "x"})  # fmt: skip
    monkeypatch.setattr(service.code, "check_job_kind", lambda commit, kind: None)
    part = service.submit_surrogate_build(_build(tmp_path), max_runs=7, code=commit)
    assert service.wait(part.id, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    resumed = service.resume(part.id)
    assert service.jobs.record(resumed.id)["code"]["commit"] == commit


def test_a_build_started_by_the_cli_resumes_as_a_job(service: ApexService, tmp_path: Path) -> None:
    first = surrogates.build(_build(tmp_path), max_runs=7)
    assert not first.complete
    handle = service.resume_run(first.path)
    assert handle.kind == "surrogate_build_resume"
    assert service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    assert surrogates.build_summary(first.path).status == "complete"


def test_a_jobs_stop_file_stops_a_build_and_an_old_one_is_cleared(tmp_path: Path) -> None:
    """``apex stop BUILD_DIR`` (and a job's stop) write the folder's stop file: a build under
    a job reads it between waves; one left from before the start is not this start's."""
    from apex.workflows.surrogate import WAVE_WIDTHS

    path = _build(tmp_path)
    config = surrogates.load_surrogate_config(path)
    assert config.surrogate.output_dir is not None
    stop_file = config.surrogate.output_dir / surrogates.STOP_FILE
    stop_file.parent.mkdir(parents=True)
    stop_file.write_text("an old stop\n")
    calls: list[int] = []

    def asked() -> bool:
        calls.append(1)
        if len(calls) == 1:
            assert not stop_file.exists()  # cleared when the build started
        else:  # apex stop, while the first wave ran: read before the second
            stop_file.write_text("stop at the next checkpoint (job x)\n")
        return False

    built = surrogates.build_config(config, stopped=asked)
    assert not built.complete
    assert built.summary.pending == 40 - WAVE_WIDTHS  # one wave of the 30 + 10 runs


def test_the_wave_width_is_the_executors_capped_by_the_simulator() -> None:
    from apex.workflows.surrogate import parallel_width

    class Remote:
        """An executor whose capacity is other hosts' slots: it says no width."""

    class Local:
        workers = 3

    class Declared:
        workers = 3
        parallel_width = 8

    class BySimulator:
        """A width of the simulator, as the ssh executor's (its slots over the threads)."""

        def parallel_width(self, simulator: object = None) -> int:
            return 12 // int(getattr(simulator, "threads", 1))

    assert parallel_width(Remote()) is None  # type: ignore[arg-type]
    assert parallel_width(Local()) == 3  # type: ignore[arg-type]
    assert parallel_width(Declared()) == 8  # type: ignore[arg-type]
    assert parallel_width(BySimulator(), type("T", (), {"threads": 4})()) == 3  # type: ignore[arg-type]
    capped = type("Eclipse", (), {"max_parallel_runs": 2})()
    assert parallel_width(Local(), capped) == 2  # type: ignore[arg-type]


def test_a_check_writes_only_where_the_policy_lets_it_and_runs_any_case(
    service: ApexService, tmp_path: Path
) -> None:
    """A check's verdicts go beside the model: within the output roots under a policy. Its
    case may be an optimization's configuration (the CLI's controls source)."""
    import tomli_w

    from apex.domain.optimization import OptimizationConfig

    built = surrogates.build(_build(tmp_path / "case"))
    model = built.path / "model" / "surrogate.json"
    readable = [str(tmp_path)]
    # The case's own run folder may be written; the model's checks/ may not.
    rules = PolicySection(input_roots=readable, code_roots=readable,
                          output_roots=[str(tmp_path / "case" / "runs" / "base")])  # fmt: skip
    spec = {"model_file": str(model), "against": str(tmp_path / "case" / "base.toml")}
    with pytest.raises(policy.PolicyViolation, match=r"verdicts .* outside"):
        policy.check_surrogate_check(spec, rules, tmp_path / "root")
    policy.check_surrogate_check(
        spec,
        PolicySection(input_roots=readable, code_roots=readable, output_roots=readable),
        tmp_path / "root",
    )
    optimization = tmp_path / "opt.toml"
    optimization.write_text(tomli_w.dumps({
        "optimization": {"name": "toy", "seed": 3},
        "controls": {"variables": [{"name": "inj", "initial": [300.0], "lower": 10.0,
                                    "upper": 400.0}]},
        "objective": {"name": "npv", "options": {"wop": 1.0}},
        "optimizer": {"name": "enopt", "options": {"ne": 4, "maxiter": 1}},
        "simulator": {"name": "toy_reservoir"},
        "execution": {"name": "serial"},
    }))  # fmt: skip
    assert isinstance(service.surrogate_check_case(optimization), OptimizationConfig)


def test_web_ui_users_build_in_their_folder_and_check_only_their_models(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from apex.api.app import create_app
    from apex.services.auth import api_token, ui_token

    root = tmp_path / "service"
    root.mkdir()
    (tmp_path / "alpha").mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    (root / "ui_access.toml").write_text(
        f'[workspace]\nfolders = ["alpha"]\n[plugins]\nsimulators = ["linear"]\n'
        f'executors = ["*"]\nsurrogate_models = ["{PLUGIN}"]\nsurrogate_designs = ["*"]\n'
        'methods = ["*"]\npriors = ["*"]\nupdate_strategies = ["*"]\n'
        "[actions]\nsubmit = true\n"
    )
    service = ApexService(root)
    http = TestClient(create_app(root))
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='g')}"}
    path = _build(tmp_path / "alpha")

    answer = http.post("/v1/surrogates", json={"surrogate_file": str(path), "client": "ui"},
                       headers=guest)  # fmt: skip
    assert answer.status_code == 202, answer.text
    assert service.wait(answer.json()["id"], timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = http.get("/v1/surrogates", headers=guest).json()
    placed = Path(row["path"])
    assert placed.parent.parent.parent.name == "g"  # UI/guests/g/surrogates/<name>/<id>
    assert placed.parent.parent.name == "surrogates"
    elsewhere = path.read_text().replace(
        'name = "lin"', f'name = "lin"\noutput_dir = "{tmp_path / "x"}"'
    )
    path.write_text(elsewhere)
    refused = http.post("/v1/surrogates", json={"surrogate_file": str(path), "client": "ui"},
                        headers=guest)  # fmt: skip
    assert refused.status_code == 403
    assert "own folder" in refused.text

    # The owner's model, in a folder the guest may see: checking it would write a verdict
    # that decides whether the owner's runs may use it.
    owner_built = surrogates.build(_build(tmp_path / "alpha" / "owner"))
    model = owner_built.path / "model" / "surrogate.json"
    check = {"model_file": str(model), "against": str(tmp_path / "alpha" / "owner" / "base.toml"),
             "runs": 3, "client": "ui"}  # fmt: skip
    denied = http.post("/v1/surrogates:check", json=check, headers=guest)
    assert denied.status_code == 403
    assert "own folder" in denied.text
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    assert http.get("/v1/surrogates/summary", params={"path": str(placed)},
                    headers=owner).status_code == 200  # fmt: skip

    # The guest's own model against the owner's run in a visible folder: its posterior is
    # not the guest's to draw (as a prediction of it would be refused).
    from apex.services import composition

    composition.start(tmp_path / "alpha" / "base.toml", max_steps=0)
    own_model = placed / "model" / "surrogate.json"
    theirs = {"model_file": str(own_model), "against": str(tmp_path / "alpha" / "runs" / "base"),
              "runs": 3, "client": "ui"}  # fmt: skip
    assert http.post("/v1/surrogates:check", json=theirs, headers=guest).status_code == 404


def test_a_builds_identity_ignores_its_cases_runs_and_follows_its_data(
    service: ApexService, tmp_path: Path
) -> None:
    """A run of the case going or done does not move the build; its data do."""
    path = _build(tmp_path)
    first = service.resolve_surrogate_build(path).surrogate.output_dir
    runs = tmp_path / "runs" / "base" / "checkpoints"
    runs.mkdir(parents=True)
    (runs / "iter_0.npz").write_bytes(b"a run of the case")
    assert service.resolve_surrogate_build(path).surrogate.output_dir == first
    observations = tmp_path / "observations.csv"
    observations.write_text(observations.read_text() + "\n")
    assert service.resolve_surrogate_build(path).surrogate.output_dir != first


def test_a_live_build_of_the_folder_is_reused_only_for_its_configuration(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LocalJobRunner, "_spawn", lambda self, record: os.getpid())
    path = _build(tmp_path)
    path.write_text(path.read_text().replace('name = "lin"', 'name = "lin"\noutput_dir = "fixed"'))
    first = service.submit_surrogate_build(path)  # queued: no worker in this test
    again = service.submit_surrogate_build(path)
    assert again.reused
    assert again.id == first.id
    path.write_text(path.read_text().replace("training = 30", "training = 31"))
    with pytest.raises(ConfigError, match="another configuration"):
        service.submit_surrogate_build(path)


def test_the_policy_judges_every_kind_a_checks_manifest_names(tmp_path: Path) -> None:
    """A manifest's options are judged by their declared kinds, not only inputs."""
    from apex.domain.surrogate import SurrogateManifest

    built = surrogates.build(_build(tmp_path))
    model = built.path / "model" / "surrogate.json"
    manifest = SurrogateManifest.model_validate_json(model.read_bytes())
    stray = manifest.model_copy(update={"options": {"helper_file": "/elsewhere/h.npz"}})
    model.write_text(stray.model_dump_json())
    readable = [str(tmp_path)]
    rules = PolicySection(input_roots=readable, code_roots=readable, output_roots=readable)
    with pytest.raises(policy.PolicyViolation, match="does not declare"):
        policy.check_surrogate_check(
            {"model_file": str(model), "against": str(tmp_path / "base.toml")},
            rules, tmp_path / "root",
        )  # fmt: skip


def test_a_build_folder_is_not_deleted_under_a_live_check(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apex.services import deletion

    build = service.submit_surrogate_build(_build(service.workspace().roots[0]))
    assert service.wait(build.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = service.surrogates()
    model = Path(row["path"]) / "model" / "surrogate.json"
    monkeypatch.setattr(LocalJobRunner, "_spawn", lambda self, record: os.getpid())
    service.submit_surrogate_check(model, service.workspace().roots[0] / "base.toml", runs=3)
    unit = deletion.plan(service, build.id)
    assert any("still uses it" in reason for reason in unit.blocked), unit.blocked


# ------------------------------------------------------- the coder's reviews of 92d81cf5


class _Ui:
    """A service whose workspace is ``tmp_path``, a guest ``g`` and the owner's REST client."""

    def __init__(self, tmp_path: Path, plugins: str = "") -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from apex.api.app import create_app
        from apex.services.auth import api_token, ui_token

        self.root = tmp_path / "service"
        self.root.mkdir()
        (tmp_path / "alpha").mkdir(exist_ok=True)
        (self.root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
        self.access(plugins or f'"{PLUGIN}"')
        self.service = ApexService(self.root)
        self.http = TestClient(create_app(self.root))
        self.guest = {"Authorization": f"Bearer {ui_token(self.root, create=True, guest='g')}"}
        self.owner = {"Authorization": f"Bearer {api_token(self.root)}"}

    def access(self, surrogate_models: str) -> None:
        (self.root / "ui_access.toml").write_text(
            '[workspace]\nfolders = ["alpha"]\n[plugins]\nsimulators = ["linear"]\n'
            f'executors = ["*"]\nsurrogate_models = [{surrogate_models}]\n'
            'surrogate_designs = ["*"]\nmethods = ["*"]\npriors = ["*"]\n'
            'update_strategies = ["*"]\n[actions]\nsubmit = true\nresume = true\n'
        )

    def build(self, path: Path, **body: object) -> str:
        answer = self.http.post(
            "/v1/surrogates", json={"surrogate_file": str(path), "client": "ui", **body},
            headers=self.guest,
        )  # fmt: skip
        assert answer.status_code == 202, answer.text
        return str(answer.json()["id"])


def test_a_frozen_resume_keeps_the_builds_own_code(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``apex submit --resume BUILD_DIR`` passes ``frozen``: the build's commit, not HEAD."""
    commit = "abcdef123456"
    monkeypatch.setattr(ApexService, "pin_code",
                        lambda self, code, run_dir=None: None if code is None
                        else {"commit": "headheadhead" if code == "frozen" else str(code),
                              "python": "x"})  # fmt: skip
    monkeypatch.setattr(service.code, "check_job_kind", lambda commit, kind: None)
    part = service.submit_surrogate_build(_build(tmp_path), max_runs=7, code=commit)
    assert service.wait(part.id, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    [row] = service.surrogates()
    resumed = service.resume_run(Path(row["path"]), code="frozen")
    assert service.jobs.record(resumed.id)["code"]["commit"] == commit


def test_a_build_job_stopped_by_the_service_resumes_to_the_uninterrupted_model(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: service.stop(job) while the first wave runs; the job ends stopped."""
    import time

    from apex.workflows import surrogate as workflow

    whole = surrogates.build(_build(tmp_path / "whole"))
    submitted: list[str] = []
    real = workflow._run_wave

    def stop_then_run(*args: object, **kwargs: object) -> None:
        began = time.monotonic()
        while not submitted and time.monotonic() - began < TIMEOUT:
            time.sleep(0.05)
        if len(submitted) == 1:
            service.stop(submitted[0])
            submitted.append("stopped")
        real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow, "_run_wave", stop_then_run)
    job = service.submit_surrogate_build(_build(tmp_path / "part"))
    submitted.append(job.id)
    assert service.wait(job.id, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    monkeypatch.setattr(workflow, "_run_wave", real)
    resumed = service.resume(job.id)
    assert service.wait(resumed.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [path] = {Path(row["path"]) for row in service.surrogates()}
    for name in ("surrogate.json", "surrogate.npz"):
        assert (path / "model" / name).read_bytes() == (whole.path / "model" / name).read_bytes()
    model = path / "model" / "surrogate.json"
    check = service.submit_surrogate_check(model, tmp_path / "part" / "base.toml", runs=2)
    service.wait(check.id, timeout_s=TIMEOUT, poll_s=0.2)
    with pytest.raises(ConfigError, match="not resumed, submit it again"):
        service.resume(check.id)


def test_a_web_ui_check_judges_what_the_model_names(tmp_path: Path) -> None:
    """Without a [policy], the guard judges a manifest's names as a configuration's: a
    file the guest may not use is refused. A manifest that is not .json is never read."""
    from apex.domain.surrogate import SurrogateManifest
    from apex.storage.surrogate_files import read_surrogate

    ui = _Ui(tmp_path)
    ui.service.wait(ui.build(_build(tmp_path / "alpha")), timeout_s=TIMEOUT, poll_s=0.2)
    [row] = ui.http.get("/v1/surrogates", headers=ui.owner).json()  # the guest's: scrubbed
    model = Path(row["path"]) / "model" / "surrogate.json"
    manifest = SurrogateManifest.model_validate_json(model.read_bytes())
    model.write_text(manifest.model_copy(
        update={"options": {"helper_file": str(tmp_path / "hidden" / "h.npz")}}
    ).model_dump_json())  # fmt: skip
    check = {"model_file": str(model), "against": str(tmp_path / "alpha" / "base.toml"),
             "runs": 3, "client": "ui"}  # fmt: skip
    refused = ui.http.post("/v1/surrogates:check", json=check, headers=ui.guest)
    assert refused.status_code == 403
    assert "not available in the UI" in refused.text
    toml = model.with_suffix(".toml")
    toml.write_text(model.read_text())
    with pytest.raises(ConfigError, match=r"is a \.json file"):
        read_surrogate(toml)


def test_a_web_ui_resume_is_judged_by_todays_rules_and_builds_are_placed(tmp_path: Path) -> None:
    """A guest's stopped build no longer resumes once its plugin is hidden; an output_dir
    the guest names, even in their own folder, is refused (the service places builds)."""
    ui = _Ui(tmp_path)
    path = _build(tmp_path / "alpha")
    job = ui.build(path, max_runs=5)
    assert ui.service.wait(job, timeout_s=TIMEOUT, poll_s=0.2).state == "stopped"
    ui.access('"other_plugin"')
    hidden = ui.http.post(f"/v1/jobs/{job}:resume", json={}, headers=ui.guest)
    assert hidden.status_code == 403, hidden.text
    ui.access(f'"{PLUGIN}"')
    own = ui.service.workspace().ui_user_folder("g")
    path.write_text(path.read_text().replace('name = "lin"', f'name = "lin"\noutput_dir = "{own}"'))
    answer = ui.http.post("/v1/surrogates", json={"surrogate_file": str(path), "client": "ui"},
                          headers=ui.guest)  # fmt: skip
    assert answer.status_code == 403
    assert "placed by the service" in answer.text


def test_retention_and_deletion_keep_a_build_another_job_uses(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder holding more than a build is not removed whole by retention, and a build a
    queued run reads (its [surrogate] model_file) blocks its deletion."""
    import json

    from apex.services import deletion, guest_retention

    build = service.submit_surrogate_build(_build(service.workspace().roots[0]))
    assert service.wait(build.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = service.surrogates()
    folder = Path(row["path"])
    model = folder / "model" / "surrogate.json"
    monkeypatch.setattr(LocalJobRunner, "_spawn", lambda self, record: os.getpid())
    reader = service.jobs.submit(
        "run", {"config": json.dumps({"surrogate": {"model_file": str(model)}})},
        run_dirs=[str(tmp_path / "elsewhere")],
    )  # fmt: skip
    assert deletion.jobs_using(service, folder) == [reader]
    unit = deletion.plan(service, build.id)
    assert any("still uses it" in reason for reason in unit.blocked), unit.blocked

    (folder / "notes.txt").write_text("not a build's\n")
    kept = guest_retention.Submission("guest:g", build.id, guest_retention._parse("2026-01-01"))
    kept.jobs, kept.dirs, kept.builds = [build.id], [folder], [folder]
    deleted, _, _ = guest_retention.delete(service, kept)
    assert not deleted
    assert folder.is_dir()


def test_the_tools_refuse_a_coarse_program_another_guest_wrote(
    service: ApexService,
) -> None:
    """A build file in a guest's folder whose [coarse] stack names a program: the owner's
    tool client must not run it (as a configuration's own program would be refused)."""
    from apex.services.tools import ToolSession, call

    folder = service.workspace().ui_user_folder("g")
    assert folder is not None
    path = _build(folder)
    path.write_text(path.read_text() + '\n[coarse]\nsimulator = "opm_flow"\n'
                    'options = { executable = "/usr/bin/true" }\n')  # fmt: skip
    session = ToolSession(service, client="llm-test")
    answer = call("validate", {"config_file": str(path)}, session)
    assert "error" in answer, answer
    assert "another web-UI user wrote" in answer["error"]


def test_a_stop_before_the_fit_resumes_to_the_uninterrupted_model(tmp_path: Path) -> None:
    """Every run made, the stop asked before the fit (which may take hours): the build ends
    stopped with nothing pending, and its resume fits the same model files."""
    whole = surrogates.build(_build(tmp_path / "whole"))
    config = surrogates.load_surrogate_config(_build(tmp_path / "part"))
    waves = 40 // 4  # the serial executor's width 1, four per wave
    calls: list[int] = []

    def before_the_fit() -> bool:
        calls.append(1)
        return len(calls) > waves

    stopped = surrogates.build_config(config, stopped=before_the_fit)
    assert stopped.summary.status == "stopped"
    assert stopped.summary.pending == 0
    resumed = surrogates.resume_build(stopped.path, stopped=lambda: False)
    assert resumed.complete
    for name in ("surrogate.json", "surrogate.npz"):
        assert (resumed.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()


def test_a_builds_coarse_options_are_judged_for_guests_and_under_a_policy(
    tmp_path: Path,
) -> None:
    """The coarse simulator's options, merged over the case's (policy.build_coarse_items):
    a file outside what a guest may use, or outside a policy's roots, is refused."""
    hidden = tmp_path / "hidden" / "coarse.npz"
    ui = _Ui(tmp_path)
    path = _build(tmp_path / "alpha")
    path.write_text(path.read_text() + f'\n[coarse]\noptions = {{ forward_file = "{hidden}" }}\n')
    refused = ui.http.post("/v1/surrogates", json={"surrogate_file": str(path), "client": "ui"},
                           headers=ui.guest)  # fmt: skip
    assert refused.status_code == 403, refused.text
    assert "forward_file" in refused.text

    readable = [str(tmp_path / "alpha")]
    rules = PolicySection(input_roots=readable, code_roots=readable, output_roots=readable)
    config = surrogates.load_surrogate_config(path)
    problems = policy.violations(config, rules, tmp_path / "root")
    assert any("[coarse] simulators.linear: forward_file" in p and "input_roots" in p
               for p in problems), problems  # fmt: skip


def test_a_web_ui_validation_does_not_open_a_hidden_case(tmp_path: Path) -> None:
    ui = _Ui(tmp_path)
    cases.write_case(tmp_path / "hidden", name="base")
    path = cases.write_build(
        tmp_path / "alpha", PLUGIN, case=str(tmp_path / "hidden" / "base.toml")
    )
    answer = ui.http.post("/v1/configs:validate", json={"config_file": str(path)},
                          headers=ui.guest)  # fmt: skip
    assert answer.status_code == 404, answer.text
    # The guest's own path is echoed, as for any file they may not see; nothing the case
    # holds (its simulator, its data) is.
    assert "forward" not in answer.text
    assert "linear" not in answer.text


def test_a_builds_identity_reads_no_locations_and_only_a_designs_iterates(tmp_path: Path) -> None:
    """An executor's location counts by its path, not its files (as a run's); a design from
    a run counts its snapshot and iterates, not the run's other files."""
    from apex.domain.config import PluginSection
    from apex.services.identity import design_run, surrogate_build_id

    config = surrogates.load_surrogate_config(_build(tmp_path))
    location = tmp_path / "case_studies"
    location.mkdir()
    remote = config.model_copy(update={"execution": PluginSection(
        name="test_not_installed", options={"local_case_dir": str(location)})})  # fmt: skip
    before = surrogate_build_id(remote)
    (location / "unrelated.txt").write_text("another case\n")
    assert surrogate_build_id(remote) == before

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.json").write_text("{}")
    (run / "checkpoints" / "iter_0.npz").write_bytes(b"iterate 0")
    (run / "checkpoints" / "iter_1.npz").write_bytes(b"iterate 1")
    first = design_run(run, [0])
    (run / "simulations").mkdir()
    (run / "simulations" / "member_0.log").write_text("a forecast\n")
    (run / "checkpoints" / "iter_1.npz").write_bytes(b"iterate 1, rewritten")
    assert design_run(run, [0]) == first  # iterate 1 is not read
    assert design_run(run, None) != design_run(run, [0])


def test_an_executor_of_unknown_width_runs_the_design_in_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As the user approved: under a job's stop check, an executor that does not say its
    width (the remote ones) runs one batch, not waves that would leave its hosts idle."""
    from apex.workflows import surrogate as workflow

    batches: list[int] = []
    real = workflow._run_wave

    def counted(*args: object, **kwargs: object) -> None:
        batches.append(len(args[3]))  # type: ignore[arg-type]
        real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow, "_run_wave", counted)
    monkeypatch.setattr(workflow, "parallel_width", lambda executor, simulator=None: None)
    built = surrogates.build_config(surrogates.load_surrogate_config(_build(tmp_path)),
                                    stopped=lambda: False)  # fmt: skip
    assert built.complete
    assert batches == [40]


def test_a_resume_is_charged_at_most_the_runs_it_may_make_now(
    service: ApexService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = surrogates.build(_build(tmp_path), max_runs=7)  # 33 runs left
    monkeypatch.setattr(LocalJobRunner, "_spawn", lambda self, record: os.getpid())
    capped = service.resume_surrogate_build(first.path, max_runs=5)
    assert service.jobs.record(capped.id)["planned_runs"] == 5
    service.stop(capped.id)
    left = service.resume_surrogate_build(first.path)
    assert service.jobs.record(left.id)["planned_runs"] == 33
