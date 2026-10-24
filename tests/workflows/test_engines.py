"""Run engines in studies: another engine's method beside APEX's, paired by seed.

A fake engine stands in for ERT: it runs the case with an APEX method of its own name
(no ERT needed) and marks its run directory. Job workers run in threads here, so that the
engine registered in this process is the one they find (real workers are processes that
find engines through the ``apex.engines`` entry points).
"""

import csv
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.cli.worker import run_job
from apex.core.engines import EngineContext, EngineMethod
from apex.core.errors import ConfigError
from apex.core.jobs import JobRecord
from apex.core.plugins import engines
from apex.domain.config import HistoryMatchConfig, MethodSection
from apex.domain.results import HistoryMatchResult
from apex.domain.study import StudySpec
from apex.services import composition
from apex.services.api import ApexService
from apex.services.identity import run_id
from apex.services.jobs import LocalJobRunner
from apex.services.study import expand, load_study, plan_study

TIMEOUT = 300.0
MARK = "engine.json"


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


class FakeEngine:
    """Runs the case with the APEX method its algorithm names; marks the run directory."""

    name = "fake"

    def check(self, config: HistoryMatchConfig, method: EngineMethod) -> list[str]:
        return ["the fake engine refuses this"] if method.options.get("refuse") else []

    def simulator_runs(self, method: EngineMethod, members: int | None) -> int | None:
        return None if members is None else 1000 + members

    def run(
        self, config: HistoryMatchConfig, method: EngineMethod, context: EngineContext
    ) -> HistoryMatchResult:
        run = config.run.model_copy(update={"output_dir": context.run_dir})
        own = MethodSection(name=method.name, options=dict(method.options))
        result = composition.start_config(
            config.model_copy(update={"run": run, "method": own}), None, context.progress
        )
        (context.run_dir / MARK).write_text(json.dumps(method.as_json()))
        return result


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(engines._factories, "fake", FakeEngine)

    def spawn(self: LocalJobRunner, record: JobRecord) -> int:
        def work() -> None:
            run_job(LocalJobRunner(self.root, role="worker"), record["id"])

        threading.Thread(target=work, daemon=True).start()
        return os.getpid()

    monkeypatch.setattr(LocalJobRunner, "_spawn", spawn)


def _study(case: Case, replicates: int = 2, fake: str = "") -> Path:
    path = case.directory / "study.toml"
    path.write_text(
        f"""
[study]
name = "engines"
seed = 3
replicates = {replicates}
metrics = ["normalized_data_mismatch:all"]

[study.cases]
linear = "config.toml"

[study.methods.esmda]
name = "esmda"
options = {{ inflation = [4, 4, 4, 4] }}
strategy = {{ name = "approx" }}

[study.methods.fake_esmda]
engine = "fake"
name = "esmda"
options = {{ inflation = [2, 2]{fake} }}

[[study.variants]]
name = "plain"

[[study.variants]]
name = "localized"
patch = {{ method = {{ localization = {{ name = "adaptive_prior" }} }} }}
"""
    )
    return path


def test_engine_cells_are_paired_by_seed_and_skip_method_variants(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    spec = load_study(_study(case))

    cells = expand(spec, tmp_path, case.directory)

    labels = [c.label for c in cells]
    assert labels == [
        "linear/esmda/plain/r0", "linear/esmda/plain/r1",
        "linear/esmda/localized/r0", "linear/esmda/localized/r1",
        "linear/fake_esmda/plain/r0", "linear/fake_esmda/plain/r1",
    ]  # fmt: skip
    fake = [c for c in cells if c.engine is not None]
    assert {c.engine.name for c in fake if c.engine} == {"esmda"}
    assert [c.config.run.seed for c in fake] == [c.config.run.seed for c in cells[:2]]
    plan = plan_study(ApexService(tmp_path / "service"), spec, case.directory)
    assert plan.validation.valid, plan.validation.errors
    assert [s.label for s in plan.skipped] == [
        "linear/fake_esmda/localized/r0",
        "linear/fake_esmda/localized/r1",
    ]
    fake_plans = [p for p in plan.validation.plans if p.engine == "fake"]
    assert [p.max_simulator_runs for p in fake_plans] == [1020, 1020]  # 20 members
    assert all(p.method == "fake:esmda" for p in fake_plans)
    apex_ids = {p.run_id for p in plan.validation.plans if p.engine == "apex"}
    assert not apex_ids & {p.run_id for p in fake_plans}
    # The run id of an engine cell is its configuration and its engine method.
    first = fake[0]
    assert first.engine is not None
    assert fake_plans[0].run_id == run_id(first.config, first.engine.as_json())
    notes = ApexService(tmp_path / "service").validate_study(spec, case.directory).notes
    assert len(notes) == 2
    assert "patches [method]" in notes[0]


def test_a_study_runs_engine_cells_and_compares_them(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    service = ApexService(tmp_path / "service", max_concurrent_jobs=3)

    handle = service.submit_study(_study(case), approve=lambda plan: True)
    status = service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.3)

    assert status.state == "complete", status
    assert len(status.children) == 6
    record = json.loads((service.root / "studies/engines/study.json").read_text())
    engines_of = {(c["method"], c["engine"]) for c in record["cells"]}
    assert engines_of == {("esmda", "apex"), ("fake_esmda", "fake")}
    assert len(record["skipped"]) == 2
    for cell in record["cells"]:
        marked = (Path(cell["run_dir"]) / MARK).exists()
        assert marked == (cell["engine"] == "fake")
    # The engine's run is what that engine computes: the fake runs ES-MDA [2, 2].
    fake_cell = next(c for c in record["cells"] if c["engine"] == "fake")
    history = json.loads((Path(fake_cell["run_dir"]) / "history.json").read_text())
    assert len(history) == 3
    # Paired by seed: the same prior as APEX's cell of the same replicate.
    priors = {}
    for cell in record["cells"]:
        if cell["variant"] == "plain" and cell["replicate"] == 1:
            with np.load(Path(cell["run_dir"]) / "checkpoints/iter_0.npz") as data:
                priors[cell["engine"]] = data["state"].copy()
    np.testing.assert_array_equal(priors["apex"], priors["fake"])
    comparison = service.compare("engines")
    (mismatch,) = comparison.metrics
    assert {(v.method, v.engine) for v in mismatch.variants} == engines_of
    assert ("fake_esmda", "esmda") in {(d.method, d.reference_method) for d in mismatch.differences}
    assert [c["label"] for c in comparison.skipped] == [
        "linear/fake_esmda/localized/r0",
        "linear/fake_esmda/localized/r1",
    ]
    rows = list(csv.DictReader((service.root / "studies/engines/comparison.csv").open()))
    assert {r["engine"] for r in rows} == {"apex", "fake"}
    # Submitting again reuses every run, the engine's too.
    again = service.submit_study(_study(case), approve=lambda plan: True)
    assert service.wait(again.id, timeout_s=TIMEOUT, poll_s=0.3).state == "complete"


def test_engine_refusals(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    service = ApexService(tmp_path / "service")
    refused = service.validate_study(_study(case, fake=", refuse = true"), case.directory)
    assert not refused.valid
    assert any("the fake engine refuses this" in e for e in refused.errors)
    text = _study(case).read_text().replace('engine = "fake"', 'engine = "nowhere"')
    (case.directory / "study.toml").write_text(text)
    unknown = service.validate_study(case.directory / "study.toml", case.directory)
    assert any("no run engine 'nowhere'" in e for e in unknown.errors)
    with pytest.raises(ValueError, match="name and options only"):
        StudySpec.model_validate(
            {
                "study": {
                    "name": "s",
                    "cases": {"c": "c.toml"},
                    "methods": {
                        "e": {"engine": "fake", "name": "es", "strategy": {"name": "approx"}}
                    },
                    "variants": [{"name": "v"}],
                }
            }
        )
    with pytest.raises(ConfigError, match="invalid study"):
        service.submit_study(_study(case, fake=", refuse = true"), approve=lambda plan: True)
