"""A plugin whose extra is not installed is refused before launch; failures say why (#268).

The reported issue (2026-11-02): an ES-MDA study comparing ``approx`` with ``ert_adaptive``
validated and launched where the ``ert`` extra was not installed; every ``ert_adaptive``
cell failed at its first update, the study ended ``failed`` with no error, and the cells'
``events.jsonl`` ended at their prior's forecast.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest

from apex.services import composition, plugins
from apex.services.jobs import LocalJobRunner
from apex.services.validation import validate_history_match


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


def _missing(factory: object) -> tuple[str, str] | None:
    """Every plugin that requires a module lacks it, as in an environment without extras."""
    requires = getattr(factory, "requires", None) or {}
    return next(iter(requires.items()), None)


def test_validation_names_the_missing_extra_and_how_to_install_it(
    linear_case: Callable[..., Case], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = linear_case(
        method='name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "ert_adaptive" }',
    )
    monkeypatch.setattr(plugins, "missing_extra", _missing)

    validation = validate_history_match(case.config)

    assert not validation.valid
    [error] = [e for e in validation.errors if "extra" in e]
    assert "update_strategies 'ert_adaptive' needs iterative_ensemble_smoother" in error
    assert "the ert extra" in error
    assert "uv sync --inexact --extra ert" in error
    # A plugin with nothing to require is not refused.
    plain = linear_case(name="plain", method='name = "esmda"\noptions = { inflation = [2, 2] }')
    assert validate_history_match(plain.config).valid


def test_a_failed_run_says_why_in_its_own_events(
    linear_case: Callable[..., Case], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = linear_case(method='name = "esmda"\noptions = { inflation = [2, 2] }')
    config = composition.load_config(case.config)

    def fail(*_: Any, **__: Any) -> None:  # noqa: ANN401
        raise RuntimeError("the first update failed")

    monkeypatch.setattr(composition.Run, "execute", fail)
    with pytest.raises(RuntimeError, match="first update"):
        composition.start_config(config, None)

    assert config.run.output_dir is not None
    events = [json.loads(line) for line in
              (config.run.output_dir / "events.jsonl").read_text().splitlines()]  # fmt: skip
    [failed] = [e for e in events if e.get("event") == "run_failed"]
    assert failed["error"] == "RuntimeError: the first update failed"


def test_a_failed_study_carries_its_cells_errors(tmp_path: Path) -> None:
    runner = LocalJobRunner(tmp_path)
    study = runner.submit("study", {}, start=False)
    cells = [("approx", None), ("ert_adaptive", "WorkflowError: ERT's smoothers need the ert "
                                                "extra"), ("ert_adaptive", None)]  # fmt: skip
    for replicate, (variant, error) in enumerate(cells):
        child = runner.submit("run", {}, parent=study, start=False,
                              annotations={"case": "c", "variant": variant,
                                           "replicate": str(replicate)})  # fmt: skip
        runner.update(child, state="failed" if error or replicate == 2 else "complete",
                      error=error)  # fmt: skip
    runner.update(study, sealed=True)

    runner._finish_studies()

    record = runner.record(study)
    assert record["state"] == "failed"
    assert record["error"].startswith("2 of 3 cells failed: ")
    assert "c/ert_adaptive/r1: WorkflowError: ERT's smoothers need the ert extra" in record["error"]
    assert "c/ert_adaptive/r2: no error recorded" in record["error"]
    events = [e for _, e in runner.store.events(study) if e.get("event") == "study_cells_failed"]
    assert [c["cell"] for c in events[0]["cells"]] == ["c/ert_adaptive/r1", "c/ert_adaptive/r2"]


def test_an_optimizer_whose_extra_is_missing_is_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of #268: ropt (the ert extra) passed an optimization's validation."""
    import tomllib

    from apex.services.api import ApexService

    example = Path(__file__).resolve().parents[2] / "examples" / "toy" / "rastrigin"
    config = tomllib.loads((example / "rastrigin_pso.toml").read_text())
    service = ApexService(tmp_path / "service")
    assert service.validate_optimization(config, base_dir=example).valid
    monkeypatch.setattr(plugins, "missing_extra", _missing)
    config["optimizer"] = {"name": "ropt", "options": {"method": "scipy/slsqp"}}

    validation = service.validate_optimization(config, base_dir=example)

    assert not validation.valid
    assert any("optimizers 'ropt' needs" in e and "extra" in e for e in validation.errors)
