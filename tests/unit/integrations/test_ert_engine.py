"""ERT as a run engine: its checks, its cost and `apex ert compare`; no ERT run."""

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apex.core.engines import EngineMethod
from apex.core.plugins import engines
from apex.domain.config import PluginSection
from apex.integrations.ert import engine as ert_engine
from apex.integrations.ert.engine import ErtEngine
from apex.services.composition import load_config

CASE = (
    Path(__file__).resolve().parents[3] / "examples" / "3dbox_production_avo" / "config_esmda.toml"
)


def test_the_engine_is_found_through_its_entry_point() -> None:
    assert isinstance(engines.create("ert"), ErtEngine)


def test_run_models_options_and_cases_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ert_engine, "_ert_version", lambda: "26.0.1")
    loaded = load_config(CASE)
    # The case's wells only: ERT's export takes no field data and no geophysics.
    observations = loaded.observations.model_copy(update={"fields": []})
    config = loaded.model_copy(update={"geophysics": None, "observations": observations})
    engine = ErtEngine()

    def problems(name: str, **options: object) -> str:
        return "; ".join(engine.check(config, EngineMethod("ert", name, dict(options))))

    assert "not one of ensemble_smoother, es_mda" in problems("ies")
    assert "takes no options ['localization']" in problems("es_mda", localization=True)
    assert "weights must be positive" in problems("es_mda", weights=[4, 0])
    assert problems("es_mda", weights=[4, 4]) == ""
    refused = engine.check(loaded, EngineMethod("ert", "es_mda", {"weights": [4, 4]}))
    assert any("geophysical data" in p for p in refused)
    linear = config.model_copy(update={"simulator": PluginSection(name="linear")})
    refused = engine.check(linear, EngineMethod("ert", "es_mda", {"weights": [4, 4]}))
    assert any("opm_flow and eclipse" in p for p in refused)
    monkeypatch.setattr(ert_engine, "_ert_version", lambda: None)
    assert "needs the `ert` extra" in problems("ensemble_smoother")


def test_the_cost_is_every_update_plus_the_prior() -> None:
    engine = ErtEngine()
    assert engine.simulator_runs(EngineMethod("ert", "es_mda", {"weights": [4] * 4}), 10) == 50
    assert engine.simulator_runs(EngineMethod("ert", "es_mda", {"weights": "2,2"}), 10) == 30
    assert engine.simulator_runs(EngineMethod("ert", "es_mda"), 10) == 40  # ERT's 4,2,1
    assert engine.simulator_runs(EngineMethod("ert", "ensemble_smoother"), 10) == 20
    assert engine.simulator_runs(EngineMethod("ert", "ensemble_smoother"), None) is None


def test_ert_compare_writes_a_study_of_apex_against_ert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import apex.cli.services_commands as commands
    from apex.cli.main import app

    submitted: list[Path] = []
    monkeypatch.setattr(commands, "study_run", lambda path, **kwargs: submitted.append(path))
    config = tmp_path / "case.toml"
    config.write_text(CASE.read_text())
    for source in CASE.parent.iterdir():
        if source != CASE:
            (tmp_path / source.name).symlink_to(source)

    result = CliRunner().invoke(
        app, ["ert", "compare", str(config), "--weights", "4,4,4,4", "--replicates", "2"]
    )

    assert result.exit_code == 0, result.output
    (path,) = submitted
    study = tomllib.loads(path.read_text())["study"]
    case = load_config(config)
    assert study["replicates"] == 2
    assert study["cases"] == {case.run.name: str(config.resolve())}
    apex_label = f"apex_{case.method.name}"
    assert study["methods"][apex_label]["name"] == case.method.name
    assert study["methods"]["ert_es_mda"] == {
        "engine": "ert",
        "name": "es_mda",
        "options": {"weights": [4.0, 4.0, 4.0, 4.0]},
    }


def test_finished_realizations_are_reported_from_the_run_paths(tmp_path: Path) -> None:
    from apex.integrations.ert.engine import RunPathProgress

    events: list[tuple[str, object]] = []

    class Recorder:
        def forecast_started(self, *, iteration: int | None, label: str, n_members: int) -> None:
            events.append(("start", (iteration, n_members)))

        def member_finished(self, *, member_id: int, ok: bool) -> None:
            events.append(("member", (member_id, ok)))

        def forecast_finished(self, *, n_ok: int, n_failed: int, seconds: float) -> None:
            events.append(("finish", (n_ok, n_failed)))

    runs = tmp_path / "runs"
    for i in range(3):
        (runs / f"realization-{i}" / "iter-0").mkdir(parents=True)
    progress = RunPathProgress(runs, Recorder(), "es_mda")  # type: ignore[arg-type]
    progress.poll()
    (runs / "realization-1" / "iter-0" / "OK").write_text("")
    (runs / "realization-2" / "iter-0" / "ERROR").write_text("")
    progress.poll()
    progress.poll()  # nothing new
    (runs / "realization-0" / "iter-0" / "OK").write_text("")
    for i in range(3):
        (runs / f"realization-{i}" / "iter-1").mkdir()
    progress.poll()
    progress.close()

    assert events == [
        ("start", (0, 3)),
        ("member", (1, True)),
        ("member", (2, False)),
        ("member", (0, True)),
        ("finish", (2, 1)),
        ("start", (1, 3)),
        ("finish", (0, 0)),
    ]
