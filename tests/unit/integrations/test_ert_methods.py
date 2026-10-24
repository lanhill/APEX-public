"""Several APEX methods in one APEX_HISTORY_MATCH, without ERT."""

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.domain.config import HistoryMatchConfig
from apex.integrations.ert import case as ert_case
from apex.integrations.ert.case import (
    ErtCaseInfo,
    ErtField,
    ErtObservation,
    MethodsFailed,
    history_match_all,
    method_runs,
    write_apex_case,
)
from apex.simulation.grdecl import write_grdecl

GIES = '{ name = "gies", strategy = { name = "rlmmac" } }'
ESMDA = '{ name = "esmda", options = { inflation = [2, 2] } }'
ESMDA_LOCALIZED = (
    '{ name = "esmda", options = { inflation = [2, 2] }, '
    'localization = { name = "adaptive_prior" } }'
)


def _case(tmp_path: Path) -> ErtCaseInfo:
    case_file = tmp_path / "case.ert"
    case_file.write_text("ECLBASE CASE\nFORWARD_MODEL FLOW\n")
    rng = np.random.default_rng(0)
    (tmp_path / "prior").mkdir()
    for i in range(4):
        write_grdecl(tmp_path / "prior" / f"PERMX_{i}.grdecl", "PERMX", rng.random(6))
    return ErtCaseInfo(
        case_file=case_file,
        realizations=4,
        eclbase="CASE",
        fields=[
            ErtField("PERMX", (3, 2, 1), "permx.grdecl", "prior/PERMX_<IENS>.grdecl", 0.0, 2.0)
        ],
        observations=[ErtObservation("WOPR:P1", "2020-01-31T00:00:00", 100.25, 3.0)],
        max_running=3,
        seed=7,
    )


def _apex_toml(tmp_path: Path, body: str, name: str = "apex.toml") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _methods(tmp_path: Path) -> Path:
    return _apex_toml(
        tmp_path,
        f'[run]\nname = "compare"\n\n[methods]\ngies = {GIES}\nesmda = {ESMDA}\n'
        f"esmda_localized = {ESMDA_LOCALIZED}\n",
    )


def _config(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def test_each_method_is_the_run_of_a_one_method_apex_toml(tmp_path: Path) -> None:
    """Same inputs, seed and prior: only the name and the folders differ (so the runs are
    bit-identical to one-method runs, which are themselves tested in B2)."""
    info = _case(tmp_path)
    runs = method_runs(info, _methods(tmp_path))
    assert [r.label for r in runs] == ["gies", "esmda", "esmda_localized"]
    assert [r.name for r in runs] == ["compare:gies", "compare:esmda", "compare:esmda_localized"]
    runs_dir = tmp_path / "apex_runs"
    assert runs[0].run_dir == runs_dir / "compare" / "gies"
    for run, entry in zip(runs, [GIES, ESMDA, ESMDA_LOCALIZED], strict=True):
        many = _config(write_apex_case(info, tmp_path / "apex.toml", run).config)
        single_toml = _apex_toml(
            tmp_path, f'method = {entry}\n[run]\nname = "single"\n', f"{run.label}.toml"
        )
        one = _config(write_apex_case(info, single_toml).config)
        assert many["run"]["name"] == run.name
        assert many["run"]["output_dir"] == str(run.run_dir)
        for document in (many, one):
            document["run"].pop("name")
            document["run"].pop("output_dir")
        assert many["run"] == one["run"]  # the same seed
        assert many["method"] == one["method"]
        assert many == one | {"prior": many["prior"]}
        prior_many = np.load(run.inputs / "prior.npz")
        prior_one = np.load(runs_dir / "single.inputs" / "prior.npz")
        np.testing.assert_array_equal(prior_many["permx"], prior_one["permx"])


def test_methods_and_method_are_exclusive_and_labels_are_folders(tmp_path: Path) -> None:
    info = _case(tmp_path)
    both = _apex_toml(tmp_path, f"method = {ESMDA}\n[methods]\nesmda = {ESMDA}\n", "both.toml")
    with pytest.raises(ConfigError, match="either"):
        method_runs(info, both)
    bad = _apex_toml(tmp_path, f'[methods]\n"a/b" = {ESMDA}\n', "bad.toml")
    with pytest.raises(ConfigError, match="folder"):
        method_runs(info, bad)
    with pytest.raises(ConfigError, match="no \\[methods"):
        method_runs(info, _methods(tmp_path), ["nope"])
    single = _apex_toml(tmp_path, f"method = {ESMDA}\n", "single.toml")
    with pytest.raises(ConfigError, match="--methods"):
        method_runs(info, single, ["esmda"])
    with pytest.raises(ConfigError, match="its own run directory"):
        ert_case.run_directory(info, _methods(tmp_path))


class _Recorded:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.written_back: list[Path] = []


@pytest.fixture
def recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Recorded:
    """history_match_all with the case given, and the runs and write-backs recorded."""
    import apex.services.composition as composition

    info = _case(tmp_path)
    record = _Recorded()

    class Result:
        complete = True

    def start(config: HistoryMatchConfig, max_steps: int | None, progress: object) -> Result:
        record.started.append(config.method.name + ":" + config.run.name)
        if config.run.name.endswith(":gies"):
            raise ValueError("diverged")
        return Result()

    monkeypatch.setattr(ert_case, "read_ert_case", lambda _: info)
    monkeypatch.setattr(composition, "start_config", start)
    monkeypatch.setattr(
        ert_case, "_write_back", lambda i, run_dir, s: record.written_back.append(run_dir)
    )
    return record


def test_a_failed_method_does_not_stop_the_others(tmp_path: Path, recorded: _Recorded) -> None:
    with pytest.raises(MethodsFailed) as failure:
        history_match_all(tmp_path / "case.ert", _methods(tmp_path))
    assert recorded.started == [
        "gies:compare:gies",
        "esmda:compare:esmda",
        "esmda:compare:esmda_localized",
    ]
    assert failure.value.failed == {"gies": "ValueError: diverged"}
    assert sorted(failure.value.results) == ["esmda", "esmda_localized"]
    runs = tmp_path / "apex_runs" / "compare"
    assert recorded.written_back == [runs / "esmda", runs / "esmda_localized"]


def test_methods_selects_entries(tmp_path: Path, recorded: _Recorded) -> None:
    results = history_match_all(tmp_path / "case.ert", _methods(tmp_path), methods=["esmda"])
    assert list(results) == ["esmda"]
    assert recorded.started == ["esmda:compare:esmda"]


def test_one_method_history_match_refuses_methods(tmp_path: Path, recorded: _Recorded) -> None:
    with pytest.raises(ConfigError, match="history_match_all"):
        ert_case.history_match(tmp_path / "case.ert", _methods(tmp_path))
