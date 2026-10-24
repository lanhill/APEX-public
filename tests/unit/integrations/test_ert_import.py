"""``apex ert import`` and ``APEX_CV``; no ERT, no simulator.

ERT's run paths are written by hand, and the summary reader is replaced by a fixed
function of the run path, so the replay's records can be checked exactly.
"""

import json
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.integrations.ert import cv, importer
from apex.integrations.ert.export import export_case
from apex.simulation.grdecl import read_grdecl, write_grdecl
from apex.storage.run_directory import RunDirectory

# The tiny case's factory (conftest.py); tests/ is not a package.
CaseFactory = Callable[..., Path]


N = 5


def _response(case: Path, keys: Sequence[DataKey]) -> np.ndarray:
    """A deterministic 'summary' of a run path: each key's value a function of path and key."""
    run = zlib.crc32(str(case.parent).encode())
    return np.array([100.0 + (zlib.crc32(f"{run}|{k}".encode()) % 1000) / 100.0 for k in keys])


def _ert_run(out: Path, iterations: int = 2, failed: tuple[tuple[int, int], ...] = ()) -> None:
    """Write ERT-like run paths: iteration 0 the exported prior, later ones shifted."""
    manifest = json.loads((out / "apex_export.json").read_text())
    for k in range(iterations):
        for i in range(N):
            path = out / "runs" / f"realization-{i}" / f"iter-{k}"
            path.mkdir(parents=True)
            if (i, k) in failed:
                continue
            for name, field in manifest["fields"].items():
                prior = read_grdecl(out / "prior" / f"{name}_{i}.grdecl", name)
                write_grdecl(path / field["file"], name, prior * (1 + 0.01 * k))


@pytest.fixture
def stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importer, "read_summary", _response)
    monkeypatch.setattr(cv, "read_summary", _response)


def test_ert_iterates_are_replayed_as_an_apex_run(
    tmp_path: Path, stored: None, tiny_case: CaseFactory
) -> None:
    out = tmp_path / "ert"
    export_case(tiny_case(), out)
    _ert_run(out)

    result = importer.import_case(out)

    run = RunDirectory(out / "apex_run")
    assert [h.iteration for h in result.history] == [0, 1]
    for k in (0, 1):
        saved = run.load_arrays(f"iter_{k}")
        np.testing.assert_array_equal(saved["member_ids"], np.arange(N))
        for i in range(N):
            expected = read_grdecl(
                out / "runs" / f"realization-{i}" / f"iter-{k}" / "poro.grdecl", "PORO"
            )
            np.testing.assert_array_equal(saved["state"][:12, i], expected)
            case = out / "runs" / f"realization-{i}" / f"iter-{k}" / "TINY_CASE"
            np.testing.assert_allclose(saved["predicted"][:, i], _response(case, _keys(run)))
    assert json.loads((out / "apex_run" / "ert_import.json").read_text())["source"] == "ert"
    assert json.loads((out / "apex_run" / "summary.json").read_text())["iterations"] == 1


def _keys(run: RunDirectory) -> list[DataKey]:
    rows = (
        (Path(json.loads(run.config_text())["observations"]["file"])).read_text().splitlines()[1:]
    )
    return [DataKey(r.split(",")[0], r.split(",")[1]) for r in rows]


def test_a_failed_ert_realization_fails_in_the_replay(
    tmp_path: Path, stored: None, tiny_case: CaseFactory
) -> None:
    out = tmp_path / "ert"
    export_case(tiny_case(), out)
    _ert_run(out, failed=((3, 1),))

    result = importer.import_case(out)

    assert result.member_ids.tolist() == [0, 1, 2, 4]


def test_cv_scores_and_selection_are_apexs(
    tmp_path: Path, stored: None, tiny_case: CaseFactory
) -> None:
    """APEX_CV's scores of ERT's run paths equal the imported run's cv metric, per iterate."""
    out = tmp_path / "ert"
    config = tiny_case(
        validation='[validation]\ncv = { wells = ["P2"] }\n', method='name = "lm_enrml"'
    )
    export_case(config, out)
    assert "HOOK_WORKFLOW APEX_CV_SCORE POST_SIMULATION" in (out / "case.ert").read_text()
    assert (out / "workflows" / "apex_cv_select").read_text().split()[2:] == ["patience", "2"]
    _ert_run(out, iterations=3)
    for k in range(3):
        cv.score_iteration(out, k, sorted((out / "runs").glob(f"realization-*/iter-{k}")))

    selection = cv.select(out, "select")
    result = importer.import_case(out)

    run = RunDirectory(out / "apex_run")
    for k in range(3):
        imported = run.load_arrays(f"data_mismatch_{k}")["metric:cv_normalized_data_mismatch:all"]
        scored = json.loads((out / "apex_cv.json").read_text())["iterations"][str(k)]["cv"]
        np.testing.assert_allclose(
            imported, [scored["members"][str(i)] for i in range(N)], rtol=1e-12
        )
    assert selection["selected"] == result.selected_iteration


def test_patience_reports_where_it_would_stop(tmp_path: Path) -> None:
    (tmp_path / "apex_cv.json").write_text(json.dumps({"iterations": {
        str(k): {"cv": {"mean": v, "members": {}}} for k, v in enumerate([5.0, 3.0, 4.0, 4.5, 2.0])
    }}))  # fmt: skip

    selection = cv.select(tmp_path, "patience", 2)

    assert selection == {
        "rule": "patience", "selected": 1, "would_stop_at": 3, "minimum_overall": 4,
    }  # fmt: skip
    assert cv.select(tmp_path, "select")["selected"] == 4


def test_an_unexported_or_unrun_case_is_refused(tmp_path: Path, tiny_case: CaseFactory) -> None:
    with pytest.raises(ConfigError, match=r"apex_export\.json"):
        importer.import_case(tmp_path)
    export_case(tiny_case(), tmp_path / "ert")
    with pytest.raises(ConfigError, match="no ERT run paths"):
        importer.import_case(tmp_path / "ert")
