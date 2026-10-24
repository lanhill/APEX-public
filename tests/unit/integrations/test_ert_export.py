"""``apex ert export`` and ``APEX_RENDER_DECK``; no ERT, no simulator."""

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.core.simulation import MemberInput
from apex.integrations.ert.export import export_case
from apex.integrations.ert.render import render
from apex.services import composition
from apex.simulation.deck import render_deck
from apex.simulation.grdecl import read_grdecl, write_grdecl

# The tiny case's factory (conftest.py); tests/ is not a package.
CaseFactory = Callable[..., Path]


def test_the_case_holds_apexs_prior_observations_and_settings(
    tmp_path: Path, tiny_case: CaseFactory
) -> None:
    config = tiny_case()
    out = tmp_path / "ert"

    result = export_case(config, out)

    text = result.case.read_text()
    assert "NUM_REALIZATIONS 5" in text
    assert "ECLBASE TINY_CASE" in text
    assert "RANDOM_SEED 11" in text
    assert (
        "FIELD PORO PARAMETER poro.grdecl INIT_FILES:prior/PORO_<IENS>.grdecl "
        "FORWARD_INIT:False MIN:0.05 MAX:0.4\n"
    ) in text
    assert "INIT_FILES:prior/PERMX_<IENS>.grdecl FORWARD_INIT:False\n" in text  # unbounded
    assert "QUEUE_OPTION LOCAL MAX_RUNNING 3" in text
    assert "ENKF_ALPHA" in text
    assert "STD_CUTOFF 1e-30" in text
    assert "ANALYSIS_SET_VAR STD_ENKF ENKF_TRUNCATION 0.98\n" in text  # approx's default energy
    for key in ("WOPR:P1", "WOPR:P2", "WBHP:I1", "FOPR"):
        assert f"SUMMARY {key}\n" in text
    # The prior is APEX's own draw, member by member, in Eclipse order.
    run = composition.build_run(composition.load_config(config))
    prior = run.workflow.layout.split(composition.draw_prior(run))
    for member in range(5):
        np.testing.assert_allclose(
            read_grdecl(out / "prior" / f"PORO_{member}.grdecl", "PORO"),
            prior["poro"][:, member],
            rtol=1e-10,
        )
    with (out / "observations" / "observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {"keyword": "WOPR", "value": "100.0", "error": repr(math.sqrt(4.0)),
                       "date": "2020-01-31", "well": "P1"}  # fmt: skip
    assert {r["keyword"] for r in rows if r["well"] == ""} == {"FOPR"}
    assert (
        out / "observations" / "obs.txt"
    ).read_text() == "SUMMARY { VALUES = observations.csv; };\n"
    assert not (out / "observations" / "held_out.csv").exists()
    manifest = json.loads((out / "apex_export.json").read_text())
    assert manifest["fields"]["PORO"] == {"variable": "poro", "file": "poro.grdecl"}
    grid = (out / "grid" / "TINY_CASE.GRID").read_bytes()
    assert grid[4:12] == b"DIMENS  "
    np.testing.assert_array_equal(np.frombuffer(grid[28:40], dtype=">i4"), [4, 3, 1])
    resfo = pytest.importorskip("resfo")  # ERT's reader (the `ert` extra), when installed
    [(keyword, dims)] = list(resfo.read(out / "grid" / "TINY_CASE.GRID"))
    assert keyword.strip() == "DIMENS"
    np.testing.assert_array_equal(dims, [4, 3, 1])


def test_held_out_data_are_simulated_but_not_assimilated(
    tmp_path: Path, tiny_case: CaseFactory
) -> None:
    config = tiny_case(
        validation='[validation]\ncv = { wells = ["P2"] }\n', method='name = "lm_enrml"'
    )  # cross validation needs an iterative method
    out = tmp_path / "ert"

    result = export_case(config, out)

    assert (result.assimilated, result.held_out) == (6, 2)
    with (out / "observations" / "observations.csv").open() as handle:
        assert all(r["well"] != "P2" for r in csv.DictReader(handle))
    with (out / "observations" / "held_out.csv").open() as handle:
        held = list(csv.DictReader(handle))
    assert {(r["role"], r["keyword"], r["well"]) for r in held} == {("cv", "WOPR", "P2")}
    assert "SUMMARY WOPR:P2\n" in result.case.read_text()  # simulated


def test_comparison_settings_can_be_left_to_ert(tmp_path: Path, tiny_case: CaseFactory) -> None:
    result = export_case(tiny_case(), tmp_path / "ert", ert_defaults=True)

    assert "ENKF_ALPHA" not in result.case.read_text()


@pytest.mark.parametrize(
    ("simulator", "match"),
    [("linear", "opm_flow")],
)
def test_cases_phase_a1_cannot_export_are_refused(
    tmp_path: Path, tiny_case: CaseFactory, simulator: str, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        export_case(tiny_case(simulator=simulator), tmp_path / "ert")


def test_an_existing_case_is_not_overwritten_unless_asked(
    tmp_path: Path, tiny_case: CaseFactory
) -> None:
    config = tiny_case()
    export_case(config, tmp_path / "ert")
    with pytest.raises(ConfigError, match="not empty"):
        export_case(config, tmp_path / "ert")
    export_case(config, tmp_path / "ert", overwrite=True)


def test_the_step_renders_apexs_deck_from_the_run_paths_fields(
    tmp_path: Path, tiny_case: CaseFactory
) -> None:
    out = tmp_path / "ert"
    export_case(tiny_case(), out)
    run_path = tmp_path / "realization-2"
    run_path.mkdir()
    rng = np.random.default_rng(0)
    poro, permx = rng.uniform(0.1, 0.3, 12), rng.normal(5, 1, 12)
    write_grdecl(run_path / "poro.grdecl", "PORO", poro)  # as ERT writes them
    write_grdecl(run_path / "permx.grdecl", "PERMX", permx)

    render(out / "apex_export.json", Path("CASE.DATA"), 2, run_path)

    expected = render_deck(
        tmp_path / "deck.mako", MemberInput(2, {"poro": poro, "permx": permx}, run_path),
        {"level": -1, "coarse": [], "label": "tiny"},
    )  # fmt: skip
    assert (run_path / "CASE.DATA").read_text() == expected
    assert "member 2, level -1, tiny" in expected


def test_a_missing_field_file_is_reported(tmp_path: Path, tiny_case: CaseFactory) -> None:
    out = tmp_path / "ert"
    export_case(tiny_case(), out)
    with pytest.raises(ValueError, match=r"poro\.grdecl is missing"):
        render(out / "apex_export.json", Path("CASE.DATA"), 0, tmp_path)


def test_grdecl_reads_repeat_counts_and_comments(tmp_path: Path, tiny_case: CaseFactory) -> None:
    path = tmp_path / "f.grdecl"
    path.write_text("-- header\nPORO\n 3*0.25 0.1 -- comment\n0.2/\n")

    np.testing.assert_array_equal(read_grdecl(path, "PORO"), [0.25, 0.25, 0.25, 0.1, 0.2])
    with pytest.raises(ValueError, match="no keyword"):
        read_grdecl(path, "PERMX")


def test_the_run_suggestion_follows_the_configurations_method(
    tmp_path: Path, tiny_case: CaseFactory
) -> None:
    """Release tester: the export suggested ``--weights 4,4,4,4`` for a case whose ES-MDA
    inflation was ``[5, 5, 5, 5, 5]``."""
    esmda = export_case(
        tiny_case(method='name = "esmda"\noptions = { inflation = [5, 5, 5, 5, 5] }'),
        tmp_path / "esmda",
    )
    assert esmda.run_arguments == ("--mode", "es_mda", "--weights", "5,5,5,5,5")
    assert "ert es_mda --weights 5,5,5,5,5 case.ert" in esmda.case.read_text()
    (tmp_path / "es").mkdir()
    es = export_case(tiny_case(method='name = "es"'), tmp_path / "es" / "case")
    assert es.run_arguments == ("--mode", "ensemble_smoother")
    (tmp_path / "gies").mkdir()
    gies = export_case(tiny_case(method='name = "gies"'), tmp_path / "gies" / "case")
    assert gies.run_arguments == ("--mode", "es_mda")  # no ERT equivalent: ERT's own weights


def test_the_export_command_suggests_the_configurations_weights(
    tmp_path: Path, tiny_case: CaseFactory
) -> None:
    """The release tester was told ``--weights 4,4,4,4`` for ``[5, 5, 5, 5, 5]``. ERT rescales
    the weights to ``sum(1/w) = 1``, which APEX's ES-MDA already requires: the header says so,
    and how ERT's environment gets APEX's forward-model step."""
    from typer.testing import CliRunner

    from apex.cli.main import app

    config = tiny_case(method='name = "esmda"\noptions = { inflation = [5, 5, 5, 5, 5] }')
    out = tmp_path / "five"
    result = CliRunner().invoke(app, ["ert", "export", str(config), str(out)])

    assert result.exit_code == 0, result.output
    assert f"run: apex ert run {out} --mode es_mda --weights 5,5,5,5,5" in result.output
    header = (out / "case.ert").read_text()
    assert "-- Run, e.g.: ert es_mda --weights 5,5,5,5,5 case.ert" in header
    assert "sum of 1/w is 1" in header
    assert "uv sync --locked --extra ert" in header
    assert "ADR" not in header.split("NUM_REALIZATIONS")[0]
