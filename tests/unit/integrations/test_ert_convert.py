"""``apex ert convert``: an ERT case as a native APEX case; no simulator run."""

from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.core.simulation import MemberInput
from apex.simulation.deck import render_deck
from apex.simulation.grdecl import write_grdecl

pytest.importorskip("ert")

DATA = """RUNSPEC
DIMENS
 3 2 1 /
GRID
INCLUDE
 'permx.grdecl' /
INCLUDE
 'static.inc' /
PROPS
INCLUDE
 'props.inc' /
-- realization <IENS>, mult <MULT>, 100% water ${not_mako}
END
"""


def _case(tmp_path: Path, distributions: str = "KX NORMAL 1 0.1\n", extra: str = "") -> Path:
    from apex.integrations.ert.export import _write_dimens

    (tmp_path / "prior").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(4):
        write_grdecl(tmp_path / "prior" / f"PERMX_{i}.grdecl", "PERMX", rng.normal(size=6))
    (tmp_path / "CASE.DATA").write_text(DATA)
    (tmp_path / "static.inc").write_text("-- static\n")
    keys = [line.split()[0] for line in distributions.splitlines()]
    (tmp_path / "props.tmpl").write_text("".join(f"MULTX\n <{k}> /\n" for k in keys))
    (tmp_path / "coef_priors.txt").write_text(distributions)
    (tmp_path / "coef.tmpl").write_text("".join(f"{k} <{k}>\n" for k in keys))
    (tmp_path / "obs.txt").write_text(
        "SUMMARY_OBSERVATION FOPR_1 { VALUE = 10; ERROR = 1; DATE = 2020-01-31; KEY = FOPR; };\n"
    )
    _write_dimens(tmp_path / "grid.GRID", (3, 2, 1))
    case = tmp_path / "case.ert"
    case.write_text(
        "NUM_REALIZATIONS 4\nRUNPATH runs/realization-<IENS>/iter-<ITER>\nECLBASE CASE\n"
        "DATA_FILE CASE.DATA\nDEFINE <MULT> 2.5\nRANDOM_SEED 7\nGRID grid.GRID\n"
        "FIELD PERMX PARAMETER permx.grdecl INIT_FILES:prior/PERMX_%d.grdecl FORWARD_INIT:False\n"
        "GEN_KW COEF coef.tmpl coef.txt coef_priors.txt\nRUN_TEMPLATE props.tmpl props.inc\n"
        f"SUMMARY FOPR\nOBS_CONFIG obs.txt\n{extra}FORWARD_MODEL FLOW\n"
    )
    return case


def test_the_case_becomes_a_native_configuration_and_deck(tmp_path: Path) -> None:
    from apex.integrations.ert.config import parse_case
    from apex.integrations.ert.convert import convert_case
    from apex.integrations.ert.parameters import gen_kw_keys
    from apex.services import composition

    case = _case(tmp_path / "case")
    out = tmp_path / "native"

    result = convert_case(case, out)

    config = composition.load_config(result.config)
    assert config.simulator.name == "opm_flow"
    assert [v.name for v in config.state.variables] == ["permx", "coef"]
    assert config.run.seed == 7
    prior = np.load(out / "prior.npz")
    from apex.simulation.grdecl import read_grdecl

    np.testing.assert_array_equal(
        prior["permx"][:, 2], read_grdecl(case.parent / "prior" / "PERMX_2.grdecl", "PERMX")
    )
    member = MemberInput(2, {"permx": prior["permx"][:, 2], "coef": prior["coef"][:, 2]}, out)
    deck = render_deck(result.template, member, {})
    # The field inlined from the state; ERT's GEN_KW transform of the latent value;
    # substitutions; the static INCLUDE pointing at the case's file; the deck's text literal.
    (key,) = gen_kw_keys(parse_case(case), "COEF")
    expected = float(key.transform_numpy(np.array([prior["coef"][0, 2]]))[0])
    lines = deck.splitlines()
    assert lines[lines.index("PERMX") + 1] == f"{prior['permx'][0, 2]:.10e}"
    assert float(lines[lines.index("MULTX") + 1].split()[0]) == pytest.approx(expected, rel=1e-9)
    assert "-- realization 2, mult 2.5, 100% water ${not_mako}" in deck
    assert f"'{case.parent / 'static.inc'}'" in deck
    assert any("coef.txt" in note for note in result.notes)  # written by ERT, not INCLUDEd
    assert not any("props.inc" in note for note in result.notes)  # it was inlined


@pytest.mark.parametrize(
    "line",
    [
        "KX NORMAL 1 0.1",
        "KX LOGNORMAL 0.5 0.3",
        "KX UNIFORM 2 5",
        "KX LOGUNIF 0.1 10",
        "KX TRUNCATED_NORMAL 1 2 0 1.5",
        "KX CONST 4.5",
        "KX RAW",
    ],
)
def test_every_distribution_is_erts_transform(tmp_path: Path, line: str) -> None:
    from apex.integrations.ert.config import parse_case
    from apex.integrations.ert.convert import convert_case
    from apex.integrations.ert.parameters import gen_kw_keys

    case = _case(tmp_path / "case", distributions=line + "\n")
    result = convert_case(case, tmp_path / "native")
    (key,) = gen_kw_keys(parse_case(case), "COEF")
    prior = np.load(tmp_path / "native" / "prior.npz")
    for x in (-1.7, 0.0, 0.4, 2.2):
        member = MemberInput(0, {"permx": prior["permx"][:, 0], "coef": np.array([x])}, tmp_path)
        deck = render_deck(result.template, member, {}).splitlines()
        value = float(deck[deck.index("MULTX") + 1].split()[0])
        assert value == pytest.approx(float(key.transform_numpy(np.array([x]))[0]), rel=1e-9)


def test_what_a_native_case_cannot_run_is_refused(tmp_path: Path) -> None:
    from apex.integrations.ert.convert import convert_case

    two_steps = _case(tmp_path / "a", extra="FORWARD_MODEL COPY_FILE(<FROM>=a, <TO>=b)\n")
    with pytest.raises(ConfigError, match="forward model"):
        convert_case(two_steps, tmp_path / "a_native")
    triangular = _case(tmp_path / "b", distributions="KX TRIANGULAR 0 1 2\n")
    with pytest.raises(ConfigError, match="distribution triangular"):
        convert_case(triangular, tmp_path / "b_native")
    good = _case(tmp_path / "c")
    convert_case(good, tmp_path / "c_native")
    with pytest.raises(ConfigError, match="not empty"):
        convert_case(good, tmp_path / "c_native")


def test_gen_kw_values_are_written_into_a_run_path_as_erts_own_writers_do(tmp_path: Path) -> None:
    """``APEX_WRITE_PARAMETERS`` runs ERT's run-path writers (a private module, moved in ERT 26)."""
    import json

    from apex.integrations.ert.config import parse_case
    from apex.integrations.ert.parameters import gen_kw_keys, write_parameters

    case = _case(tmp_path / "case", "KX NORMAL 1 0.1\nKY LOGNORMAL 0 1\n")
    config = parse_case(case)
    run_path = tmp_path / "run"
    run_path.mkdir()

    write_parameters(config, run_path, 2, {"COEF": np.array([0.5, -1.0])})

    kx, ky = (float(k.transform_numpy(np.array([x]))[0])
              for k, x in zip(gen_kw_keys(config, "COEF"), (0.5, -1.0), strict=True))  # fmt: skip
    parameters = json.loads((run_path / "parameters.json").read_text())
    assert parameters["KX"]["value"] == pytest.approx(kx)
    assert parameters["KY"]["value"] == pytest.approx(ky)
    assert "LOG10_COEF:KY" in (run_path / "parameters.txt").read_text()
    assert (run_path / "props.inc").read_text() == f"MULTX\n {kx:g} /\nMULTX\n {ky:g} /\n"
