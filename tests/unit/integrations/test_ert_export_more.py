"""Export (A1) of the cases C3 adds: scalars, Eclipse, parameterizations, transforms."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.core.plugins import data_transforms, parameterizations
from apex.domain.observations import DataKey, ObservationSet
from apex.integrations.ert import transform_step
from apex.integrations.ert.export import export_case, read_scalars
from apex.integrations.ert.render import render
from apex.services import composition
from apex.simulation.grdecl import write_grdecl

pytest.importorskip("surfio")  # ERT's surface library (the `ert` extra)

CaseFactory = Callable[..., Path]

SCALAR = """
[[state.variables]]
name = "mult"
size = 2
"""


class Logistic:
    """Maps the latent field ``z`` to ``poro = 0.05 + 0.35 / (1 + exp(-z))``."""

    variables = ("z",)
    outputs = ("poro",)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        return {"poro": 0.05 + 0.35 / (1 + np.exp(-np.asarray(state["z"])))}


class Doubled:
    """A scalar data transform: every datum doubled (variances times 4)."""

    output_datatype = "doubled"

    def fit(self, observations: ObservationSet) -> ObservationSet:
        keys = [DataKey(k.time, f"doubled({k.datatype})", k.index) for k in observations.keys]
        return ObservationSet(keys, 2 * observations.values, 4 * observations.variances)

    def apply(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * predicted


@pytest.fixture
def plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(parameterizations._factories, "test_logistic", Logistic)
    monkeypatch.setitem(data_transforms._factories, "test_doubled", Doubled)


def _edit(config: Path, old: str, new: str) -> Path:
    text = config.read_text()
    assert old in text
    config.write_text(text.replace(old, new, 1))
    return config


def test_scalar_state_travels_as_one_surface(tmp_path: Path, tiny_case: CaseFactory) -> None:
    config = _edit(tiny_case(), "[prior]", SCALAR + "\n[prior]")
    out = tmp_path / "ert"

    result = export_case(config, out)

    text = result.case.read_text()
    assert (
        "SURFACE APEXSCAL OUTPUT_FILE:apexscal.irap INIT_FILES:prior/APEXSCAL_%d.irap "
        "BASE_SURFACE:prior/APEXSCAL_0.irap FORWARD_INIT:False\n"
    ) in text
    run = composition.build_run(composition.load_config(config))
    prior = run.workflow.layout.split(composition.draw_prior(run))
    for member in range(5):
        np.testing.assert_allclose(
            read_scalars(out / "prior" / f"APEXSCAL_{member}.irap"),
            prior["mult"][:, member].astype(np.float32),  # ERT's surfaces are float32
            rtol=0,
        )
    manifest = json.loads((out / "apex_export.json").read_text())
    assert manifest["scalars"] == {"file": "apexscal.irap", "variables": [["mult", 2]]}
    assert set(manifest["fields"]) == {"PORO", "PERMX"}
    # The render step reads the scalars back for the deck.
    (tmp_path / "deck.mako").write_text("MULT ${mult[0]} ${mult[1]}\n")
    run_path = tmp_path / "rp"
    run_path.mkdir()
    write_grdecl(run_path / "poro.grdecl", "PORO", np.full(12, 0.2))
    write_grdecl(run_path / "permx.grdecl", "PERMX", np.full(12, 5.0))
    (run_path / "apexscal.irap").write_bytes((out / "prior" / "APEXSCAL_3.irap").read_bytes())
    render(out / "apex_export.json", Path("CASE.DATA"), 3, run_path)
    values = [float(v) for v in (run_path / "CASE.DATA").read_text().split()[1:]]
    np.testing.assert_allclose(values, prior["mult"][:, 3].astype(np.float32), rtol=1e-6)


def test_an_eclipse_case_runs_erts_eclipse_step(tmp_path: Path, tiny_case: CaseFactory) -> None:
    config = _edit(
        tiny_case(simulator="eclipse"),
        'template_values = { label = "tiny" } }',
        'template_values = { label = "tiny" }, eclrun_args = ["--version=2023.1"], licenses = 2 }',
    )

    text = export_case(config, tmp_path / "ert").case.read_text()

    assert "FORWARD_MODEL ECLIPSE100(<VERSION>=2023.1, <NUM_CPU>=1)\n" in text
    assert "QUEUE_OPTION LOCAL MAX_RUNNING 2" in text  # the licences cap the 3 workers
    assert "FORWARD_MODEL FLOW" not in text
    unversioned = tmp_path / "unversioned"
    unversioned.mkdir()
    bare = tiny_case(simulator="eclipse")
    with pytest.raises(ConfigError, match="needs the version"):
        export_case(bare, unversioned / "ert")


def test_a_parameterized_case_exports_the_state_and_renders_the_properties(
    tmp_path: Path, tiny_case: CaseFactory, plugins: None
) -> None:
    config = _edit(
        tiny_case(),
        'name = "poro"\nsize = 12\ngrid = [4, 3, 1]\nlower = 0.05\nupper = 0.4\n',
        'name = "z"\nsize = 12\ngrid = [4, 3, 1]\n',
    )
    _edit(
        config, "[observations]", '[[parameterizations]]\nname = "test_logistic"\n\n[observations]'
    )
    _edit(config, "[prior]\nensemble_size = 5", "[prior]\nensemble_size = 5")
    out = tmp_path / "ert"

    export_case(config, out)

    manifest = json.loads((out / "apex_export.json").read_text())
    assert manifest["parameterized"] is True
    assert set(manifest["fields"]) == {"Z", "PERMX"}
    run_path = tmp_path / "rp"
    run_path.mkdir()
    z = np.linspace(-2, 2, 12)
    write_grdecl(run_path / "z.grdecl", "Z", z)
    write_grdecl(run_path / "permx.grdecl", "PERMX", np.full(12, 5.0))
    render(out / "apex_export.json", Path("CASE.DATA"), 0, run_path)
    # The tiny deck: "-- a tiny deck", PORO, its values, "/".
    deck = (run_path / "CASE.DATA").read_text().split("/")[0].split()[5:]
    np.testing.assert_allclose([float(v) for v in deck], 0.05 + 0.35 / (1 + np.exp(-z)), rtol=1e-6)


def test_transformed_data_are_assimilated_as_gen_data(
    tmp_path: Path, tiny_case: CaseFactory, plugins: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _edit(
        tiny_case(),
        "[observations]",
        '[[data_transforms]]\nname = "test_doubled"\n\n[observations]',
    )
    out = tmp_path / "ert"

    text = export_case(config, out).case.read_text()

    assert "GEN_DATA APEX_TRANSFORMED RESULT_FILE:apex_transformed_%d.txt REPORT_STEPS:0\n" in text
    assert "FORWARD_MODEL APEX_TRANSFORM(<EXPORT>=<CONFIG_PATH>/apex_export.json" in text
    for key in ("WOPR:P1", "FOPR"):  # the raw keys are still simulated, for the step
        assert f"SUMMARY {key}\n" in text
    assert "GENERAL_OBSERVATION APEX_TRANSFORMED" in (out / "observations" / "obs.txt").read_text()
    rows = (out / "observations" / "transformed.txt").read_text().split("\n")[:2]
    assert [float(x) for x in rows[0].split()] == pytest.approx([200.0, 4.0])  # 2*100, 2*sqrt(4)
    # The step applies the fitted transform to the run path's raw summary.
    raw = np.arange(8, dtype=float)
    monkeypatch.setattr(transform_step, "read_summary", lambda case, keys: raw)
    run_path = tmp_path / "rp"
    run_path.mkdir()
    transform_step.transform(out / "apex_export.json", Path("apex_transformed_0.txt"), run_path)
    written = [float(v) for v in (run_path / "apex_transformed_0.txt").read_text().split()]
    assert written == list(2 * raw)


def test_cases_export_cannot_take_are_refused(
    tmp_path: Path, tiny_case: CaseFactory, plugins: None
) -> None:
    with pytest.raises(ConfigError, match="opm_flow and eclipse"):
        export_case(tiny_case(simulator="linear"), tmp_path / "a")
    both = _edit(
        tiny_case(validation='[validation]\ncv = { wells = ["P2"] }\n', method='name = "lm_enrml"'),
        "[observations]",
        '[[data_transforms]]\nname = "test_doubled"\n\n[observations]',
    )
    with pytest.raises(ConfigError, match="held-out rows"):
        export_case(both, tmp_path / "b")
