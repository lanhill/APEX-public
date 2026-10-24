"""History matching a plurigaussian facies case end to end.

The latent fields are drawn by ``gstools_pgs`` (conditioned to one cell's facies), mapped
by ``pgs_truncation`` to facies and porosity, and the linear simulator reads the porosity.
The truth is the true latent fields; the metrics compare facies and porosity.
"""

from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.facies import LithotypeRule, PGSPrior
from apex.core.errors import ConfigError
from apex.domain.state import StateVariable
from apex.services import composition
from apex.storage.run_directory import RunDirectory

GRID = (8, 6, 1)
CELLS = 48
CODES = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 1]])
N_DATA = 12
CONDITIONING_CELL = (3, 2, 1)  # 1-based Eclipse (i, j, k)
FIELDS = '[{ model = "Gaussian", len_scale = [4, 3] }, { model = "Exponential", len_scale = 3 }]'


def _case(directory: Path, name: str, method: str, truncation: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "rule.npy", CODES)
    rng = np.random.default_rng(42)
    truth = PGSPrior(
        [{"model": "Gaussian", "len_scale": [4, 3]}, {"model": "Exponential", "len_scale": 3}],
        rule_file=directory / "rule.npy",
    ).generate_joint(
        [StateVariable(n, CELLS, grid=GRID) for n in ("y1", "y2")], 1, rng
    )  # fmt: skip
    np.savez(directory / "truth.npz", y1=truth["y1"][:, 0], y2=truth["y2"][:, 0])
    rule = LithotypeRule.on_box(CODES, 3.0)
    facies = rule.facies(np.stack([truth["y1"][:, 0], truth["y2"][:, 0]]))
    poro = np.array([0.05, 0.15, 0.30])[facies]
    forward = rng.standard_normal((N_DATA, CELLS))
    np.savez(directory / "forward.npz", matrix=forward)
    variances = np.full(N_DATA, 1e-3)
    observed = forward @ poro + np.sqrt(variances) * rng.standard_normal(N_DATA)
    rows = ["time,datatype,value,variance"] + [
        f"t{i},y,{float(observed[i])!r},{float(variances[i])!r}" for i in range(N_DATA)
    ]
    (directory / "observations.csv").write_text("\n".join(rows) + "\n")
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(N_DATA))
    i, j, k = CONDITIONING_CELL
    code = int(facies[(i - 1) + GRID[0] * (j - 1)])
    truncation = truncation or (
        '{ name = "pgs_truncation", options = { fields = ["y1", "y2"], '
        'facies = ["shale", "silt", "sand"], rule_file = "rule.npy", values = { '
        "shale = { poro = 0.05 }, silt = { poro = 0.15 }, sand = { poro = 0.30 } } } }"
    )
    config = directory / f"{name}.toml"
    config.write_text(
        f"""
[run]
name = "{name}"
seed = 7

[[state.variables]]
name = "y1"
size = {CELLS}
grid = [{GRID[0]}, {GRID[1]}, {GRID[2]}]

[[state.variables]]
name = "y2"
size = {CELLS}
grid = [{GRID[0]}, {GRID[1]}, {GRID[2]}]

[prior]
ensemble_size = 30

[prior.groups.facies]
variables = ["y1", "y2"]
name = "gstools_pgs"
options.fields = {FIELDS}
options.rule_file = "rule.npy"
options.gibbs_sweeps = 20
options.conditioning = {{ cells = [[{i}, {j}, {k}]], facies = [{code}] }}

[[parameterizations]]
{truncation.removeprefix("{ ").removesuffix(" }").replace(", options = ", "\noptions = ")}

[observations]
file = "observations.csv"

[case]
truth_file = "truth.npz"

[simulator]
name = "linear"
options = {{ forward_file = "forward.npz", variables = ["poro"], outputs = [{outputs}] }}

[method]
{method}
"""
    )
    return config


ESMDA = (
    'name = "esmda"\noptions = { inflation = [2, 2] }\n'
    'strategy = { name = "approx", options = { energy = 0.99 } }'
)


def test_a_facies_case_matches_its_data_and_resumes_identically(tmp_path: Path) -> None:
    whole = composition.start(_case(tmp_path / "whole", "whole", ESMDA))
    composition.start(_case(tmp_path / "parts", "parts", ESMDA), max_steps=1)
    resumed = composition.resume(tmp_path / "parts" / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, whole.posterior)
    assert resumed.history == whole.history
    first, last = whole.history[0], whole.history[-1]
    assert last.data_mismatch_mean < first.data_mismatch_mean
    # The facies metric is recorded by default, on the properties, and improves.
    run = RunDirectory(tmp_path / "whole" / "runs" / "whole")
    prior_scores = run.load_arrays("data_mismatch_0")
    final_scores = run.load_arrays(f"data_mismatch_{len(whole.history) - 1}")
    prior_facies = prior_scores["metric:facies_mismatch:facies"].mean()
    assert final_scores["metric:facies_mismatch:facies"].mean() < prior_facies
    assert "metric:rmse:poro" in prior_scores  # RMSE of a property, not of a latent field
    assert "metric:rmse:facies" not in prior_scores  # codes are scored by facies_mismatch


def test_every_prior_member_honours_the_conditioning(tmp_path: Path) -> None:
    config = _case(tmp_path, "cond", ESMDA)
    run = composition.build_run(composition.load_config(config))
    prior = run.workflow.layout.split(composition.draw_prior(run))
    properties = run.workflow.property_map.ensemble(prior)
    i, j, _ = CONDITIONING_CELL
    cell = (i - 1) + GRID[0] * (j - 1)
    assert np.unique(properties["facies"][cell]).size == 1


def test_a_truncation_with_another_rule_is_refused(tmp_path: Path) -> None:
    other = (
        '{ name = "pgs_truncation", options = { fields = ["y1", "y2"], '
        'facies = ["shale", "silt", "sand"], rule_file = "rule.npy", extent = 2.0, values = { '
        "shale = { poro = 0.05 }, silt = { poro = 0.15 }, sand = { poro = 0.30 } } } }"
    )
    config = _case(tmp_path, "bad", ESMDA, truncation=other)

    with pytest.raises(ConfigError, match="different rules"):
        composition.build_run(composition.load_config(config))


def test_the_adjoint_update_is_refused_with_a_parameterization(tmp_path: Path) -> None:
    method = (
        'name = "esmda"\noptions = { inflation = [1] }\n'
        'strategy = { name = "approx", options = { sensitivities = true } }'
    )
    with pytest.raises(ConfigError, match="gradients with respect to the state"):
        composition.build_run(composition.load_config(_case(tmp_path, "adj", method)))
