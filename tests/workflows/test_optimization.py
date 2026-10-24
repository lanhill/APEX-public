"""Optimization runs through the composition: checkpoints, resume and refusals."""

import json
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError, WorkflowError
from apex.services import optimization
from apex.storage import optimization_directory as layout

CONFIG = """
[optimization]
name = "{name}"
seed = 11

[controls]
realizations = 1

[[controls.variables]]
name = "x"
initial = [0.5, 0.5]
lower = -2.0
upper = 2.0
variance = 0.01

[[controls.variables]]
name = "y"
initial = 0.25
lower = -2.0
upper = 2.0
variance = 0.01

[objective]
name = "{objective}"

[optimizer]
name = "{optimizer}"
options = {options}
"""


def _config(
    tmp_path: Path,
    name: str = "run",
    optimizer: str = "enopt",
    options: str = "{ ne = 8, maxiter = 25, resample = 1, ensemble_hessian = true }",
    objective: str = "rosenbrock",
) -> Path:
    path = tmp_path / f"{name}.toml"
    path.write_text(
        CONFIG.format(name=name, optimizer=optimizer, options=options, objective=objective),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("optimizer", "options"),
    [
        ("enopt", "{ ne = 8, maxiter = 25, resample = 1, ensemble_hessian = true }"),
        ("smcopt", "{ ne = 8, maxiter = 15, survival_factor = 0.5 }"),
        ("genopt", "{ ne = 8, maxiter = 15, cma = true }"),
        ("line_search", '{ method = "BFGS", maxiter = 30 }'),
        ("ga", "{ pop_size = 8, maxiter = 10, transform = true }"),
        ("pso", "{ pop_size = 8, maxiter = 10, transform = true }"),
    ],
)
def test_resumed_runs_equal_uninterrupted_runs(
    tmp_path: Path, optimizer: str, options: str
) -> None:
    full = optimization.optimize(_config(tmp_path, "full", optimizer, options))
    assert full.finished
    first = optimization.optimize(_config(tmp_path, "split", optimizer, options), max_iterations=3)
    assert not first.finished
    assert not (first.run_dir / layout.RESULT).exists()
    resumed = optimization.resume(first.run_dir, max_iterations=2)
    while not resumed.finished:
        resumed = optimization.resume(first.run_dir)
    np.testing.assert_array_equal(resumed.result.x, full.result.x)
    assert resumed.result.nfev == full.result.nfev
    full_history = layout.read_history(full.run_dir)
    assert layout.read_history(first.run_dir) == full_history
    assert full_history[0]["kind"] == "start"
    assert all(row["kind"] == "step" for row in full_history[1:])
    result = json.loads((full.run_dir / layout.RESULT).read_text())
    assert result["names"] == ["x[0]", "x[1]", "y"]
    assert result["x"] == pytest.approx(list(full.result.x))


def test_an_interrupted_checkpoint_resumes_from_the_last_row(tmp_path: Path) -> None:
    """History rows written after the last checkpoint are dropped on resume."""
    run = optimization.optimize(_config(tmp_path), max_iterations=2)
    layout.append_history(run.run_dir, {"kind": "step", "iteration": 99})
    resumed = optimization.resume(run.run_dir, max_iterations=1)
    rows = layout.read_history(resumed.run_dir)
    assert [row["iteration"] for row in rows] == [0, 1, 2, 3]


def test_a_second_run_in_the_same_directory_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path)
    optimization.optimize(config, max_iterations=1)
    with pytest.raises(WorkflowError, match="already holds"):
        optimization.optimize(config)
    with pytest.raises(ConfigError, match="no optimization run"):
        optimization.resume(tmp_path / "nowhere")


@pytest.mark.parametrize(
    ("optimizer", "options", "match"),
    [
        ("enopt", "{ ne = 8, speed = 2 }", "speed"),
        ("annealing", "{}", "unknown optimizers plugin"),
        ("trust_region", "{ epf = { r = 1.0 } }", "cannot use EPF"),
    ],
)
def test_bad_configurations_are_refused(
    tmp_path: Path, optimizer: str, options: str, match: str
) -> None:
    objective = "quadratic" if optimizer == "trust_region" else "rosenbrock"
    with pytest.raises(ConfigError, match=match):
        optimization.optimize(
            _config(tmp_path, optimizer=optimizer, options=options, objective=objective)
        )


def test_ensemble_optimizers_need_variances(tmp_path: Path) -> None:
    path = _config(tmp_path)
    path.write_text(path.read_text().replace("variance = 0.01\n", "", 1))
    with pytest.raises(ConfigError, match="variance"):
        optimization.optimize(path)


def test_the_examples_validate() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples" / "optimization"
    for path in sorted(examples.glob("*.toml")):
        if path.name == "example.toml":  # the catalogue's manifest, not a configuration
            continue
        config = optimization.load_optimization_config(path)
        optimization.build(config)


@pytest.mark.parametrize("optimizer", ["ga", "pso"])
def test_a_constrained_population_run_resumes_through_its_checkpoint_files(
    tmp_path: Path, optimizer: str
) -> None:
    """The population, its constraints and the EPF state survive the checkpoint
    files (npz and JSON), not only an in-memory state."""
    options = (
        '{ pop_size = 6, maxiter = 3, transform = true, constraint_handling = "epf", '
        "epf = { r = 10.0, max_epf_iter = 3, conv_crit = 1e-9 } }"
    )
    full = optimization.optimize(_config(tmp_path, "full", optimizer, options, "quadratic"))
    first = optimization.optimize(
        _config(tmp_path, "split", optimizer, options, "quadratic"), max_iterations=2
    )
    resumed = optimization.resume(first.run_dir, max_iterations=1)
    while not resumed.finished:
        resumed = optimization.resume(first.run_dir)
    np.testing.assert_array_equal(resumed.result.x, full.result.x)
    assert layout.read_history(first.run_dir) == layout.read_history(full.run_dir)
