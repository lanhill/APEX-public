"""A prediction's control scenario with a surrogate (decision 6, §3).

The model is built over an optimization-shaped case of the forecast: the linear case's
forecast "deck" ``y = G_pred @ [x; u]`` with two report times beyond the history, its
controls ``u`` with bounds, and the run's posterior as ``[model]`` realizations. ``G_pred``'s
history rows are the run's ``G`` with zero control columns, so the history check holds. An
affine model is exact here: the prediction by the surrogate must be the simulator's to
round-off, also when the scenario names its control otherwise and maps it onto the deck's,
and with the model copied out of its build directory; and the refusals of §3: no scenario,
a model built without controls, other inputs, and a control outside the design's range as
the model sees it (after the scenario's parameterizations).
"""

import importlib.util
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.core.plugins import surrogate_models
from apex.services import composition, prediction, surrogates


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_controls", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()
AFFINE = "test_affine_controls"
TIMES = (*cases.TIMES, "5", "6")
ROUND_OFF = 1e-9
SCENARIO = [0.5, -1.0]


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register(AFFINE)(cases.Affine)
    surrogates.fitted_model.cache_clear()
    try:
        yield
    finally:
        surrogate_models.remove(AFFINE)


def _outputs() -> str:
    return ", ".join(f'["{t}", "{d}"]' for d in cases.DATATYPES for t in TIMES)


def _simulator_options() -> str:
    return f'forward_file = "forward_pred.npz", variables = ["x", "u"], outputs = [{_outputs()}]'


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A finished ES-MDA run of the linear case, its forecast deck and posterior."""
    composition.start(cases.write_case(tmp_path, members=40))
    forward = cases.forward_matrix()
    observed = cases.keys_in_order()
    rng = np.random.default_rng(5)
    rows = []
    for datatype in cases.DATATYPES:
        for time in TIMES:
            key = (time, datatype)
            if key in observed:
                rows.append(np.concatenate([forward[observed.index(key)], np.zeros(2)]))
            else:
                rows.append(rng.normal(size=5))
    np.savez(tmp_path / "forward_pred.npz", matrix=np.array(rows))
    results = composition.load_results(tmp_path / "runs" / "hm")
    posterior = np.asarray(results.iterate(results.posterior_iteration).state["x"])
    np.savez(tmp_path / "x_post.npz", x=posterior)
    return tmp_path


def _forecast_case(
    folder: Path, *, name: str = "forecast", realizations: str = "x_post.npz"
) -> None:
    """The optimization-shaped case of the forecast: its deck, controls and realizations."""
    (folder / f"{name}.toml").write_text(f"""
[optimization]
name = "{name}"
seed = 5

[[controls.variables]]
name = "u"
initial = [0.0, 0.0]
lower = -2.0
upper = 2.0
variance = 0.1

[objective]
name = "npv"
options = {{ start = "2000-01-01", times = ["2001-01-01"], wop = 1.0, wwi = 1.0, disc = 0.0 }}

[optimizer]
name = "enopt"
options = {{ ne = 4, maxiter = 1 }}

[simulator]
name = "linear"
options = {{ {_simulator_options()} }}

[execution]
name = "serial"

[model]
realizations_file = "{realizations}"
""")


def _build(folder: Path, case: str = "forecast.toml", name: str = "fc") -> Path:
    vectors = ", ".join(f'"{d}"' for d in cases.DATATYPES)
    times = ", ".join(f'"{t}"' for t in TIMES)
    path = folder / f"build_{name}.toml"
    path.write_text(f"""
[surrogate]
name = "{name}"
model = "{AFFINE}"
seed = 7

[case]
config_file = "{case}"
outputs = {{ vectors = [{vectors}], times = [{times}] }}

[design]
plan = "lhs"
training = 30
held_out = 10
""")
    result = surrogates.build(path)
    assert result.summary.gate is not None
    assert result.summary.gate.passed, result.summary.gate.failures()
    return result.path / "model" / "surrogate.json"


def _prediction(folder: Path, name: str, *, controls: str = "", surrogate: str = "") -> Path:
    path = folder / f"{name}.toml"
    path.write_text(f"""
[prediction]
name = "{name}"
run_dir = "runs/hm"
ensembles = ["posterior", "truth"]

[prediction.outputs]
vectors = ["WOPR:*", "WBHP:*"]

[simulator]
options = {{ {_simulator_options()} }}

{controls}

{surrogate}
""")
    return path


def _scenario(values: list[float], name: str = "u") -> str:
    return f'[[controls.variables]]\nname = "{name}"\nvalues = {json.dumps(values)}\n'


def _mapped(values: list[float]) -> str:
    """A scenario of the control ``w``, held over two steps as the deck's ``u``."""
    return (
        _scenario(values, name="w") + "\n[[controls.parameterizations]]\nname = "
        '"control_schedule"\noptions = { variable = "w", output = "u", knots = [0, 1], '
        "n_steps = 2 }\n"
    )


def _surrogate(model: Path) -> str:
    return f'[surrogate]\nmodel_file = "{model}"\n'


def test_a_scenario_by_a_surrogate_built_over_controls_is_the_simulators(run: Path) -> None:
    _forecast_case(run)
    model = _build(run)
    plain = prediction.predict(_prediction(run, "simulated", controls=_scenario(SCENARIO)))
    fast = prediction.predict(
        _prediction(run, "fast", controls=_scenario(SCENARIO), surrogate=_surrogate(model))
    )
    expected = plain.forecasts["posterior"].values
    np.testing.assert_allclose(
        fast.forecasts["posterior"].values, expected,
        rtol=0, atol=ROUND_OFF * float(np.max(np.abs(expected))),
    )  # fmt: skip
    # The scenario changes the forecast: another one gives other values beyond the history.
    other = prediction.predict(
        _prediction(run, "other", controls=_scenario([-1.5, 1.5]), surrogate=_surrogate(model))
    )
    assert not np.allclose(other.forecasts["posterior"].values, expected)
    provenance = json.loads((run / "runs/hm/predictions/fast/provenance.json").read_text())
    assert provenance["surrogate"]["identity"]
    # The same scenario named otherwise and mapped onto the deck's u: the same inputs (the
    # review: refused by a check on the controls' names).
    renamed = prediction.predict(
        _prediction(run, "renamed", controls=_mapped(SCENARIO), surrogate=_surrogate(model))
    )
    np.testing.assert_array_equal(
        renamed.forecasts["posterior"].values, fast.forecasts["posterior"].values
    )
    # A model copied on its own (the manifest records its inputs; no snapshots needed).
    copied = run / "copied"
    shutil.copytree(model.parent, copied)
    shutil.rmtree(model.parent.parent)
    again = prediction.predict(
        _prediction(run, "copied", controls=_scenario(SCENARIO),
                    surrogate=_surrogate(copied / "surrogate.json"))
    )  # fmt: skip
    np.testing.assert_array_equal(
        again.forecasts["posterior"].values, fast.forecasts["posterior"].values
    )


def test_what_a_scenario_with_a_surrogate_refuses(run: Path) -> None:
    _forecast_case(run)
    model = _build(run)
    with pytest.raises(ConfigError, match="built over controls"):
        prediction.predict(_prediction(run, "none", surrogate=_surrogate(model)))
    with pytest.raises(ConfigError, match="outside the range"):
        prediction.predict(
            _prediction(run, "far", controls=_scenario([3.0, 0.0]), surrogate=_surrogate(model))
        )
    # A model built without controls (on the run's own case): the scenario would be ignored.
    built = surrogates.build(cases.write_build(run, AFFINE))
    with pytest.raises(ConfigError, match="built without controls"):
        prediction.predict(_prediction(
            run, "blind", controls=_scenario(SCENARIO),
            surrogate=_surrogate(built.path / "model" / "surrogate.json"),
        ))  # fmt: skip
    # Within the controls' bounds as given, outside the design's range as the model sees it.
    with pytest.raises(ConfigError, match=r"u\[1\] = 3 \(after its parameterizations\)"):
        prediction.predict(
            _prediction(run, "mapped", controls=_mapped([0.5, 3.0]), surrogate=_surrogate(model))
        )
    # Built on other inputs: realizations with one more variable than the run's state.
    posterior = np.load(run / "x_post.npz")["x"]
    np.savez(run / "xz_post.npz", x=posterior, z=np.ones((1, posterior.shape[1])))
    _forecast_case(run, name="extra", realizations="xz_post.npz")
    extra = _build(run, case="extra.toml", name="extra")
    with pytest.raises(ConfigError, match="was fitted on inputs"):
        prediction.predict(
            _prediction(run, "inputs", controls=_scenario(SCENARIO), surrogate=_surrogate(extra))
        )
    # The same inputs, but a realization of the model set as a control (the coder's review).
    with pytest.raises(ConfigError, match="takes 'z' as a realization"):
        prediction.predict(_prediction(
            run, "realization",
            controls=_scenario(SCENARIO) + "\n" + _scenario([1.0], name="z"),
            surrogate=_surrogate(extra),
        ))  # fmt: skip
    # Built over controls on another deck (another forward model): another stack.
    matrix = np.load(run / "forward_pred.npz")["matrix"]
    np.savez(run / "forward_other.npz", matrix=matrix * 1.5)
    other = (run / "forecast.toml").read_text().replace("forward_pred.npz", "forward_other.npz")
    (run / "other.toml").write_text(other.replace('name = "forecast"', 'name = "other"'))
    deck = _build(run, case="other.toml", name="other")
    with pytest.raises(ConfigError, match="another simulator stack"):
        prediction.predict(
            _prediction(run, "deck", controls=_scenario(SCENARIO), surrogate=_surrogate(deck))
        )


def _older(model: Path, folder: Path, drop: tuple[str, ...]) -> Path:
    """A copy of ``model``'s manifest without ``drop`` (a model written before them)."""
    document = json.loads(model.read_text())
    for name in drop:
        document.pop(name, None)
    target = folder / "surrogate.json"
    folder.mkdir()
    target.write_text(json.dumps(document))
    shutil.copy(model.with_suffix(".npz"), folder / model.with_suffix(".npz").name)
    return target


def test_older_models_built_over_controls(run: Path) -> None:
    """Built before the manifest recorded its inputs: checked against the build's snapshots
    (beside it); built before the controls' box: refused, saying to rebuild (the review)."""
    _forecast_case(run)
    model = _build(run)
    expected = (
        prediction.predict(
            _prediction(run, "now", controls=_scenario(SCENARIO), surrogate=_surrogate(model))
        )
        .forecasts["posterior"]
        .values
    )
    # Beside its build's snapshots (../snapshots.npz from the model's folder).
    inputs_unrecorded = _older(model, model.parent.parent / "older",
                               ("input_sizes", "control_inputs"))  # fmt: skip
    older = prediction.predict(_prediction(
        run, "older", controls=_scenario(SCENARIO), surrogate=_surrogate(inputs_unrecorded),
    ))  # fmt: skip
    np.testing.assert_array_equal(older.forecasts["posterior"].values, expected)
    with pytest.raises(ConfigError, match="outside the range"):
        prediction.predict(_prediction(
            run, "older_far", controls=_scenario([3.0, 0.0]),
            surrogate=_surrogate(inputs_unrecorded),
        ))  # fmt: skip
    # Away from its build (no ../snapshots.npz): the fallback says why it needs them.
    away = _older(model, run / "away", ("input_sizes", "control_inputs"))
    with pytest.raises(ConfigError, match="checked against its build's snapshots, which cannot"):
        prediction.predict(_prediction(
            run, "away", controls=_scenario(SCENARIO), surrogate=_surrogate(away),
        ))  # fmt: skip
    oldest = _older(model, run / "oldest", ("control_box", "input_sizes", "control_inputs"))
    for name, controls in (("o1", _scenario(SCENARIO)), ("o2", "")):
        with pytest.raises(ConfigError, match="before its controls' box was recorded"):
            prediction.predict(_prediction(run, name, controls=controls,
                                           surrogate=_surrogate(oldest)))  # fmt: skip


def test_damaged_recorded_ranges_and_other_cases_are_refused(run: Path) -> None:
    """The coder's review: a recorded control range not of its input's size is refused by
    name (not an IndexError); a manifest recording controls with another case's digest too."""
    _forecast_case(run)
    model = _build(run)
    document = json.loads(model.read_text())
    document["control_inputs"]["u"][0] = [0.0]  # one lower bound for two entries
    damaged = _older(model, run / "damaged", ())
    damaged.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="manifest is damaged"):
        prediction.predict(_prediction(
            run, "damaged", controls=_scenario(SCENARIO), surrogate=_surrogate(damaged),
        ))  # fmt: skip
    document = json.loads(model.read_text())
    document["provenance"]["case_digest"] = "0" * 64
    edited = _older(model, run / "edited", ())
    edited.write_text(json.dumps(document))
    with pytest.raises(ConfigError, match="not built on an optimization's case"):
        prediction.predict(_prediction(
            run, "edited", controls=_scenario(SCENARIO), surrogate=_surrogate(edited),
        ))  # fmt: skip


def test_only_the_controls_mapped_inputs_are_recorded_as_controls() -> None:
    """The coder's review: a realization mapped to another name (ln k to ``permx``) is not a
    control; a control's schedule (``u`` to ``u_steps``) is."""
    from types import SimpleNamespace
    from typing import Any, cast

    from apex.domain.state import StateLayout, StateVariable

    case = SimpleNamespace(
        layout=StateLayout([StateVariable("u", 2), StateVariable("lnk", 4)]),
        realizations={"lnk": np.zeros((4, 1))},
        properties=SimpleNamespace(parameterizations=[
            SimpleNamespace(variables=("lnk",), outputs=("permx",)),
            SimpleNamespace(variables=("u",), outputs=("u_steps",)),
        ]),
        input_layout=StateLayout([StateVariable("permx", 4), StateVariable("u_steps", 6)]),
    )  # fmt: skip
    assert surrogates._control_inputs(cast(Any, case)) == {"u_steps"}
