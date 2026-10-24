"""Optional diagnostics: nothing is saved unless requested; plugin switches win."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.diagnostics import Item, RecordingDiagnostics
from apex.core.errors import ConfigError
from apex.services import composition
from apex.storage.run_results import RunResults


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    variances: np.ndarray


CaseFactory = Callable[..., Case]

ESMDA = 'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
GIES = 'name = "gies"\nschedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 4\n'
STRATEGY = {"esmda": "approx", "gies": "rlmmac"}


def _method(head: str, strategy: str = "", localization: str | None = None) -> str:
    """A method section; ``strategy``/``localization``: extra inline-table entries."""
    name = STRATEGY["gies" if "gies" in head else "esmda"]
    text = f'{head}strategy = {{ name = "{name}", options = {{ energy = 1.0 }}{strategy} }}'
    if localization is not None:
        text += (
            '\nlocalization = { name = "adaptive_prior", options = { taper = "soft" }'
            f"{localization} }}"
        )
    return text


def _run(linear_case: CaseFactory, method: str, output: str = "", name: str = "case") -> RunResults:
    case = linear_case(name=name, ensemble_size=24, method=method, failures=f"[output]\n{output}")
    composition.start(case.config)
    return composition.load_results(case.directory / "runs" / name)


def _saved(results: RunResults) -> dict[int, set[str]]:
    return {k: set(results.diagnostics(k)[0]) for k in results.diagnostics_iterations()}


def test_nothing_is_saved_by_default(linear_case: CaseFactory) -> None:
    results = _run(linear_case, _method(ESMDA, localization=""))
    assert results.diagnostics_iterations() == []
    assert not (results.path / "checkpoints" / "diagnostics_plan.json").exists()
    with pytest.raises(FileNotFoundError, match="only when requested"):
        results.diagnostics(1)


def test_output_names_save_every_plugin_that_produces_them(linear_case: CaseFactory) -> None:
    results = _run(
        linear_case,
        _method(ESMDA, localization=""),
        'diagnostics = ["taper", "strategy.singular_values"]',
    )
    assert _saved(results) == {
        k: {"localization.taper", "strategy.singular_values"} for k in (1, 2, 3, 4)
    }
    items, index = results.diagnostics(1)
    assert items["localization.taper"].shape == (3, 8)
    assert index["localization.taper"] == "saved ([output])"
    plan = json.loads((results.path / "checkpoints" / "diagnostics_plan.json").read_text())
    assert {row["item"] for row in plan["items"]} == {
        "localization.taper",
        "strategy.singular_values",
    }


def test_the_saved_gain_is_the_update_of_that_step(linear_case: CaseFactory) -> None:
    # Without localization or bounds, ES-MDA step 1 is X_1 - X_0 = K L^-1 (D - Y), with
    # K the saved gain, L = sqrt(alpha C_d) (alpha = 4, C_d diagonal), D the step's
    # perturbed observations and Y the prior forecast.
    case = linear_case(
        ensemble_size=24, method=_method(ESMDA), failures='[output]\ndiagnostics = ["gain", "step"]'
    )
    composition.start(case.config)
    results = composition.load_results(case.directory / "runs" / "case")
    items, _ = results.diagnostics(1)
    prior, first = results.iterate(0), results.iterate(1)
    step = first.state["x"] - prior.state["x"]
    np.testing.assert_allclose(items["method.step"], step, rtol=1e-12, atol=1e-14)
    assert first.perturbed_observations is not None
    whitened = (first.perturbed_observations - prior.predicted) / np.sqrt(4.0 * case.variances)[
        :, None
    ]
    np.testing.assert_allclose(items["strategy.gain"] @ whitened, step, rtol=1e-10, atol=1e-12)


def test_a_plugin_switch_overrides_output_both_ways(
    linear_case: CaseFactory, caplog: pytest.LogCaptureFixture
) -> None:
    method = _method(
        ESMDA,
        strategy=", diagnostics = { gain = true }",
        localization=", diagnostics = { taper = false }",
    )
    with caplog.at_level(logging.INFO):
        results = _run(linear_case, method, 'diagnostics = ["taper"]')
    assert _saved(results)[1] == {"strategy.gain"}
    assert results.diagnostics(1)[1]["strategy.gain"] == "saved (plugin)"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("taper = false overrides [output]" in r.getMessage() for r in warnings)
    (plan,) = [r for r in caplog.records if getattr(r, "event", None) == "diagnostics_plan"]
    assert "strategy.gain (plugin)" in plan.getMessage()
    saved_plan = json.loads((results.path / "checkpoints" / "diagnostics_plan.json").read_text())
    assert saved_plan["overrides"] == [r.getMessage() for r in warnings]


def test_a_qualified_name_selects_one_role(linear_case: CaseFactory) -> None:
    results = _run(linear_case, _method(GIES), 'diagnostics = ["strategy.singular_values"]')
    saved = _saved(results)
    assert saved
    assert all(names == {"strategy.singular_values"} for names in saved.values())


def test_plugin_limits_and_interval_override_output(linear_case: CaseFactory) -> None:
    method = _method(
        ESMDA,
        strategy=", diagnostics = { gain = true, every = 2 }",
        localization=", diagnostics = { taper = true, max_mb = 1e-6 }",
    )
    results = _run(linear_case, method)
    # The gain every second update; the too-large taper is noted as skipped each time.
    assert {k: names for k, names in _saved(results).items() if names} == {
        2: {"strategy.gain"},
        4: {"strategy.gain"},
    }
    _, index = results.diagnostics(2)
    assert index["localization.taper"].startswith("skipped:")
    assert "max_mb" in index["localization.taper"]


def test_an_unrequested_gain_is_never_formed(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    formed: list[str] = []
    add = RecordingDiagnostics.add

    def spy(
        self: RecordingDiagnostics, role: str, name: str, item: Item, *, nbytes: int | None = None
    ) -> None:
        def counted() -> NDArray[Any]:
            formed.append(f"{role}.{name}")
            return item() if callable(item) else item

        add(self, role, name, counted, nbytes=nbytes)

    monkeypatch.setattr(RecordingDiagnostics, "add", spy)
    _run(linear_case, _method(GIES), 'diagnostics = ["singular_values"]')
    assert "strategy.singular_values" in formed
    assert "strategy.gain" not in formed
    _run(linear_case, _method(GIES), 'diagnostics = ["gain"]', name="with_gain")
    assert "strategy.gain" in formed


@pytest.mark.parametrize(
    ("output", "method", "message"),
    [
        ('diagnostics = ["no_such"]', _method(ESMDA), "no configured plugin produces"),
        ('diagnostics = ["localization.taper"]', _method(ESMDA), "localization"),
        ("", _method(ESMDA, strategy=", diagnostics = { gian = true }"), "gian"),
    ],
)
def test_unknown_diagnostics_are_rejected(
    linear_case: CaseFactory, output: str, method: str, message: str
) -> None:
    case = linear_case(ensemble_size=24, method=method, failures=f"[output]\n{output}")
    with pytest.raises(ConfigError, match=message):
        composition.start(case.config)


def test_resumed_runs_save_the_same_diagnostics(linear_case: CaseFactory) -> None:
    output = 'diagnostics = ["taper", "step"]'
    whole = _run(linear_case, _method(ESMDA, localization=""), output)
    case = linear_case(
        name="resumed",
        ensemble_size=24,
        method=_method(ESMDA, localization=""),
        failures=f"[output]\n{output}",
    )
    composition.start(case.config, max_steps=2)
    composition.resume(case.directory / "runs" / "resumed")
    resumed = composition.load_results(case.directory / "runs" / "resumed")
    assert resumed.diagnostics_iterations() == whole.diagnostics_iterations()
    for k in whole.diagnostics_iterations():
        for key, array in whole.diagnostics(k)[0].items():
            np.testing.assert_array_equal(resumed.diagnostics(k)[0][key], array)
