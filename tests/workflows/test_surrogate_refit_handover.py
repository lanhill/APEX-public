"""A history match's refit takes the options its model hands its refits.

As an optimization's refits do: the fresh plugin is created with the configuration's options
and, over them, the fitted model's ``refit_options`` (e.g. what its fit calibrated and the
configuration keeps), and the refit's manifest records them.
"""

import importlib.util
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from apex.core.plugins import surrogate_models
from apex.services import composition, surrogates
from apex.storage.surrogate_files import read_surrogate, write_surrogate


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_handover", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()
PLUGIN = "test_affine_handover"


class Handing(cases.Affine):  # type: ignore[misc, name-defined]
    """The affine test surrogate, handing its refits a marker (what a fit would keep)."""

    def __init__(self, std: float = 0.0, marker: float = 0.0) -> None:
        super().__init__(std=std)
        self.marker = marker

    def refit_options(self) -> Mapping[str, object]:
        return {"marker": 2.5}


def _biased(model: Path) -> Path:
    """The model with every intercept shifted by 3 sigma (as test_surrogate_history_matching's):
    its checks fail, so it refits."""
    stored = read_surrogate(model)
    fitted = Handing.from_arrays(stored.manifest.model, stored.arrays)
    fitted.weights = fitted.weights.copy()
    fitted.weights[:, -1] += 3.0 * np.sqrt(cases.VARIANCE)
    target = model.parent.parent / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        target, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
    )  # fmt: skip
    return target


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register(PLUGIN)(Handing)
    surrogates.fitted_model.cache_clear()
    try:
        yield
    finally:
        surrogate_models.remove(PLUGIN)


def test_a_history_matchs_refit_takes_the_models_refit_options(tmp_path: Path) -> None:
    cases.write_case(tmp_path, name="base")
    result = surrogates.build(cases.write_build(tmp_path, PLUGIN, case="base.toml"))
    assert result.summary.gate is not None
    model = tmp_path / "runs" / "lin" / "model" / "surrogate.json"
    stored = read_surrogate(model)
    assert "marker" not in stored.manifest.options
    biased = _biased(model)  # every intercept shifted: its first check fails, it refits
    section = "\n".join([
        "[surrogate]", f'model_file = "{biased}"', "check_every = 1", "check_members = 6",
        'on_check_failure = "refit"',
    ])  # fmt: skip
    composition.start(cases.write_case(tmp_path, name="hm", surrogate=section))
    refit = tmp_path / "runs" / "hm" / "surrogates" / "refit_1" / "model" / "surrogate.json"
    assert refit.is_file()
    assert read_surrogate(refit).manifest.options["marker"] == 2.5
