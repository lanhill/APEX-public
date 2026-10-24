from pathlib import Path

import pytest
from typer.testing import CliRunner

from apex import __version__
from apex.cli.main import app


def test_version_command_prints_installed_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_localization_needs_a_strategy_that_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A [method.localization] block is refused, not ignored, for strategies without one."""
    import numpy as np

    from apex.algorithms.data_assimilation import update_strategies
    from apex.core.errors import ConfigError
    from apex.domain.config import HistoryMatchConfig
    from apex.domain.state import StateLayout, StateVariable
    from apex.services.composition import _build_method

    class Plain:
        requires_reference_prediction = False

        def __init__(self, energy: float = 1.0) -> None:
            self.energy = energy

        def step(self, state: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
            return np.zeros_like(state)

    # A temporary registry entry, removed when the test ends.
    monkeypatch.setitem(update_strategies._factories, "plain_test_only", Plain)
    config = HistoryMatchConfig.model_validate(
        {
            "run": {"name": "r", "seed": 1},
            "state": {"variables": [{"name": "x", "size": 2}]},
            "prior": {"ensemble_size": 4},
            "observations": {"file": str(tmp_path / "obs.csv")},
            "simulator": {"name": "linear"},
            "case": {"kind": "real"},
            "method": {
                "name": "esmda",
                "options": {"inflation": [1.0]},
                "strategy": {"name": "plain_test_only"},
                "localization": {"name": "adaptive"},
            },
        }
    )
    with pytest.raises(ConfigError, match="does not support localization"):
        _build_method(config, StateLayout([StateVariable("x", 2)]))


def test_the_old_composition_modules_forward_with_a_warning() -> None:
    """Moved composition to apex.services; apex.cli keeps it for one release."""
    import importlib
    import sys

    from apex.services import composition, prediction

    for name, moved in (("composition", composition), ("prediction", prediction)):
        sys.modules.pop(f"apex.cli.{name}", None)
        with pytest.warns(DeprecationWarning, match=f"apex.services.{name}"):
            old = importlib.import_module(f"apex.cli.{name}")
        assert old.load_prediction_config if name == "prediction" else old.load_config
        assert getattr(old, "PATH_OPTION_SUFFIXES", None) is getattr(
            moved, "PATH_OPTION_SUFFIXES", None
        )


def test_a_run_stopped_while_it_waits_exits_like_a_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of the follow-ups: a stop asked while a run waited outside its steps
    (RunStopped) is a stop, not a failure: exit 0, with what to do next."""
    from apex.core.errors import RunStopped
    from apex.services import composition

    def stopped(*_: object) -> None:
        raise RunStopped("asked to stop while it waited")

    monkeypatch.setattr(composition, "start", stopped)
    result = CliRunner().invoke(app, ["run", str(tmp_path / "config.toml"), "--no-progress"])
    assert result.exit_code == 0, result.output
    assert "stopped: asked to stop while it waited" in result.output
