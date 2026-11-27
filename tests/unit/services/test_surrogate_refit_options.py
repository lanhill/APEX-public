"""The options a fitted surrogate hands an optimization's refits.

A model with ``refit_options`` (``core.surrogates.RefitOptions``) gives options its refits
take over the configuration's; the refitter accepts options that name nothing (numbers,
flags, text) and refuses those its plugin declares as files (and code, programs, arguments,
hosts, mappings), or that are not the plugin's.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.core.options import InputFile, Text
from apex.core.plugins import surrogate_models
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable
from apex.domain.surrogate import SnapshotProvenance, SnapshotSet
from apex.services.surrogates import _optimization_refitter, refit_options_of

NAME = "test_refit_options"


class _Plugin:
    """A surrogate plugin with a number, a text and a file option (nothing is fitted)."""

    version = "1"

    def __init__(
        self, *, scale: float = 1.0, label: Text = "a", data_file: InputFile | None = None
    ) -> None:
        self.scale, self.label, self.data_file = scale, label, data_file
        self.keys: tuple[DataKey, ...] = ()


class _Fitted:
    """A fitted model handing ``given`` to its refits."""

    def __init__(self, given: Mapping[str, object]) -> None:
        self.given = dict(given)

    def refit_options(self) -> Mapping[str, object]:
        return self.given


@pytest.fixture
def plugin() -> Iterator[None]:
    surrogate_models.register(NAME)(_Plugin)
    try:
        yield
    finally:
        surrogate_models.remove(NAME)


def _snapshots() -> SnapshotSet:
    return SnapshotSet(
        layout=StateLayout([StateVariable("x", 1)]), inputs={"x": np.ones((1, 2))},
        keys=(DataKey("1", "A"),), outputs=np.ones((1, 2)), run_ids=(0, 1),
        provenance=SnapshotProvenance(simulator="test", plan="prior"),
    )  # fmt: skip


@pytest.mark.usefixtures("plugin")
def test_a_number_and_a_text_reach_the_refits_fresh_model(tmp_path: Path) -> None:
    loaded: Any = SimpleNamespace(
        model=_Fitted({"scale": 2.5, "label": "kept"}), path=tmp_path / "model.json",
        identity="build", domain=SimpleNamespace(extent=None),
        manifest=SimpleNamespace(plugin=NAME, options={"scale": 1.0}, apex_commit="test"),
    )  # fmt: skip
    simulation: Any = SimpleNamespace(keys=(DataKey("1", "A"),))
    refitter = _optimization_refitter(loaded, _snapshots(), simulation, lambda x, r: {}, 1,
                                      lambda found: cast(Any, None))  # fmt: skip
    fresh = cast(_Plugin, refitter.fresh_model())
    assert (fresh.scale, fresh.label) == (2.5, "kept")


@pytest.mark.usefixtures("plugin")
def test_a_file_option_or_an_unknown_one_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"naming files.*\['data_file'\]"):
        refit_options_of(_Fitted({"data_file": "elsewhere.npz"}), NAME)
    with pytest.raises(ConfigError, match=r"not its options \['nothing'\]"):
        refit_options_of(_Fitted({"nothing": 1.0}), NAME)
    # A number option holding text (an undeclared kind) is refused too.
    with pytest.raises(ConfigError, match="naming files"):
        refit_options_of(_Fitted({"scale": "/etc/passwd"}), NAME)
    assert refit_options_of(object(), NAME) == {}
