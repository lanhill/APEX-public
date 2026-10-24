"""Without ropt (no ``ert`` extra), APEX's optimizers still load and ``ropt`` says why it
cannot be built: the entry point ``apex.optimizers`` is installed regardless."""

import sys

import pytest

from apex.algorithms.optimization.registries import optimizers


def test_the_registry_works_without_ropt(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "ropt" or m.startswith("ropt.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "apex.integrations.everest.optimizer", raising=False)
    monkeypatch.setitem(sys.modules, "ropt", None)  # import ropt -> ImportError
    monkeypatch.setattr(optimizers, "_entry_points_loaded", False)
    monkeypatch.setattr(optimizers, "_factories", dict(optimizers._factories))
    optimizers._factories.pop("ropt", None)
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a: None if n == "ropt" else real(n, *a)
    )
    assert "enopt" in optimizers.names()
    assert optimizers.create("enopt", ne=4) is not None
    with pytest.raises(ValueError, match="'ert' extra"):
        optimizers.create("ropt", method="scipy/slsqp")
