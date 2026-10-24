"""The remote client sends absolute paths (review of finding 31 of the release tester).

The service reads a relative path as one under its workspace, so a client resolves its own
relative paths where it runs before sending them.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from apex.client import RemoteClient


class _Recorder(RemoteClient):
    """A remote client that records what it would send."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:  # noqa: ANN401
        self.sent.append({"method": method, "path": path, **kwargs})
        raise RuntimeError("recorded")


def test_relative_paths_are_resolved_before_they_are_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = _Recorder()
    calls: list[Callable[[], object]] = [
        lambda: client.validate({"run": {}}, base_dir=Path("cases")),
        lambda: client.submit_run({"run": {}}, base_dir=Path("cases")),
        lambda: client.submit_study(Path("study.toml"), base_dir=Path("cases")),
        lambda: client.validate_optimization({"x": 1}, base_dir=Path("cases")),
        lambda: client.submit_optimization({"x": 1}, base_dir=Path("cases")),
        lambda: client.prediction_summary(Path("runs/r/predictions/p")),
        lambda: client.optimization_summary(Path("opt/o")),
        lambda: client.submit_run(Path("run.toml")),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="recorded"):
            call()
    for sent in client.sent:
        values = [*(sent.get("body") or {}).values(), *(sent.get("params") or {}).values()]
        paths = [v for v in values if isinstance(v, str) and ("/" in v or v.endswith(".toml"))]
        assert paths, sent
        for value in paths:
            assert os.path.isabs(value), sent
            assert value.startswith(str(tmp_path.resolve())), sent
