"""The services root: an empty ``APEX_HOME`` counts as unset (a stray apex.db, 2026-10-04)."""

from pathlib import Path

import pytest

from apex.services.api import default_root


def test_an_empty_apex_home_is_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APEX_HOME", "")
    assert default_root() == tmp_path / ".apex" / "service"
    monkeypatch.setenv("APEX_HOME", str(tmp_path / "root"))
    assert default_root() == tmp_path / "root"
    monkeypatch.delenv("APEX_HOME")
    assert default_root() == tmp_path / ".apex" / "service"
