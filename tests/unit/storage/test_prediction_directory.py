"""A prediction that never finished a member is replaced by a changed one (release tester).

A failed ``apex predict`` left its folder, and the corrected rerun was refused ("holds a
prediction with another configuration"), with nothing in it worth keeping.
"""

from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.storage.locks import exclusive
from apex.storage.prediction_directory import PredictionDirectory


def test_an_unfinished_prediction_is_replaced(tmp_path: Path) -> None:
    folder = tmp_path / "future"
    with exclusive(folder, "prediction"):
        directory = PredictionDirectory(folder)
        assert directory.open(config_text="{}", fingerprint="a") is False
        (folder / "events.jsonl").write_text("{}\n")
        (folder / "simulations" / "posterior" / "member_0").mkdir(parents=True)

        assert directory.open(config_text='{"b": 1}', fingerprint="b") is False

        assert directory.read_state()["prediction_fingerprint"] == "b"
        assert (folder / "config.json").read_text() == '{"b": 1}'
        assert not (folder / "simulations").exists()
        assert not (folder / "events.jsonl").exists()
        assert (folder / "run.lock").exists()  # the lock this process holds stays


def test_a_prediction_with_finished_members_is_kept(tmp_path: Path) -> None:
    folder = tmp_path / "future"
    directory = PredictionDirectory(folder)
    directory.open(config_text="{}", fingerprint="a")
    directory.save_member("posterior", 0, np.zeros((1, 3)))

    with pytest.raises(ConfigError, match=r"1 finished member.*rm -r"):
        directory.open(config_text="{}", fingerprint="b")
    assert directory.load_member("posterior", 0) is not None


def test_a_folder_with_other_files_is_kept(tmp_path: Path) -> None:
    folder = tmp_path / "future"
    directory = PredictionDirectory(folder)
    directory.open(config_text="{}", fingerprint="a")
    (folder / "notes.txt").write_text("mine\n")

    with pytest.raises(ConfigError, match="other files"):
        directory.open(config_text="{}", fingerprint="b")
    assert (folder / "notes.txt").exists()


def test_the_same_configuration_resumes(tmp_path: Path) -> None:
    directory = PredictionDirectory(tmp_path / "future")
    directory.open(config_text="{}", fingerprint="a")
    directory.save_member("posterior", 0, np.zeros((1, 3)))

    assert directory.open(config_text="{}", fingerprint="a") is True
    assert directory.load_member("posterior", 0) is not None
