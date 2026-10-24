from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError, WorkflowError
from apex.domain import DataKey, IterationRecord, StateLayout, StateVariable
from apex.domain.config import ObservationsSection
from apex.storage.inputs import read_observation_section, read_observations, read_prior_ensemble
from apex.storage.run_directory import RunDirectory


def test_run_directory_round_trips_arrays_state_and_history(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run")
    run.initialise("config")
    record = IterationRecord(0, 10.0, 1.0, 5)

    run.save_arrays("iter_0", {"state": np.eye(2)})
    run.append_history(record)
    run.write_state({"completed": 0})

    np.testing.assert_array_equal(run.load_arrays("iter_0")["state"], np.eye(2))
    assert run.history() == [record]
    assert run.read_state() == {"completed": 0}
    assert run.config_text() == "config"
    assert run.simulation_dir(2, -1).parts[-2:] == ("iter_2", "mean")


def test_run_directory_refuses_to_overwrite_and_truncates_history(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run")
    run.initialise("config")
    for i in range(3):
        run.append_history(IterationRecord(i, 1.0, 0.0, 2))
    run.write_state({})

    run.truncate_history(1)

    assert len(run.history()) == 1
    with pytest.raises(WorkflowError, match="already holds a run"):
        run.initialise("config")
    with pytest.raises(WorkflowError, match="no saved state"):
        RunDirectory(tmp_path / "empty").read_state()


def test_read_observations(tmp_path: Path) -> None:
    path = tmp_path / "obs.csv"
    path.write_text("time,datatype,value,variance\n2024-01-01,WOPR:P1,10.5,0.25\n")

    observations = read_observations(path)

    assert observations.keys == (DataKey("2024-01-01", "WOPR:P1"),)
    np.testing.assert_array_equal(observations.values, [10.5])


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("time,type,value,variance\n", "expected the header line"),
        ("time,datatype,value,variance\nt,a,x,1\n", "line 2: value is not a number"),
        ("time,datatype,value,variance\nt,a,1,-1\n", "positive"),
    ],
)
def test_read_observations_rejects_bad_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "obs.csv"
    path.write_text(content)

    with pytest.raises(ConfigError, match=message):
        read_observations(path)


def test_read_prior_ensemble(tmp_path: Path) -> None:
    layout = StateLayout([StateVariable("a", 2), StateVariable("b", 1)])
    path = tmp_path / "prior.npz"
    np.savez(path, a=np.zeros((2, 4)), b=np.ones((1, 4)))

    prior = read_prior_ensemble(path, layout)

    np.testing.assert_array_equal(prior, [[0] * 4, [0] * 4, [1] * 4])
    np.savez(path, a=np.zeros((2, 4)))
    with pytest.raises(ConfigError, match="missing state variables"):
        read_prior_ensemble(path, layout)
    np.savez(path, a=np.zeros((2, 4)), b=np.ones((1, 5)))
    with pytest.raises(ConfigError, match="equal n_ensemble"):
        read_prior_ensemble(path, layout)


def test_field_observations_follow_the_scalar_csv(tmp_path: Path) -> None:
    (tmp_path / "obs.csv").write_text("time,datatype,value,variance\nt0,q,1.5,0.1\n")
    np.savez(tmp_path / "f.npz", other=np.zeros(2), avo=np.array([[1.0, 2.0, 3.0]]))
    section = ObservationsSection.model_validate(
        {
            "file": tmp_path / "obs.csv",
            "fields": [
                {
                    "time": "t1",
                    "datatype": "avo",
                    "file": tmp_path / "f.npz",
                    "key": "avo",
                    "variance": 0.2,
                }
            ],
        }
    )

    observations = read_observation_section(section)

    assert observations.keys == (
        DataKey("t0", "q"),
        DataKey("t1", "avo", 0),
        DataKey("t1", "avo", 1),
        DataKey("t1", "avo", 2),
    )
    np.testing.assert_array_equal(observations.values, [1.5, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(observations.variances, [0.1, 0.2, 0.2, 0.2])


def test_field_observation_errors(tmp_path: Path) -> None:
    np.save(tmp_path / "grid.npy", np.zeros((3, 4)))

    def section(**field: object) -> ObservationsSection:
        entry = {"time": "t", "datatype": "avo", "variance": 1.0, **field}
        return ObservationsSection.model_validate({"fields": [entry]})

    with pytest.raises(ConfigError, match="not found"):
        read_observation_section(section(file=tmp_path / "missing.npz"))
    with pytest.raises(ConfigError, match="1-D"):
        read_observation_section(section(file=tmp_path / "grid.npy"))
    with pytest.raises(ValueError, match="needs a file, fields"):
        ObservationsSection.model_validate({})
