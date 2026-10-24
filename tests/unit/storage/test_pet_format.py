from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.domain import DataKey
from apex.storage.inputs import read_observations
from apex.storage.pet_format import (
    PetField,
    field_variance,
    read_pet_data,
    read_pet_observations,
    write_observations,
)


def _write(tmp_path: Path, data: str, var: str) -> tuple[Path, Path]:
    (tmp_path / "data.csv").write_text(data)
    (tmp_path / "var.csv").write_text(var)
    return tmp_path / "data.csv", tmp_path / "var.csv"


def test_pet_observations_convert_abs_and_rel_variances_and_skip_missing(tmp_path: Path) -> None:
    data, var = _write(
        tmp_path,
        "dates,WOPR:P1,WWPR:P1\n2024-01-01,200.0,\n2025-01-01,100.0,4.0\n",
        "dates,WOPR:P1,WWPR:P1\n"
        "2024-01-01,\"['rel', 10]\",\"['abs', 1.0]\"\n"
        "2025-01-01,\"['abs', 25.0]\",\"['abs', 0.5]\"\n",
    )

    observations = read_pet_observations(data, var)

    assert observations.keys == (
        DataKey("2024-01-01", "WOPR:P1"),
        DataKey("2025-01-01", "WOPR:P1"),
        DataKey("2025-01-01", "WWPR:P1"),
    )
    np.testing.assert_allclose(observations.variances, [400.0, 25.0, 0.5])  # (0.1 * 200)^2


def test_converted_observations_round_trip_through_the_native_csv(tmp_path: Path) -> None:
    data, var = _write(
        tmp_path, "dates,Y\n2024-01-01,1.25\n", "dates,Y\n2024-01-01,\"['abs', 0.1]\"\n"
    )
    observations = read_pet_observations(data, var)

    write_observations(observations, tmp_path / "obs.csv")
    again = read_observations(tmp_path / "obs.csv")

    assert again.keys == observations.keys
    np.testing.assert_array_equal(again.values, observations.values)
    np.testing.assert_array_equal(again.variances, observations.variances)


@pytest.mark.parametrize(
    ("var_cell", "message"),
    [("\"['emp', 3]\"", "unsupported variance type"), ("oops", "cannot parse")],
)
def test_unsupported_variance_cells_are_rejected(
    tmp_path: Path, var_cell: str, message: str
) -> None:
    data, var = _write(tmp_path, "dates,Y\nt,1.0\n", f"dates,Y\nt,{var_cell}\n")

    with pytest.raises(ConfigError, match=message):
        read_pet_observations(data, var)


def test_mismatched_files_are_rejected(tmp_path: Path) -> None:
    data, var = _write(tmp_path, "dates,Y\nt,1.0\n", "dates,Z\nt,\"['abs', 1]\"\n")

    with pytest.raises(ConfigError, match="same columns"):
        read_pet_observations(data, var)


def test_npz_cells_are_fields_in_pets_order(tmp_path: Path) -> None:
    np.savez(tmp_path / "f.npz", np.array([[1.0, 2.0, 3.0]]))  # squeezed, as PET does
    (tmp_path / "data.csv").write_text("t,avo,q\n1,f.npz,4.0\n2,,5.0\n")
    (tmp_path / "var.csv").write_text(
        "t,avo,q\n1,\"['rel', 10]\",\"['abs', 1]\"\n2,,\"['abs', 1]\"\n"
    )

    entries = read_pet_data(tmp_path / "data.csv", tmp_path / "var.csv", tmp_path)

    assert [(type(e).__name__, e.time, e.datatype) for e in entries] == [
        ("PetField", "1", "avo"),
        ("PetScalar", "1", "q"),
        ("PetScalar", "2", "q"),
    ]
    field = entries[0]
    assert isinstance(field, PetField)
    np.testing.assert_array_equal(field.values, [1.0, 2.0, 3.0])
    assert field_variance(field, tmp_path / "var.csv") == ("rel", 10.0)
    with pytest.raises(ConfigError, match="need read_pet_data"):
        read_pet_observations(tmp_path / "data.csv", tmp_path / "var.csv")
    (tmp_path / "data.csv").write_text("t,avo,q\n1,missing.npz,4.0\n2,,5.0\n")
    with pytest.raises(ConfigError, match="not found"):
        read_pet_data(tmp_path / "data.csv", tmp_path / "var.csv", tmp_path)
