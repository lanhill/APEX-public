"""`apex pet-import` of a PET `compress` block builds the data space PET's reader builds.

The reference (`pet_import_compress`) is PET's `DataReader` + `DataLayout` on a small
seismic case: two compressed types (`avo`, `ts`) interleaved with a scalar `grav`, rows out
of label order, and three masks, so PET's single vintage counter (rows, then columns)
assigns avo@3000, ts@3000 and avo@1000 their masks and `min_noise` entries and leaves the
rest of the table uncompressed, with the cutoff inside a row. Masks 1 and 2 are 3-D and
applied in PET's C-order flattening to F-ordered data (D2). Here the same files are
imported and the converted configuration's observations are fitted as a run would.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from numpy.typing import NDArray

from apex.domain.observations import ObservationSet
from apex.services.composition import build_transforms, load_config
from apex.storage.inputs import read_observation_section
from apex.storage.pet_config import import_pet_case

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: As `IMPORT_COMPRESS` in tools/pet_reference/generate.py.
COMPRESS = {
    "compress_data": ["avo", "ts"],
    "dim": [8, 6, 4],
    "mask": ["mask_0.npz", "mask_1.npz", "mask_2.npz"],
    "level": 1,
    "wname": "db2",
    "threshold_rule": "universal",
    "th_mult": 0.5,
    "order": "F",
    "min_noise": [1e-9, 1e-6, 1e-9],
    "colored_noise": "yes",
    "use_hard_th": "yes",
    "keep_ca": "no",
    "inactive_value": 0.0,
}
FIELDS = [(3000, "avo"), (3000, "ts"), (1000, "avo"), (1000, "ts"), (2000, "avo"), (2000, "ts")]


def write_pet_case(
    directory: Path, ref: dict[str, NDArray[np.generic]], **compress: object
) -> Path:
    """Write the reference's PET case (CSV data, masks, fields, deck, prior, config)."""
    directory.mkdir()
    for label, column in FIELDS:
        np.savez(directory / f"{column}_{label}.npz", np.asarray(ref[f"in_{column}_{label}"]))
    for i in range(3):
        np.savez(directory / f"mask_{i}.npz", mask=np.asarray(ref[f"in_mask_{i}"]))
    (directory / "data.csv").write_text(str(ref["in_data_csv"]))
    (directory / "var.csv").write_text(str(ref["in_var_csv"]))
    (directory / "RUNFILE.mako").write_text("${permx[0]}\n")
    np.savez(directory / "prior.npz", permx=np.zeros((6, 4)))
    config = {
        "ensemble": {"ne": 4, "state": "permx", "importstate": "prior.npz"},
        "dataassim": {
            "scheme": "esmda",
            "data": "data.csv",
            "datavar": "var.csv",
            "mda": {"tot_assim_steps": 2},
            "compress": {**COMPRESS, **compress},
        },
        "simulator": {"runfile": "RUNFILE"},
    }
    path = directory / "pet.toml"
    path.write_text(tomli_w.dumps(config))
    return path


def fitted_observations(apex_config: Path) -> ObservationSet:
    """The observations of the converted run after its data transforms."""
    config = load_config(apex_config)
    observations = read_observation_section(config.observations)
    for transform in build_transforms(config):
        observations = transform.fit(observations)
    return observations


@pytest.mark.pet_parity
def test_imported_compression_reproduces_pets_data_and_variances(
    pet_reference: ReferenceLoader, tmp_path: Path
) -> None:
    ref = pet_reference("pet_import_compress")
    result = import_pet_case(write_pet_case(tmp_path / "pet", ref), tmp_path / "apex")
    observations = fitted_observations(result.config)

    rows: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(observations.keys):
        rows.setdefault((key.time, key.datatype), []).append(i)
    compressed = {(3000, "avo"), (3000, "ts"), (1000, "avo")}  # the first three vintages
    expected = {
        (str(t), f"wavelet({c})" if (t, c) in compressed else c): f"{c}_{t}" for t, c in FIELDS
    }
    expected.update({(str(t), "grav"): f"grav_{t}" for t in (3000, 1000, 4000, 2000)})
    assert set(rows) == set(expected)
    for group, name in expected.items():
        # rtol: the same transform and thresholds as the kernel parity tests (bit-identical
        # there); the margin absorbs only a future pywt's rounding.
        np.testing.assert_allclose(
            observations.values[rows[group]], np.asarray(ref[f"out_{name}"], float), rtol=1e-12
        )
        np.testing.assert_allclose(
            observations.variances[rows[group]],
            np.asarray(ref[f"out_{name.replace('_', '_var_')}"], float),
            rtol=1e-12,
        )
    assert any("D2" in note for note in result.notes)  # the transposed mask is reported


def test_the_converted_configuration(pet_reference: ReferenceLoader, tmp_path: Path) -> None:
    ref = pet_reference("pet_import_compress")
    result = import_pet_case(write_pet_case(tmp_path / "pet", ref), tmp_path / "apex")
    config = tomllib.loads(result.config.read_text())

    summary = [
        (t["options"]["datatype"], t["options"]["times"], t["options"]["min_noise"])
        for t in config["data_transforms"]
    ]
    assert summary == [("avo", ["3000"], 1e-9), ("ts", ["3000"], 1e-6), ("avo", ["1000"], 1e-9)]
    assert ["mask_file" in t["options"] for t in config["data_transforms"]] == [True] * 3
    assert config["data_transforms"][0]["options"]["noise"] == "coloured"
    assert config["data_transforms"][0]["options"]["thresholding"] == "hard"
    fields = config["observations"]["fields"]
    assert [(f["time"], f["datatype"]) for f in fields] == [(str(t), c) for t, c in FIELDS]
    assert fields[3]["variance"] == 0.0004  # ts@1000, uncompressed
    assert any("beyond the 3 mask(s)" in note for note in result.notes)


@pytest.mark.parametrize(
    ("compress", "message"),
    [
        ({"use_ensemble": "yes"}, "use_ensemble is not supported"),
        ({"wavelet": "haar"}, "unknown keys wavelet"),
        ({"compress_data": "seis"}, "no field data of seis"),
        ({"compress_data": "grav"}, "is a scalar"),
        ({"th_mult": -1.0}, "th_mult must be >= 0"),
        ({"min_noise": [1e-9]}, "one number per vintage"),
        ({"compress_data": "avots"}, "substrings of compress_data"),
        ({"keep_ca": "No"}, "must be yes/no"),
        ({"threshold_rule": "Universal"}, "unknown threshold_rule"),
        ({"order": "X"}, "order must be"),
    ],
)
def test_bad_compress_blocks_are_refused(
    pet_reference: ReferenceLoader, tmp_path: Path, compress: dict[str, object], message: str
) -> None:
    from apex.core.errors import ConfigError

    ref = pet_reference("pet_import_compress")
    with pytest.raises(ConfigError, match=message):
        import_pet_case(write_pet_case(tmp_path / "pet", ref, **compress), tmp_path / "apex")


def test_pet_only_behaviour_is_reproduced_and_noted(
    pet_reference: ReferenceLoader, tmp_path: Path
) -> None:
    ref = pet_reference("pet_import_compress")
    source = write_pet_case(
        tmp_path / "pet", ref, use_hard_th="no", colored_noise="no", threshold_rule="bayesian"
    )
    result = import_pet_case(source, tmp_path / "apex")
    options = tomllib.loads(result.config.read_text())["data_transforms"][0]["options"]

    assert options["threshold_rule"] == "universal"  # what PET computes with white noise
    assert options["thresholding"] == "soft"
    notes = " ".join(result.notes)
    assert "D4" in notes
    assert "D1" in notes
    fitted_observations(result.config)  # the converted case is valid


def test_missing_masks_are_all_active_and_overwrite_later_ones_as_in_pet(
    pet_reference: ReferenceLoader, tmp_path: Path
) -> None:
    """PET writes an all-true `mask_1.npz` for the missing first entry, then reads it (D5)."""
    ref = pet_reference("pet_import_compress")
    source = write_pet_case(tmp_path / "pet", ref, mask=["absent.npz", "mask_1.npz", "mask_2.npz"])
    result = import_pet_case(source, tmp_path / "apex")
    transforms = tomllib.loads(result.config.read_text())["data_transforms"]

    assert ["mask_file" in t["options"] for t in transforms] == [False, False, True]
    notes = " ".join(result.notes)
    assert "not found" in notes
    assert "overwrites" in notes
