"""``[geophysics]``: what it accepts and refuses before anything runs."""

from typing import Any

import pytest
from pydantic import ValidationError

from apex.domain.config import (
    FileWaveletSection,
    GeophysicsSection,
    OverburdenPressureFile,
    OverburdenPressureGradient,
    RickerWaveletSection,
    StationFileSection,
    avo_datatype,
)


def _section(**changes: Any) -> dict[str, Any]:  # noqa: ANN401
    options = {
        "angles": [10, 20],
        "wavelet": {"frequency": 45, "length": 0.05},
        "overburden": {"vp": 2450, "vs": 849, "density": 2300},
        "times": {"start": 1.9, "end": 2.2, "step": 0.001},
        **changes.pop("options", {}),
    }
    datum = {"kind": "avo", "datatype": "avo", "vintages": ["2025-06-01"], "options": options}
    return {
        "pem": {"name": "standard"},
        "overburden_pressure": {"gradient": 22600.0},
        "grid_file": "MODEL.EGRID",
        "data": [{**datum, **changes.pop("datum", {})}],
        **changes,
    }


def test_angles_name_the_fields_with_p_for_the_decimal_point() -> None:
    assert avo_datatype("avo", 10) == "avo_10"
    assert avo_datatype("avo", 12.5) == "avo_12p5"
    section = GeophysicsSection.model_validate(_section(options={"angles": [5, 12.5]}))
    assert section.data[0].datatypes() == ["avo_5", "avo_12p5"]


def test_the_defaults_and_the_shorthands() -> None:
    section = GeophysicsSection.model_validate(_section(options={"reflectivity": "fatti"}))
    options = section.data[0].avo()
    assert isinstance(options.wavelet, RickerWaveletSection)  # no name needed
    assert options.reflectivity.name == "fatti"
    assert (options.form, options.attribute, options.sampling) == ("maps", "rms", "interfaces")
    pressure = section.overburden_pressure
    assert isinstance(pressure, OverburdenPressureGradient)
    assert pressure.surface == 101325.0
    assert GeophysicsSection.model_validate(_section()).data[0].avo().reflectivity.name == (
        "zoeppritz"
    )
    wavelet = {"name": "file", "file": "w.npy", "dt": 0.001}
    filed = GeophysicsSection.model_validate(
        _section(options={"wavelet": wavelet}, overburden_pressure={"file": "ob.npz"})
    )
    assert isinstance(filed.data[0].avo().wavelet, FileWaveletSection)
    assert isinstance(filed.overburden_pressure, OverburdenPressureFile)
    assert filed.overburden_pressure.unit == "Pa"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"datum": {"vintages": ["1 JAN 2025"]}}, "not an ISO date"),
        ({"datum": {"vintages": ["2025-01-01", "2025-01-01"]}}, "vintages must be distinct"),
        ({"options": {"angles": [10, 10]}}, "angles must be distinct"),
        ({"options": {"angles": [95]}}, r"\[0, 90\)"),
        ({"options": {"time_window": [2.1, 2.0]}}, "start < end"),
        ({"options": {"form": "traces", "attribute": "mean"}}, "drop attribute"),
        ({"options": {"times": {"start": 2.0, "end": 1.0, "step": 0.001}}}, "end must be after"),
        ({"overburden_pressure": {"gradient": -1.0}}, "greater than 0"),
        ({"overburden_pressure": {"file": "ob.npz", "unit": "atm"}}, "unit"),
        ({"datum": {"kind": "magnetic"}}, "kind"),
        ({"datum": {"kind": "gravity"}}, "stations"),
        ({"datum": {"datatype": "avo 10"}}, "pattern"),
    ],
)
def test_settings_that_hold_for_no_member_are_refused(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        GeophysicsSection.model_validate(_section(**changes))


def test_two_data_may_not_give_the_same_field() -> None:
    raw = _section()
    raw["data"].append({**raw["data"][0], "vintages": ["2030-01-01"]})
    with pytest.raises(ValidationError, match=r"repeat the datatypes \['avo_10', 'avo_20'\]"):
        GeophysicsSection.model_validate(raw)


def _potential(kind: str, **options: Any) -> dict[str, Any]:  # noqa: ANN401
    datum = {
        "kind": kind,
        "datatype": "grav" if kind == "gravity" else "subsidence",
        "vintages": ["2026-01-01"],
        "options": {"baseline": "2025-01-01", "stations": {"spacing": 500.0}, **options},
    }
    return {"grid_file": "MODEL.EGRID", "data": [datum]}


def test_gravity_and_displacement_need_no_petro_elastic_model() -> None:
    gravity = GeophysicsSection.model_validate(_potential("gravity"))
    assert gravity.pem is None
    assert gravity.data[0].datatypes() == ["grav"]
    options = gravity.data[0].gravity()
    assert (options.model, options.gravitational_constant) == ("point_mass", 6.67430e-11)
    assert gravity.data[0].baseline() == "2025-01-01"
    with pytest.raises(TypeError, match="not AVO"):
        gravity.data[0].avo()
    shifted = GeophysicsSection.model_validate(
        _potential("seafloor_displacement", compressibility=1e-10, kernel="van_opstal")
    )
    displacement = shifted.data[0].displacement()
    assert displacement.kernel.name == "van_opstal"  # the shorthand
    assert (displacement.volume_change, displacement.volume) == ("pressure", "bulk")
    filed = GeophysicsSection.model_validate(
        _potential("gravity", stations={"file": "stations.npy"})
    )
    assert isinstance(filed.data[0].gravity().stations, StationFileSection)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_potential("seafloor_displacement"), "needs compressibility"),
        (
            _potential("seafloor_displacement", volume_change="pore_volume", volume="pore"),
            "volume and compressibility are for volume_change = 'pressure'",
        ),
        (_potential("gravity", baseline="2026-06-01"), "after"),
        (_potential("gravity", stations={"spacing": 0.0}), "greater than 0"),
        (_potential("gravity", compressibility=1e-10), "compressibility"),
        ({**_potential("gravity"), "data": [_section()["data"][0]]}, "pem and overburden"),
    ],
)
def test_potential_field_settings_that_hold_for_no_member_are_refused(
    raw: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        GeophysicsSection.model_validate(raw)


@pytest.mark.parametrize(
    "raw",
    [
        _section(),
        _section(options={"form": "traces"}),
        _section(options={"four_d": "differences", "baseline": "2025-01-01"}),
        _potential("gravity"),
        _potential("gravity", model="prism", stations={"file": "s.npy"}),
        _potential("seafloor_displacement", compressibility=1e-10),
        _potential("seafloor_displacement", compressibility=1e-10, volume="pore"),
        _potential("seafloor_displacement", volume_change="pore_volume"),
        _potential(
            "seafloor_displacement",
            volume_change="pore_volume",
            kernel={"name": "van_opstal", "options": {"basement_depth": 4000.0}},
            stations={
                "spacing": 500.0,
                "seabed_file": "seabed.npz",
                "seabed_key": "z",
                "refine": [100.0, 200.0, 800.0],
            },
        ),
    ],
)
def test_every_kind_reads_back_from_its_own_dump(raw: dict[str, Any]) -> None:
    """A regression (the phase-3 review): config.json and every job's spec are dumps,
    which wrote fields left at their defaults (volume, attribute) that a check on the
    fields set then refused, so a resume or a job failed at validation."""
    section = GeophysicsSection.model_validate(raw)
    again = GeophysicsSection.model_validate_json(section.model_dump_json())
    assert again == section


@pytest.mark.parametrize(
    ("stations", "message"),
    [
        ({"spacing": 500.0, "depth": 300.0, "seabed_file": "s.npy"}, "drop depth"),
        ({"spacing": 500.0, "seabed_key": "z"}, "seabed_key is for seabed_file"),
        ({"spacing": 500.0, "refine": [0.0, 0.0, 0.0]}, "radius must be positive"),
        ({"spacing": 1.0, "refine": [0.0, 0.0, 10.0]}, "spacing above 1 m"),
    ],
)
def test_stations_that_would_be_ignored_or_divide_by_zero_are_refused(
    stations: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        GeophysicsSection.model_validate(_potential("gravity", stations=stations))


def test_a_petro_elastic_model_without_avo_data_is_refused() -> None:
    raw = {**_potential("gravity"), "pem": {"name": "standard"}}
    with pytest.raises(ValidationError, match="used only by AVO data"):
        GeophysicsSection.model_validate(raw)


def test_errors_of_the_options_name_their_path() -> None:
    raw = _potential("seafloor_displacement")
    with pytest.raises(ValidationError) as error:
        GeophysicsSection.model_validate(raw)
    assert "options.:" not in str(error.value)
    assert "seafloor_displacement options: " in str(error.value)
