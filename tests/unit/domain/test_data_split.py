"""Resolution of the [validation] data roles."""

import pytest

from apex.domain.config import DataSelection, PluginSection, ValidationSection
from apex.domain.data_split import keyword_of, resolve, well_of
from apex.domain.observations import DataKey, ObservationSet

KEYS = [
    DataKey(t, dt)
    for t in ("t0", "t1")
    for dt in ("WOPR:PRO1", "WWPR:PRO1", "WOPR:PRO2", "WBHP:INJ1", "FOPR")
] + [DataKey("t1", "SEIS", i) for i in range(3)]
OBSERVATIONS = ObservationSet(KEYS, [1.0] * len(KEYS), [0.1] * len(KEYS))


def _names(keys: tuple[DataKey, ...]) -> set[str]:
    return {str(k) for k in keys}


def test_well_and_keyword_of_eclipse_names() -> None:
    assert (well_of("WOPR:PRO1"), keyword_of("WOPR:PRO1")) == ("PRO1", "WOPR")
    assert (well_of("FOPR"), keyword_of("FOPR")) == ("FOPR", "FOPR")


def test_cv_of_a_well_takes_all_its_types_and_the_rest_is_history_matching() -> None:
    split = resolve(OBSERVATIONS, cv=DataSelection(wells=["PRO1"]))
    assert _names(split.cv) == {
        "WOPR:PRO1@t0",
        "WWPR:PRO1@t0",
        "WOPR:PRO1@t1",
        "WWPR:PRO1@t1",
    }
    assert len(split.history_matching) == len(KEYS) - 4
    assert all(k.datatype != "WOPR:PRO1" for k in split.history_matching)
    assert split.unused == ()
    # Roles keep the observations' order.
    assert list(split.cv) == sorted(split.cv, key=OBSERVATIONS.index_of)


def test_selectors_combine_and_times_restrict() -> None:
    split = resolve(
        OBSERVATIONS,
        cv=DataSelection(keywords=["WOPR"], times=["t1"]),
        test=DataSelection(datatypes=["WBHP:INJ1"]),
    )
    assert _names(split.cv) == {"WOPR:PRO1@t1", "WOPR:PRO2@t1"}
    assert _names(split.test) == {"WBHP:INJ1@t0", "WBHP:INJ1@t1"}
    description = split.describe()
    assert description["counts"] == {"history_matching": 9, "cv": 2, "test": 2, "unused": 0}
    assert description["ratio_history_matching_to_cv"] == 4.5


def test_an_explicit_history_matching_role_keeps_field_data_and_leaves_the_rest_unused() -> None:
    split = resolve(
        OBSERVATIONS,
        cv=DataSelection(wells=["PRO2"]),
        history_matching=DataSelection(wells=["PRO1"]),
    )
    assert {k.datatype for k in split.history_matching} == {"WOPR:PRO1", "WWPR:PRO1", "SEIS"}
    assert {k.datatype for k in split.unused} == {"WBHP:INJ1", "FOPR"}
    assert split.describe()["keys"]["history_matching"] == [  # type: ignore[index]
        "SEIS@t1",
        "WOPR:PRO1@t0",
        "WOPR:PRO1@t1",
        "WWPR:PRO1@t0",
        "WWPR:PRO1@t1",
    ]


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ({"cv": DataSelection(wells=["PRO9"])}, "wells entry 'PRO9' matches no data"),
        ({"cv": DataSelection(wells=["PRO1"], times=["t7"])}, "no data at times t7"),
        ({"cv": DataSelection(datatypes=["SEIS"])}, "field data are used for history matching"),
        ({"test": DataSelection(wells=["SEIS"])}, "field data are used for history matching"),
        (
            {"cv": DataSelection(wells=["PRO1"]), "test": DataSelection(keywords=["WOPR"])},
            "cv and test overlap: WOPR:PRO1@t0",
        ),
        (
            {
                "cv": DataSelection(wells=["PRO1"]),
                "history_matching": DataSelection(wells=["PRO1"]),
            },
            "cv and history_matching overlap",
        ),
    ],
)
def test_invalid_roles_are_refused(roles: dict[str, DataSelection], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve(OBSERVATIONS, **roles)


def test_a_selection_must_select_something_and_stopping_needs_cv() -> None:
    with pytest.raises(ValueError, match="needs wells, datatypes or keywords"):
        DataSelection(times=["t0"])
    with pytest.raises(ValueError, match="stopping needs a cv"):
        ValidationSection(stopping={"name": "select"})  # type: ignore[arg-type]
    section = ValidationSection(cv=DataSelection(wells=["PRO1"]))
    assert section.stopping_rule().name == "patience"
    assert section.stopping_rule().options == {"patience": 2}


def test_a_stopping_rule_may_be_given_by_its_name() -> None:
    """``stopping = "select"``, as the reference offers it, was refused (release tester).

    A bare string is a plugin section's name (``PluginSection``'s shorthand).
    """
    section = ValidationSection.model_validate({"cv": {"wells": ["P1"]}, "stopping": "select"})
    assert section.stopping_rule() == PluginSection(name="select")
    with pytest.raises(ValueError, match="at least 1 character"):
        ValidationSection.model_validate({"cv": {"wells": ["P1"]}, "stopping": ""})


def test_a_variant_cannot_switch_the_method_by_its_name_alone() -> None:
    """The shorthand must not slip a method switch past the check."""
    from apex.domain.study import VariantSection

    with pytest.raises(ValueError, match=r"patches method\.name"):
        VariantSection.model_validate({"name": "v", "patch": {"method": "esmda"}})


def test_only_the_auxiliary_sections_take_a_bare_name() -> None:
    """``method = "es"`` and the like stay tables: their readers expect one (review).

    The shorthand is the auxiliary parts' (strategy, localization, damping, stopping,
    generators, geophysics models, model error), and their JSON Schema says so.
    """
    from apex.domain.config import MethodSection

    method = MethodSection.model_validate(
        {"name": "gies", "strategy": "rlmmac", "localization": "adaptive", "damping": "lm"}
    )
    assert method.strategy == PluginSection(name="rlmmac")
    assert method.damping == PluginSection(name="lm")
    with pytest.raises(ValueError, match="valid dictionary"):
        PluginSection.model_validate("es")
    schema = ValidationSection.model_json_schema()["properties"]["stopping"]
    kinds = [o.get("type", o.get("$ref")) for o in schema["anyOf"]]
    assert kinds == ["string", "#/$defs/PluginSection", "null"]
