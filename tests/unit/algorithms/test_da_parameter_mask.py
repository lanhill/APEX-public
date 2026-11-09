"""A model's parameter mask on the localization taper.

A model-error model whose parameters can affect only some data declares the mask;
``adaptive_prior`` multiplies its taper's rows by it, the rest of the taper unchanged; the
masked taper is saved and restored as before (so resume is bit-identical); composition
hands the mask over, and refuses a run that cannot apply it.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apex.algorithms.data_assimilation.chop.method import Chop
from apex.algorithms.data_assimilation.chop.taper import ChopTaper
from apex.algorithms.data_assimilation.localization import (
    DistanceLocalization,
    Localization,
    LocalizedMethod,
    PriorAdaptiveLocalization,
    RestrictableLocalization,
)
from apex.core.errors import ConfigError
from apex.domain import DataKey
from apex.services.composition import attach_parameter_mask

KEYS = [DataKey(f"2000-01-0{t}", n) for t in (1, 2) for n in ("WOPR:P1", "WBHP:P1", "FOPR")]


def _ensemble() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    state = rng.standard_normal((10, 100))
    forecast = rng.standard_normal((len(KEYS), 100))
    forecast[:3] += 3 * state[:3].sum(axis=0)  # real correlations of rows 0-2
    forecast[3:] += 3 * state[6:9].sum(axis=0)
    return state, forecast


def _mask(keys: Any) -> np.ndarray:  # noqa: ANN401
    """Rows 6-9 (the second block) affect the second date's data only."""
    return np.array([[k.time.endswith("2") for k in keys]] * 4)


@pytest.mark.parametrize("unlocalized", [(), ("nn",)])
def test_the_mask_multiplies_its_rows_only_and_survives_resume(
    unlocalized: tuple[str, ...],
) -> None:
    state, forecast = _ensemble()
    options: dict[str, Any] = {"threshold": "fixed", "unlocalized": list(unlocalized),
                               "variable_names": ["k", "nn"]}  # fmt: skip
    plain = PriorAdaptiveLocalization([6, 4], **options)
    plain.prepare(state, forecast, KEYS, np.random.default_rng(1))
    masked = PriorAdaptiveLocalization([6, 4], **options)
    assert isinstance(masked, RestrictableLocalization)
    masked.restrict(slice(6, 10), _mask)
    masked.prepare(state, forecast, KEYS, np.random.default_rng(1))
    assert plain.taper_matrix is not None
    assert masked.taper_matrix is not None
    np.testing.assert_array_equal(masked.taper_matrix[:6], plain.taper_matrix[:6])
    np.testing.assert_array_equal(masked.taper_matrix[6:], plain.taper_matrix[6:] * _mask(KEYS))
    assert masked.taper_matrix[6:, :3].sum() == 0
    if unlocalized:  # the mask alone: ones inside it
        np.testing.assert_array_equal(masked.taper_matrix[6:], _mask(KEYS).astype(float))
    restored = PriorAdaptiveLocalization([6, 4], **options)
    restored.restrict(slice(6, 10), _mask)
    restored.restore_state(masked.export_state())
    np.testing.assert_array_equal(restored.taper_matrix, masked.taper_matrix)


def test_the_mask_is_checked() -> None:
    state, forecast = _ensemble()
    loc = PriorAdaptiveLocalization([6, 4])
    with pytest.raises(ValueError, match="not a range"):
        loc.restrict(slice(6, 11), _mask)
    with pytest.raises(ValueError, match="not a range"):
        loc.restrict(slice(6, 6), _mask)
    loc.restrict(slice(5, 10), _mask)  # 5 rows, a mask of 4
    with pytest.raises(ValueError, match=r"booleans \(5, 6\), got bool \(4, 6\)"):
        loc.prepare(state, forecast, KEYS, np.random.default_rng(1))
    # Open slices are normalised (slice(None, n) has no start).
    ok = PriorAdaptiveLocalization([4, 6])
    ok.restrict(slice(None, 4), _mask)
    ok.prepare(state, forecast, KEYS, np.random.default_rng(1))
    assert ok.taper_matrix is not None
    assert ok.taper_matrix[:4, :3].sum() == 0


def _method(localization: object) -> object:
    """A method reporting its localization (LocalizedMethod)."""
    return SimpleNamespace(current_localization=lambda: localization)


LAYOUT: Any = SimpleNamespace(variables=[SimpleNamespace(name="k", size=6),
                                         SimpleNamespace(name="nn", size=4)])  # fmt: skip
MODEL: Any = SimpleNamespace(state_variable="nn", masks_parameters=True, parameter_mask=_mask)


def test_composition_hands_the_mask_over_and_prepare_applies_it() -> None:
    state, forecast = _ensemble()
    loc = PriorAdaptiveLocalization([6, 4])
    attach_parameter_mask(_method(loc), LAYOUT, MODEL, KEYS)
    loc.prepare(state, forecast, KEYS, np.random.default_rng(1))
    assert loc.taper_matrix is not None
    assert loc.taper_matrix[6:, :3].sum() == 0  # the first date's data: masked off
    # The mask is evaluated on the keys the run assimilates (here a held-out subset).
    held = PriorAdaptiveLocalization([6, 4])
    attach_parameter_mask(_method(held), LAYOUT, MODEL, KEYS)
    held.prepare(state, forecast[3:], KEYS[3:], np.random.default_rng(1))
    assert held.taper_matrix is not None
    assert held.taper_matrix.shape == (10, 3)


@pytest.mark.parametrize(
    ("localization", "match"),
    [(None, "it has none"), ("distance", "DistanceLocalization cannot apply one")],
)
def test_composition_refuses_a_run_that_cannot_apply_the_mask(
    localization: str | None, match: str
) -> None:
    found = None if localization is None else DistanceLocalization(
        grid=[2, 3, 1], positions={"P1": [[1, 1, 1]]}, radius=1.0, block_sizes=[6, 6]
    )  # fmt: skip
    with pytest.raises(ConfigError, match=match):
        attach_parameter_mask(_method(found), LAYOUT, MODEL, KEYS)
    with pytest.raises(ConfigError, match=r"object cannot report its localization\. Use a method"):
        attach_parameter_mask(object(), LAYOUT, MODEL, KEYS)  # a method that cannot say
    with pytest.raises(ConfigError, match="data_transforms"):
        attach_parameter_mask(_method(PriorAdaptiveLocalization([6, 4])), LAYOUT, MODEL, KEYS,
                              transformed=True)  # fmt: skip
    other: Any = SimpleNamespace(state_variable="theta", masks_parameters=True,
                                 parameter_mask=_mask)  # fmt: skip
    with pytest.raises(ConfigError, match="not in \\[state\\]"):
        attach_parameter_mask(_method(PriorAdaptiveLocalization([6, 4])), LAYOUT, other, KEYS)


def test_runs_without_a_declared_mask_are_unchanged() -> None:
    loc = PriorAdaptiveLocalization([6, 4])
    for model in (None, SimpleNamespace(state_variable="nn"),
                  SimpleNamespace(state_variable="nn", masks_parameters=False,
                                  parameter_mask=_mask)):  # fmt: skip
        attach_parameter_mask(_method(loc), LAYOUT, model, KEYS)
        attach_parameter_mask(_method(None), LAYOUT, model, KEYS)
    state, forecast = _ensemble()
    plain = PriorAdaptiveLocalization([6, 4])
    loc.prepare(state, forecast, KEYS, np.random.default_rng(1))
    plain.prepare(state, forecast, KEYS, np.random.default_rng(1))
    np.testing.assert_array_equal(loc.taper_matrix, plain.taper_matrix)


def test_a_wrong_or_empty_mask_is_refused() -> None:
    def wrong(keys: Any) -> np.ndarray:  # noqa: ANN401
        return np.ones((3, len(keys)), dtype=bool)

    def numbers(keys: Any) -> np.ndarray:  # noqa: ANN401
        return np.ones((4, len(keys)))

    for mask, match in ((wrong, r"\(4, 6\), got bool \(3, 6\)"),
                        (numbers, "got float64")):  # fmt: skip
        model: Any = SimpleNamespace(state_variable="nn", masks_parameters=True,
                                     parameter_mask=mask)  # fmt: skip
        with pytest.raises(ConfigError, match=match):
            attach_parameter_mask(_method(PriorAdaptiveLocalization([6, 4])), LAYOUT, model,
                                  KEYS)  # fmt: skip
    # A mask keeping no assimilated datum would freeze the variable: refused in prepare.
    state, forecast = _ensemble()
    loc = PriorAdaptiveLocalization([6, 4])
    loc.restrict(slice(6, None), _mask)
    with pytest.raises(ValueError, match="keeps no assimilated datum"):
        loc.prepare(state, forecast[:3], KEYS[:3], np.random.default_rng(1))


def test_a_mask_whose_columns_depend_on_other_keys_is_refused() -> None:
    """Each column must follow from its own key: the run assimilates a subset of the
    observations (held-out data left out), on which the mask is evaluated again."""

    def by_position(keys: Any) -> np.ndarray:  # noqa: ANN401
        first = np.arange(len(keys)) < len(keys) // 2  # the first half of whatever is given
        return np.array([first] * 4)

    model: Any = SimpleNamespace(state_variable="nn", masks_parameters=True,
                                 parameter_mask=by_position)  # fmt: skip
    with pytest.raises(ConfigError, match="from its key alone"):
        attach_parameter_mask(_method(PriorAdaptiveLocalization([6, 4])), LAYOUT, model, KEYS)

    def numbers_on_subsets(keys: Any) -> Any:  # noqa: ANN401
        found = _mask(keys)
        return found if len(keys) == len(KEYS) else found.astype(float)

    def only_all_keys(keys: Any) -> np.ndarray:  # noqa: ANN401
        if len(keys) != len(KEYS):
            raise KeyError("a key it was not built on")
        return _mask(keys)

    for mask, match in ((numbers_on_subsets, "from its key alone"),
                        (only_all_keys, "fails on a subset of the keys \\(KeyError")):  # fmt: skip
        odd: Any = SimpleNamespace(state_variable="nn", masks_parameters=True,
                                   parameter_mask=mask)  # fmt: skip
        with pytest.raises(ConfigError, match=match):
            attach_parameter_mask(_method(PriorAdaptiveLocalization([6, 4])), LAYOUT, odd, KEYS)
    attach_parameter_mask(_method(PriorAdaptiveLocalization([6, 4])), LAYOUT, MODEL, KEYS)


def _chop_taper_is_a_localization(taper: ChopTaper) -> Localization:
    return taper  # checked by mypy: CHOP's taper satisfies the protocol it reports under


def _chop_reports_its_localization(method: Chop) -> LocalizedMethod:
    return method  # checked by mypy


def test_the_methods_report_a_localization_protocol() -> None:
    """``LocalizedMethod.current_localization`` returns ``Localization | None``; CHOP's
    ChopTaper is one (the functions above are type-checked), which it cannot restrict."""
    taper = ChopTaper([])
    assert _chop_taper_is_a_localization(taper) is taper
    assert not isinstance(taper, RestrictableLocalization)
