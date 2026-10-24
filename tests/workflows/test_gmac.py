"""GIES with the ``gmac`` strategy end to end, on the linear cases of conftest.py.

Independent references: ``rlmmac`` (one identity term is RLM-MAC, bit for bit, through the
whole loop), and the per-member weighted least-squares fit of the perturbed data, which an
iterative smoother approaches on this linear, overdetermined case.
"""

import importlib
import itertools
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
import tomli_w
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.state import StateLayout, StateVariable
from apex.services import composition
from apex.storage.run_directory import RunDirectory
from apex.workflows.parameterization import PropertyMap, StackedProperties

#: Halving after an accepted step (``reduction = 0.5``) reaches the fixed point within the
#: 12 iterations; the default 0.9 lowers the damping more gently.
SCHEDULE = "schedule = { data_mismatch_tolerance = 1e-8, reduction = 0.5 }\nmax_iterations = 12"


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path

    def least_squares(self, perturbed: np.ndarray) -> np.ndarray: ...


CaseFactory = Callable[..., Case]


def _gies(strategy: str) -> str:
    return f'name = "gies"\nstrategy = {{ name = "{strategy}", options = {{ energy = 1.0'


def test_gies_with_one_identity_term_is_gies_with_rlmmac(linear_case: CaseFactory) -> None:
    rlmmac = linear_case(
        name="rlmmac", ensemble_size=20, method=_gies("rlmmac") + " } }\n" + SCHEDULE
    )
    gmac = linear_case(
        name="gmac", ensemble_size=20, method=_gies("gmac") + ', code = "100" } }\n' + SCHEDULE
    )

    expected = composition.start(rlmmac.config)
    result = composition.start(gmac.config)

    np.testing.assert_array_equal(result.posterior, expected.posterior)
    assert [r.data_mismatch_mean for r in result.history] == [
        r.data_mismatch_mean for r in expected.history
    ]


def test_gies_with_a_member_dependent_term_approaches_the_least_squares_fit(
    linear_case: CaseFactory,
) -> None:
    """Identity l2sq + state l1sq (MATLAB ``RLM_MAC_L1``): the step metric changes, the
    fixed point does not, since the metric only shapes the steps toward each member's fit."""
    terms = (
        ', terms = [{ transform = "identity" }, '
        '{ transform = { name = "state" }, norm = "l1sq", weight = { alpha = 0.5 } }] } }\n'
    )
    case = linear_case(ensemble_size=20, method=_gies("gmac") + terms + SCHEDULE)

    result = composition.start(case.config)

    perturbed = RunDirectory(case.directory / "runs" / "case").load_arrays(
        "perturbed_observations"
    )["perturbed_observations"]
    np.testing.assert_allclose(result.posterior, case.least_squares(perturbed), atol=2e-3)
    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert all(b < a for a, b in itertools.pairwise(accepted))


@pytest.mark.parametrize(
    ("option", "source"),
    [("state_variables", "\\[state\\]"), ("property_map", "\\[\\[parameterizations\\]\\]")],
)
def test_supplied_options_are_refused(linear_case: CaseFactory, option: str, source: str) -> None:
    method = _gies("gmac") + f', code = "100", {option} = [] }} }}'
    case = linear_case(ensemble_size=20, method=method)

    with pytest.raises(ConfigError, match=f"{option} comes from {source}"):
        composition.start(case.config)


MEMBER_DEPENDENT = (
    ', terms = [{ transform = "identity" }, '
    '{ transform = { name = "state" }, norm = "l1sq", weight = { alpha = 0.5 } }] } }\n'
)


@pytest.mark.parametrize("localization", ["adaptive", "adaptive_per_update"])
def test_a_localized_member_dependent_run_resumes_identically(
    linear_case: CaseFactory, localization: str
) -> None:
    """Revision 1 through the loop: per-member gains, localized (``adaptive_per_update``
    draws from the run's random stream at every update), and resume is bit for bit."""
    method = (
        _gies("gmac") + MEMBER_DEPENDENT + SCHEDULE
        + f'\nlocalization = {{ name = "{localization}" }}'
    )  # fmt: skip
    whole = linear_case(name="whole", ensemble_size=20, method=method)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", ensemble_size=20, method=method)
    composition.start(case.config, max_steps=2)

    resumed = composition.resume(case.directory / "runs" / "parts")

    accepted = [r.data_mismatch_mean for r in uninterrupted.history if r.accepted]
    assert accepted[-1] < accepted[0]
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [r.data_mismatch_mean for r in resumed.history] == [
        r.data_mismatch_mean for r in uninterrupted.history
    ]


def test_with_a_parameterization_the_terms_see_the_properties(linear_case: CaseFactory) -> None:
    """The terms name the simulator's properties (``y = 2 x + 1``);
    the latent state's names are refused."""
    importlib.import_module("tests.workflows.toy_parameterization")

    def term(variable: str) -> str:
        return (
            ', terms = [{ transform = "identity" }, { transform = { name = "state", options = '
            f'{{ variables = ["{variable}"] }} }}, norm = "l1sq" }}] }} }}\n' + SCHEDULE
        )

    allowed = linear_case(name="allowed", ensemble_size=20, method=_gies("gmac") + term("y"))
    refused = linear_case(name="refused", ensemble_size=20, method=_gies("gmac") + term("x"))
    for case in (allowed, refused):
        raw = tomllib.loads(case.config.read_text())
        raw["simulator"]["options"]["variables"] = ["y"]  # the simulator reads the property
        raw["parameterizations"] = [
            {"name": "test_affine", "options": {"scale": 2.0, "shift": 1.0}}
        ]
        case.config.write_text(tomli_w.dumps(raw))

    result = composition.start(allowed.config)

    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert accepted[-1] < accepted[0]
    with pytest.raises(ConfigError, match="terms name properties"):
        composition.start(refused.config)


class _TwoOutputs:
    """``x -> (b = x + 1, a = 2 x)``: two outputs, named out of order."""

    def __init__(self) -> None:
        self.variables: tuple[str, ...] = ("x",)
        self.outputs: tuple[str, ...] = ("b", "a")

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        return {"b": state["x"] + 1.0, "a": 2.0 * state["x"]}


def test_stacked_properties_follow_the_property_maps_names() -> None:
    layout = StateLayout([StateVariable("x", 6, grid=(3, 2, 1)), StateVariable("z", 2)])
    stacked = StackedProperties(PropertyMap([_TwoOutputs()], layout), layout)
    state = np.arange(16, dtype=float).reshape(8, 2)

    values, variables = stacked(state)

    assert stacked.names == ("a", "b", "z")
    assert variables == [
        {"name": "a", "size": 6, "grid": (3, 2, 1)},
        {"name": "b", "size": 6, "grid": (3, 2, 1)},
        {"name": "z", "size": 2, "grid": None},
    ]
    np.testing.assert_array_equal(values, np.concatenate([2 * state[:6], state[:6] + 1, state[6:]]))
