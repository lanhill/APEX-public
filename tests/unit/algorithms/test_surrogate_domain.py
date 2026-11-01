"""A surrogate's validated domain (validation 1): built and checked by core.

References are constructed points whose answers are known by hand: a box, a triangle's convex
hull, an ellipse whose Mahalanobis radius is computed in closed form, and labels on both time
axes.
"""

import json
import math
import pickle

import numpy as np
import pytest

from apex.algorithms.surrogates.domain import (
    check_coordinates,
    check_request,
    domain_from_coordinates,
)
from apex.core.errors import ApexError, SimulationError, SurrogateDomainError
from apex.domain.observations import DataKey
from apex.domain.surrogate import (
    DomainExtent,
    TrainingDomain,
    TrainingDomainDocument,
    time_position,
)

KEYS = (DataKey("2024-01-01", "WOPR:P1"), DataKey("2024-07-01", "WOPR:P1"))


def _box() -> TrainingDomain:
    # Two coordinates spanning [0, 1] x [-2, 4]; residuals at most 0.5.
    points = np.array([[0.0, 1.0, 0.5], [-2.0, 4.0, 0.0]])
    return domain_from_coordinates(points, np.array([0.0, 0.5, 0.2]), KEYS, input_scale=10.0)


def test_the_box_is_the_validated_points_bounds_and_names_the_excess() -> None:
    domain = _box()
    np.testing.assert_array_equal(domain.lower, [0.0, -2.0])
    np.testing.assert_array_equal(domain.upper, [1.0, 4.0])
    assert domain.residual_bound == 0.5
    assert domain.time_range == ("2024-01-01", "2024-07-01")

    members = np.array([[0.0, 1.0, 0.5, 1.25, -0.5], [4.0, -2.0, 1.0, 1.0, 5.0]])
    found = check_coordinates(domain, members, np.zeros(5), [10, 11, 12, 13, 14])

    # The corners are inside (inclusive); 13 is above on #0, 14 below on #0 and above on #1.
    assert found.members == (13, 14)
    [above] = found.outside[0].coordinates
    assert (above.index, above.side, above.bound) == (0, "above", 1.0)
    assert above.excess == pytest.approx(0.25)
    by_index = {c.index: c for c in found.outside[1].coordinates}
    assert by_index[0].side == "below"
    assert by_index[0].excess == pytest.approx(0.5)
    assert by_index[1].side == "above"
    assert by_index[1].excess == pytest.approx(1.0)
    assert "member 13" in found.message()
    assert "by 0.25" in found.message()


def test_round_off_is_inside_and_anything_beyond_it_outside() -> None:
    domain = _box()
    # A relative 1e-12 beyond a bound is round-off (batch sizes round differently); 1e-6 is not.
    members = np.array([[1.0 + 1e-12, 1.0 + 1e-6], [4.0, 4.0]])
    assert check_coordinates(domain, members, np.zeros(2), [0, 1]).members == (1,)


def test_a_residual_above_the_bound_is_outside_even_inside_the_box() -> None:
    domain = _box()
    members = np.array([[0.5, 0.5], [0.0, 0.0]])

    found = check_coordinates(domain, members, np.array([0.5, 0.6]), [0, 1])

    assert found.members == (1,)
    assert found.outside[0].residual == 0.6
    assert found.outside[0].coordinates == ()
    assert "projection residual 0.6 above 0.5" in found.message()


def test_values_that_are_not_finite_are_outside() -> None:
    found = check_coordinates(_box(), np.array([[np.nan], [0.0]]), np.array([np.nan]), [3])
    assert found.members == (3,)
    assert found.outside[0].residual is not None


def test_the_convex_hull_refuses_a_point_inside_the_box_but_outside_the_triangle() -> None:
    triangle = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    domain = domain_from_coordinates(triangle, np.zeros(3), KEYS, extent="hull")
    assert domain.extent == "hull"

    members = np.array([[0.25, 0.9, 1.0, 0.5], [0.25, 0.9, 0.0, 0.5]])
    found = check_coordinates(domain, members, np.zeros(4), [0, 1, 2, 3])

    # (0.9, 0.9) is in the box [0,1]^2 but beyond the edge x + y = 1; a vertex and a point on
    # the edge are inside.
    assert found.members == (1,)
    assert found.outside[0].outside_hull
    assert found.outside[0].coordinates == ()


def test_the_mahalanobis_ellipse_has_its_closed_form_radius() -> None:
    # Points (+-1, 0), (0, +-2): mean 0, sample covariance diag(2/3, 8/3), so every point is at
    # distance sqrt(1.5), the radius.
    points = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 2.0, -2.0]])
    domain = domain_from_coordinates(points, np.zeros(4), KEYS, extent="mahalanobis")
    assert domain.mahalanobis is not None
    assert domain.mahalanobis.radius == pytest.approx(math.sqrt(1.5), rel=1e-14)

    # (0.8, 1.6): d^2 = 0.64 * 1.5 + 2.56 * 3/8 = 1.92 > 1.5, inside the box; (0.5, 1): 0.75.
    found = check_coordinates(domain, np.array([[0.8, 0.5], [1.6, 1.0]]), np.zeros(2), [0, 1])

    assert found.members == (0,)
    assert found.outside[0].distance == pytest.approx(math.sqrt(1.92), rel=1e-12)


@pytest.mark.parametrize("extent", ["hull", "mahalanobis"])
def test_a_flat_hull_or_ellipse_is_refused_when_built(extent: DomainExtent) -> None:
    # Points on the line y = x, and four points in five coordinates: every new member would
    # be outside.
    line = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="flat"):
        domain_from_coordinates(line, np.zeros(3), KEYS, extent=extent)
    with pytest.raises(ValueError, match="at least 6"):
        domain_from_coordinates(
            np.random.default_rng(0).normal(size=(5, 4)), np.zeros(4), KEYS, extent=extent
        )
    # A coordinate that does not vary is the box's, not a flat direction.
    constant = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [5.0, 5.0, 5.0]])
    domain = domain_from_coordinates(constant, np.zeros(3), KEYS, extent=extent)
    assert check_coordinates(domain, np.array([[0.2], [0.2], [5.0]]), np.zeros(1), [0]).inside
    assert check_coordinates(domain, np.array([[0.2], [0.2], [5.1]]), np.zeros(1), [0]).members


@pytest.mark.parametrize("extent", ["box", "hull", "mahalanobis"])
@pytest.mark.parametrize("offset", [0.0, 1e8, 1e10])
def test_validated_points_evaluated_again_with_round_off_stay_inside(
    extent: DomainExtent, offset: float
) -> None:
    """The slack claim: coordinates computed again (other batch sizes round differently) move
    by ulps, and must not leave the domain, whatever their size."""
    rng = np.random.default_rng(4)
    points = offset + rng.normal(size=(3, 12)) * [[1.0], [1e-3], [50.0]]
    domain = domain_from_coordinates(points, np.zeros(12), KEYS, extent=extent)

    for ulps in (1, 4):
        moved = points.copy()
        for _ in range(ulps):
            moved = np.nextafter(moved, np.where(rng.random(moved.shape) < 0.5, -np.inf, np.inf))
        assert check_coordinates(domain, moved, np.zeros(12), range(12)).inside, ulps
    # A coordinate whose span is round-off only (1 and 1 + eps) is the box's alone.
    tiny = np.vstack([rng.normal(size=(2, 12)), 1.0 + np.finfo(float).eps * (np.arange(12) % 2)])
    domain = domain_from_coordinates(tiny, np.zeros(12), KEYS, extent=extent)
    moved = tiny.copy()
    moved[2] = np.nextafter(moved[2], 2.0)
    assert check_coordinates(domain, moved, np.zeros(12), range(12)).inside


@pytest.mark.parametrize("extent", ["box", "hull", "mahalanobis"])
def test_a_coordinate_constant_up_to_ulps_or_round_off_is_left_to_the_box(
    extent: DomainExtent,
) -> None:
    """Review case: 100 + 0..3 ulps over 12 points gave a direction of scale ~1e-14, so the
    validated points fell outside; and 1 + 1e-13 noise made a member at 1 + 1e-11 a distance
    of 106 against a radius of 2.8."""
    rng = np.random.default_rng(6)
    varying = rng.normal(size=(2, 12))
    ulps = np.array([np.nextafter(100.0, np.inf) if k % 4 else 100.0 for k in range(12)])
    for k in range(12):
        for _ in range(k % 4):
            ulps[k] = np.nextafter(ulps[k], np.inf)
    points = np.vstack([varying, ulps])
    domain = domain_from_coordinates(points, np.zeros(12), KEYS, extent=extent)
    assert check_coordinates(domain, points, np.zeros(12), range(12)).inside

    noisy = np.vstack([varying, 1.0 + 1e-13 * rng.normal(size=12)])
    domain = domain_from_coordinates(noisy, np.zeros(12), KEYS, extent=extent)
    member = np.append(varying.mean(axis=1), 1.0 + 1e-11)[:, None]
    assert check_coordinates(domain, member, np.zeros(1), [0]).inside
    if extent == "mahalanobis":
        assert domain.mahalanobis is not None
        assert domain.mahalanobis.coordinates == (0, 1)


@pytest.mark.parametrize("offset", [0.0, 1e4, 1e6, 1e8])
def test_the_hull_lends_no_coordinate_another_ones_slack(offset: float) -> None:
    """Review case: one threshold on t, scaled by the smallest band, let (0.9, 0.9) into the
    triangle x + y <= 1 when y carried a large offset."""
    triangle = np.array([[0.0, 1.0, 0.0], [offset, offset, offset + 1.0]])
    domain = domain_from_coordinates(triangle, np.zeros(3), KEYS, extent="hull")

    members = np.array([[0.9, 0.5, 0.0], [offset + 0.9, offset + 0.5, offset + 1.0]])
    found = check_coordinates(domain, members, np.zeros(3), [0, 1, 2])

    assert found.members == (0,)
    assert found.outside[0].outside_hull


def test_an_infinite_coordinate_is_outside_on_its_side() -> None:
    found = check_coordinates(_box(), np.array([[np.inf], [-np.inf]]), np.zeros(1), [0])
    sides = {c.index: c.side for c in found.outside[0].coordinates}
    assert sides == {0: "above", 1: "below"}


def test_keys_fields_and_times_outside_are_named() -> None:
    domain = domain_from_coordinates(
        np.zeros((1, 2)), np.zeros(2), KEYS, fields=("PRESSURE",), field_times=("2024-03-01",)
    )

    found = check_request(
        domain,
        [KEYS[0], DataKey("2024-01-01", "WWCT:P1")],
        fields=["PRESSURE", "SWAT"],
        times=["2024-03-01", "2024-07-01", "2024-07-02", "3", "not a time"],
    )

    assert found.unknown_keys == (DataKey("2024-01-01", "WWCT:P1"),)
    assert found.unknown_fields == ("SWAT",)
    # The last validated time is the horizon; a step index is never inside a range of dates.
    assert found.times_outside == ("2024-07-02", "3", "not a time")
    assert not found.inside


def test_time_labels_order_on_their_own_axis() -> None:
    assert time_position("9") < time_position("10")
    assert time_position("2024-01-02") > time_position("2024-01-01T23:00:00")
    assert time_position("-1")[0] == 0
    with pytest.raises(ValueError, match="neither"):
        time_position("soon")
    with pytest.raises(ValueError, match="mix"):
        domain_from_coordinates(np.zeros((1, 1)), np.zeros(1), [DataKey("1", "x"), KEYS[0]])


def test_outside_raises_a_domain_error_that_is_not_a_simulation_error() -> None:
    found = check_coordinates(_box(), np.array([[2.0], [0.0]]), np.zeros(1), [7])

    with pytest.raises(SurrogateDomainError) as raised:
        found.raise_if_outside()

    error = raised.value
    assert isinstance(error, ApexError)
    assert not isinstance(error, SimulationError)
    assert error.members == (7,)
    assert "refuses to extrapolate" in str(error)
    assert "member 7" in str(error)
    # Process executors send it back from a worker.
    copy = pickle.loads(pickle.dumps(error))
    assert copy.members == (7,)
    assert str(copy) == str(error)
    check_coordinates(_box(), np.array([[0.5], [0.0]]), np.zeros(1), [7]).raise_if_outside()


@pytest.mark.parametrize("extent", ["box", "hull", "mahalanobis"])
def test_the_domain_round_trips_through_json_exactly(extent: DomainExtent) -> None:
    rng = np.random.default_rng(3)
    points = rng.normal(size=(3, 7)) * [[1e-3], [1.0], [1e5]]
    domain = domain_from_coordinates(
        points,
        rng.random(7),
        KEYS,
        extent=extent,
        input_scale=2.5,
    )

    text = json.dumps(domain.to_document().model_dump(mode="json"))
    back = TrainingDomain.from_document(TrainingDomainDocument.model_validate(json.loads(text)))

    assert back.to_document() == domain.to_document()
    np.testing.assert_array_equal(back.lower, domain.lower)
    if extent == "hull":
        assert back.hull is not None
        assert domain.hull is not None
        np.testing.assert_array_equal(back.hull, domain.hull)
    if extent == "mahalanobis":
        assert back.mahalanobis is not None
        assert domain.mahalanobis is not None
        np.testing.assert_array_equal(back.mahalanobis.basis, domain.mahalanobis.basis)
    members = rng.normal(size=(3, 20)) * [[1e-3], [1.0], [1e5]]
    residuals = rng.random(20)
    assert (
        check_coordinates(back, members, residuals, range(20)).members
        == check_coordinates(domain, members, residuals, range(20)).members
    )
