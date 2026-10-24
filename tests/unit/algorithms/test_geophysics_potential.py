"""Gravity and displacement kernels, against independent references.

- Geertsma's nucleus of strain in closed form against its Hankel integral taken to
  infinity (scipy ``quad``), at stations above and below a cell, and at the free surface
  against Geertsma (1973)'s ``(1 - nu)/pi dV c/R³``.
- A disc reservoir (radius a, depth D, thickness h, compaction c_m ΔP): the uplift over
  its centre against Geertsma's closed form for a disc, ``2 (1 - nu) c_m ΔP h
  (1 - D/√(D² + a²))``, as the disc is refined.
- The truncation's loss for a wide reservoir: ``1/√10`` of the uplift beyond 3 depths.
- The prism (Nagy et al. 2000) against a numerical triple integral near it, and against
  the point mass far from it; a sphere of prisms against its point mass (exact outside a
  sphere).
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad, tplquad
from scipy.special import j0

from apex.algorithms.geophysics import (
    GRAVITATIONAL_CONSTANT,
    Geertsma,
    VanOpstal,
    displacement_kernels,
    point_mass_gravity,
    prism_gravity,
    regular_stations,
)

type Floats = NDArray[np.float64]


@pytest.mark.parametrize(
    ("z", "c", "rho", "nu"),
    [(0.0, 2000.0, 500.0, 0.25), (300.0, 2000.0, 1500.0, 0.25), (300.0, 2400.0, 0.0, 0.3),
     (2500.0, 2000.0, 700.0, 0.2), (150.0, 900.0, 4000.0, 0.45)],
)  # fmt: skip
def test_geertsma_is_its_hankel_integral(z: float, c: float, rho: float, nu: float) -> None:
    """The reference's integrand, ``λ (ε e^{λε(z-c)} + (3-4nu+2zλ) e^{-λ(z+c)}) J₀(λrho)``,
    integrated to infinity, over 4π."""
    eps = np.sign(c - z)

    def integrand(lam: float) -> float:
        return float(
            lam
            * (
                eps * np.exp(lam * eps * (z - c))
                + (3 - 4 * nu + 2 * z * lam) * np.exp(-lam * (z + c))
            )
            * j0(lam * rho)
        )  # fmt: skip

    expected = quad(integrand, 0.0, np.inf, limit=500)[0] / (4 * np.pi)
    got = Geertsma(nu, free_surface="datum").matrix([[rho, 0.0, z]], [[0.0, 0.0, c]])[0, 0]
    assert got == pytest.approx(expected, rel=1e-8)


def test_at_the_free_surface_geertsma_is_his_closed_form() -> None:
    rng = np.random.default_rng(3)
    centres = np.column_stack((rng.uniform(-500, 500, 20), rng.uniform(-500, 500, 20),
                               rng.uniform(800, 3000, 20)))  # fmt: skip
    stations = np.column_stack(
        (rng.uniform(-3e3, 3e3, 15), rng.uniform(-3e3, 3e3, 15), np.zeros(15))
    )
    nu = 0.27
    d = stations[:, np.newaxis, :] - centres[np.newaxis, :, :]
    r3 = np.linalg.norm(d, axis=2) ** 3
    expected = (1 - nu) / np.pi * centres[np.newaxis, :, 2] / r3
    np.testing.assert_allclose(Geertsma(nu).matrix(stations, centres), expected, rtol=1e-13)


def _disc_cells(radius: float, depth: float, n: int) -> tuple[Floats, Floats]:
    """Square cells of a disc's horizontal section (side ``2 radius / n``) at ``depth``."""
    side = 2 * radius / n
    x = -radius + side * (np.arange(n) + 0.5)
    gx, gy = np.meshgrid(x, x)
    inside = gx**2 + gy**2 <= radius**2
    centres = np.column_stack((gx[inside], gy[inside], np.full(int(inside.sum()), depth)))
    return centres, np.full(int(inside.sum()), side * side)


def test_a_compacting_disc_subsides_as_geertsmas_disc() -> None:
    """Geertsma (1973): over the centre of a disc of radius a at depth D, thickness h,
    uniaxial compaction c_m ΔP h, u_z = 2 (1 - nu) c_m ΔP h (1 - D/√(D² + a²))."""
    a, depth, h, nu, compaction = 1500.0, 2000.0, 30.0, 0.25, 1e-4  # compaction: c_m ΔP
    expected = 2 * (1 - nu) * compaction * h * (1 - depth / np.hypot(depth, a))
    errors = []
    for n in (20, 40, 80):
        centres, area = _disc_cells(a, depth, n)
        volume = compaction * h * area  # each column's volume decrease
        u = Geertsma(nu).matrix([[0.0, 0.0, 0.0]], centres) @ volume
        errors.append(abs(u[0] - expected) / expected)
    assert errors[-1] < 2e-3  # the disc's staircase edge
    assert errors[0] > errors[-1]


def test_truncation_keeps_one_minus_one_over_root_ten_of_a_wide_reservoirs_uplift() -> None:
    """A reservoir much wider than deep: stations zeroed beyond 3 depths keep the uplift of
    a disc of radius 3 D, ``1 - D/√(D² + 9 D²) = 1 - 1/√10`` of the whole."""
    depth = 1000.0
    centres, area = _disc_cells(40 * depth, depth, 400)
    whole = Geertsma(0.25).matrix([[0.0, 0.0, 0.0]], centres) @ area
    kept = Geertsma(0.25, truncation_factor=3.0).matrix([[0.0, 0.0, 0.0]], centres) @ area
    assert kept[0] / whole[0] == pytest.approx(
        (1 - 1 / np.sqrt(10)) / (1 - 1 / np.hypot(1, 40)), rel=5e-3
    )


def test_geertsma_refuses_what_it_cannot_model() -> None:
    with pytest.raises(ValueError, match="poisson"):
        Geertsma(0.6)
    with pytest.raises(ValueError, match="truncation_factor"):
        Geertsma(0.25, truncation_factor=0.0)
    with pytest.raises(ValueError, match="singular"):
        Geertsma(free_surface="datum").matrix([[0.0, 0.0, 1000.0]], [[0.0, 0.0, 1000.0]])
    with pytest.raises(ValueError, match="above the free surface"):
        Geertsma(free_surface="datum").matrix([[0.0, 0.0, -10.0]], [[0.0, 0.0, 1000.0]])
    with pytest.raises(ValueError, match="not below a station"):
        Geertsma().matrix([[0.0, 0.0, 1200.0]], [[0.0, 0.0, 1000.0]])
    with pytest.raises(ValueError, match="free_surface"):
        Geertsma(free_surface="sea")  # type: ignore[arg-type]
    assert isinstance(displacement_kernels.create("geertsma", poisson=0.3), Geertsma)


def test_a_prism_is_its_triple_integral_near_it() -> None:
    """``G rho ∫∫∫ (z' - z_s)/r³`` over the prism, numerically, at stations close to and
    beside it (where a point mass is far off)."""
    bounds = np.array([[-60.0, 40.0, -30.0, 50.0, 200.0, 240.0]])
    rho = 300.0
    for station in ([0.0, 0.0, 150.0], [70.0, -50.0, 190.0], [0.0, 0.0, 0.0]):
        sx, sy, sz = station

        def f(
            z: float, y: float, x: float, sx: float = sx, sy: float = sy, sz: float = sz
        ) -> float:
            return float((z - sz) / ((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) ** 1.5)

        x1, x2, y1, y2, z1, z2 = bounds[0]
        integral = tplquad(f, x1, x2, y1, y2, z1, z2, epsabs=1e-12, epsrel=1e-10)[0]
        expected = GRAVITATIONAL_CONSTANT * rho * integral
        got = prism_gravity([station], bounds, [rho])[0]
        assert got == pytest.approx(expected, rel=1e-7)


def test_a_prism_far_away_is_a_point_mass() -> None:
    """Approaching it as (size/distance)², and taken as it beyond 50 diagonals (6.2 km here),
    where the prism's cancelling corner terms would lose precision."""
    bounds = np.array([[-50.0, 50.0, -30.0, 30.0, 2000.0, 2040.0]])
    rho, volume = 250.0, 100.0 * 60.0 * 40.0
    errors = []
    for distance in (300.0, 1200.0, 4800.0, 1e5):
        h = distance / np.sqrt(2)
        station = [[h, 0.0, 2020.0 - h]]  # at 45 degrees above the prism
        prism = prism_gravity(station, bounds, [rho])[0]
        point = point_mass_gravity(station, [[0.0, 0.0, 2020.0]], [rho * volume])[0]
        errors.append(abs(prism - point) / abs(point))
    assert errors[0] > errors[1] > errors[2] > 0  # (size / distance)², near
    assert errors[2] < 1e-3
    assert errors[3] == 0.0  # far: the point mass itself


def test_a_sphere_of_prisms_pulls_as_its_mass_at_its_centre() -> None:
    """Outside a uniform sphere g = G M z/r³ exactly; a sphere of 1/16 R cubes approaches it."""
    radius, depth, rho = 400.0, 1500.0, 500.0
    side = radius / 8
    x = -radius + side * (np.arange(16) + 0.5)
    gx, gy, gz = np.meshgrid(x, x, x)
    inside = gx**2 + gy**2 + gz**2 <= radius**2
    c = np.column_stack((gx[inside], gy[inside], gz[inside] + depth))
    half = side / 2
    bounds = np.column_stack((c[:, 0] - half, c[:, 0] + half, c[:, 1] - half,
                              c[:, 1] + half, c[:, 2] - half, c[:, 2] + half))  # fmt: skip
    stations = np.array([[0.0, 0.0, 0.0], [900.0, 500.0, 100.0]])
    mass = rho * 4 / 3 * np.pi * radius**3
    sphere = point_mass_gravity(stations, [[0.0, 0.0, depth]], [mass])
    np.testing.assert_allclose(
        prism_gravity(stations, bounds, np.full(len(c), rho)), sphere, rtol=2e-2
    )
    # The point masses of the cubes are the sphere's too, to the voxels' volume error.
    cubes = point_mass_gravity(stations, c, np.full(len(c), rho * side**3))
    np.testing.assert_allclose(cubes, sphere, rtol=2e-2)


def test_gravity_refuses_a_station_on_a_mass_and_bad_prisms() -> None:
    with pytest.raises(ValueError, match="singular"):
        point_mass_gravity([[0.0, 0.0, 10.0]], [[0.0, 0.0, 10.0]], [1.0])
    assert point_mass_gravity([[0.0, 0.0, 10.0]], [[0.0, 0.0, 10.0]], [0.0])[0] == 0.0
    with pytest.raises(ValueError, match="x1 < x2"):
        prism_gravity([[0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0, 1.0, 0.0, 1.0]], [1.0])
    # A station on a prism's edge line (x = y = 0 relative to two corners) stays finite.
    g = prism_gravity([[0.0, 0.0, 0.0]], [[0.0, 10.0, 0.0, 10.0, 5.0, 15.0]], [1.0])
    assert np.isfinite(g).all()


def test_stations_cover_the_cells_and_refine_near_a_well() -> None:
    x, y = np.array([0.0, 1000.0, 500.0]), np.array([0.0, 400.0, 800.0])
    plain = regular_stations(x, y, spacing=200.0, padding=300.0, depth=250.0)
    assert plain[:, 0].min() == -300.0
    assert plain[:, 0].max() == 1300.0
    assert np.all(plain[:, 2] == 250.0)
    assert plain[1, 1] == plain[0, 1]  # x fastest
    refined = regular_stations(x, y, spacing=200.0, padding=300.0, refine=(500.0, 400.0, 250.0))
    near = np.hypot(refined[:, 0] - 500.0, refined[:, 1] - 400.0) <= 250.0
    assert near.sum() > np.sum(np.hypot(plain[:, 0] - 500.0, plain[:, 1] - 400.0) <= 250.0)
    with pytest.raises(ValueError, match="spacing"):
        regular_stations(x, y, spacing=0.0, padding=0.0)


def test_seabed_stations_are_on_the_free_surface_by_default() -> None:
    """Water carries no shear: a seabed station at depth w over a cell at c sees Geertsma's
    free-surface form at depth c - w; the reference's datum mode puts it inside the rock."""
    nu, w, c = 0.25, 300.0, 2000.0
    stations = np.array([[0.0, 0.0, w], [1000.0, 0.0, w], [3000.0, 0.0, w]])
    rho = stations[:, 0]
    expected = (1 - nu) / np.pi * (c - w) / np.hypot(rho, c - w) ** 3
    seabed = Geertsma(nu).matrix(stations, [[0.0, 0.0, c]])[:, 0]
    np.testing.assert_allclose(seabed, expected, rtol=1e-13)
    datum = Geertsma(nu, free_surface="datum").matrix(stations, [[0.0, 0.0, c]])[:, 0]
    assert datum[0] / seabed[0] == pytest.approx(0.79, abs=0.01)  # G13, directly above
    # Onshore, a station above the datum (z < 0) is on the ground: the cell is deeper.
    onshore = Geertsma(nu).matrix([[0.0, 0.0, -100.0]], [[0.0, 0.0, c]])[0, 0]
    assert onshore == pytest.approx((1 - nu) / np.pi / (c + 100.0) ** 2, rel=1e-13)


def test_many_stations_are_computed_in_chunks_alike() -> None:
    rng = np.random.default_rng(5)
    stations = np.column_stack((rng.uniform(-3e3, 3e3, 150), rng.uniform(-3e3, 3e3, 150),
                                rng.uniform(0, 300, 150)))  # fmt: skip
    centres = np.column_stack((rng.uniform(-1e3, 1e3, 40), rng.uniform(-1e3, 1e3, 40),
                               rng.uniform(1500, 2500, 40)))  # fmt: skip
    for kernel in (Geertsma(), Geertsma(free_surface="datum", truncation_factor=3.0)):
        whole = kernel.matrix(stations, centres)
        rows = np.vstack([kernel.matrix(stations[i : i + 1], centres) for i in range(150)])
        np.testing.assert_array_equal(whole, rows)


# --- van Opstal: a rigid basement (Tempone, Fjær & Landrø 2010) ----------------------------


def _paper_unscaled(nu: float, k: float, c: float, z: float, r: float) -> float:
    """System 3 of Tempone et al. (2010), eqs. 26 and 28-32 as printed (ε = -1), per
    unit volume change: the basement's part of u_z, integrated with ``quad``. The upper
    limit 300/k is ample while 2k - c - z is not much below k/6, as in these cases."""

    def w3(lam: float) -> float:
        kl = k * lam
        big_l = (4 * nu - 3 + 2 * kl) * np.exp(-lam * (k + c)) - np.exp(-lam * (k - c))
        big_m = (4 * nu - 3 - 2 * kl) * np.exp(-lam * (k + c)) + np.exp(-lam * (k - c))
        delta = (4 * nu - 3) * np.cosh(kl) ** 2 - kl**2 - (1 - 2 * nu) ** 2
        b = (
            lam
            * (
                big_l * (2 * (1 - nu) * np.cosh(kl) - kl * np.sinh(kl))
                + big_m * ((1 - 2 * nu) * np.sinh(kl) + kl * np.cosh(kl))
            )
            / delta
        )
        c_ = (
            lam
            * (
                big_l * ((1 - 2 * nu) * np.sinh(kl) - kl * np.cosh(kl))
                + big_m * (2 * (1 - nu) * np.cosh(kl) + kl * np.sinh(kl))
            )
            / delta
        )
        zb = lam * z * np.cosh(z * lam) - (1 - 2 * nu) * np.sinh(z * lam)
        zc = 2 * (1 - nu) * np.cosh(z * lam) - lam * z * np.sinh(z * lam)
        return float(-(b * zb + c_ * zc) * j0(lam * r) / (4 * np.pi))  # fmt: skip

    # cosh(kλ)² overflows near kλ = 355; the integrand is below 1e-120 well before.
    return float(quad(w3, 0.0, 300.0 / k, limit=500, epsabs=1e-22)[0])


@pytest.mark.parametrize(
    ("z", "r", "nu"),
    [(0.0, 0.0, 0.25), (0.0, 300.0, 0.25), (0.0, 1500.0, 0.25), (0.0, 5000.0, 0.25),
     (400.0, 700.0, 0.25), (950.0, 100.0, 0.25), (0.0, 900.0, 0.35), (300.0, 2000.0, 0.1)],
)  # fmt: skip
def test_van_opstal_is_the_papers_equations(z: float, r: float, nu: float) -> None:
    """The scaled form and Gauss-Legendre quadrature against eqs. 26, 28-32 as printed."""
    k, c = 1000.0, 800.0
    expected = _paper_unscaled(nu, k, c, z, r)
    half_space = Geertsma(nu, free_surface="datum").matrix([[r, 0.0, z]], [[0.0, 0.0, c]])
    got = VanOpstal(k, nu, free_surface="datum").matrix([[r, 0.0, z]], [[0.0, 0.0, c]])
    assert (got - half_space)[0, 0] == pytest.approx(expected, rel=1e-9, abs=1e-22)


def test_van_opstals_displacement_vanishes_at_the_rigid_basement() -> None:
    """u_z = 0 at z = k (the boundary condition the basement imposes), below and beside."""
    k = 1000.0
    for r in (0.0, 200.0, 600.0, 2500.0):
        station, cell = [[r, 0.0, k]], [[0.0, 0.0, 800.0]]
        half_space = Geertsma(free_surface="datum").matrix(station, cell)[0, 0]
        total = VanOpstal(k, free_surface="datum").matrix(station, cell)[0, 0]
        assert abs(total) < 1e-10 * abs(half_space)


def test_a_deep_basement_is_geertsmas_half_space() -> None:
    stations = np.array([[0.0, 0.0, 0.0], [700.0, 0.0, 0.0], [2000.0, 300.0, 0.0]])
    cells = np.array([[0.0, 0.0, 800.0], [200.0, 100.0, 1500.0]])
    deep = VanOpstal(1e6).matrix(stations, cells)
    np.testing.assert_allclose(deep, Geertsma().matrix(stations, cells), rtol=2e-5)


def test_van_opstal_reproduces_tempone_et_al_figure_3() -> None:
    """Tempone et al. (2010), Figs. 3-4: a cylinder of radius 500 m, 100 m thick, centred at
    800 m over a basement at 1000 m, nu = 0.25, 715 nuclei; their solution coincides with
    van Opstal's (Fig. 3). Digitised from the figures (±0.001 m): the centre's subsidence
    is 0.0945 m with the basement and 0.061 m without (Fig. 4), a ratio of 1.55, and the
    profile falls to 0.0645, 0.022 and 0.0045 m at 500, 1000 and 1500 m. The figures' scale
    is about 7 % above Geertsma's own disc formula, also for the half space, so the ratio and
    the profile relative to the centre are compared: the paper's absolute values match a
    slightly larger body (about 8 % more volume), to which the ratio is insensitive."""
    a, n = 500.0, 715
    side = np.sqrt(np.pi * a * a / n)
    x = np.arange(-a, a + side, side)
    gx, gy = np.meshgrid(x - x.mean(), x - x.mean())
    inside = gx**2 + gy**2 <= a * a
    cells = np.column_stack((gx[inside], gy[inside], np.full(int(inside.sum()), 800.0)))
    volume = np.full(len(cells), 2.5e-4 * 10.0 * 100.0 * side**2)  # C_m Δp V
    stations = np.array(
        [[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1500.0, 0.0, 0.0]]
    )
    basement = VanOpstal(1000.0).matrix(stations, cells) @ volume
    half = Geertsma().matrix(stations[:1], cells) @ volume
    # 0.061 is read off Fig. 4's 0.05-per-tick axis (about ±3 %).
    assert basement[0] / half[0] == pytest.approx(0.0945 / 0.061, rel=0.04)
    # Digitised to ±0.001 m, ±0.011 relative to the centre.
    paper = np.array([0.0945, 0.0645, 0.022, 0.0045]) / 0.0945
    np.testing.assert_allclose(basement / basement[0], paper, rtol=0, atol=0.012)


def test_van_opstal_refuses_cells_or_stations_beyond_the_basement() -> None:
    with pytest.raises(ValueError, match="basement_depth"):
        VanOpstal(0.0)
    with pytest.raises(ValueError, match="not above the rigid basement"):
        VanOpstal(1000.0).matrix([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1000.0]])
    with pytest.raises(ValueError, match="below the rigid basement"):
        VanOpstal(1000.0, free_surface="datum").matrix([[0.0, 0.0, 1100.0]], [[0.0, 0.0, 800.0]])
    assert isinstance(displacement_kernels.create("van_opstal", basement_depth=900.0), VanOpstal)


def test_seabed_stations_count_the_basement_from_the_seabed_too() -> None:
    """free_surface = "stations": a station at depth w is the free surface, so the cell and
    the basement are at c - w and k - w below it."""
    w, k, c = 300.0, 1500.0, 1100.0
    seabed = VanOpstal(k).matrix([[200.0, 0.0, w]], [[0.0, 0.0, c]])
    shifted = VanOpstal(k - w, free_surface="datum").matrix(
        [[200.0, 0.0, 0.0]], [[0.0, 0.0, c - w]]
    )
    np.testing.assert_allclose(seabed, shifted, rtol=1e-14)


def test_van_opstal_refuses_what_its_quadrature_cannot_resolve() -> None:
    """A station some 140 decay lengths (2k - c - z) from a cell: J₀ would need more nodes
    than the cap, which is refused rather than integrated wrongly."""
    kernel = VanOpstal(1000.0, free_surface="datum")
    with pytest.raises(ValueError, match="quadrature nodes"):
        kernel.matrix([[2000.0, 0.0, 999.5]], [[0.0, 0.0, 999.0]])
    # Cells of other depths get their own range: a near-basement cell does not set them.
    near, far = [[0.0, 0.0, 990.0]], [[300.0, 0.0, 600.0]]
    both = kernel.matrix([[100.0, 0.0, 50.0]], np.vstack((near, far)))
    np.testing.assert_allclose(
        both[:, 1], kernel.matrix([[100.0, 0.0, 50.0]], far)[:, 0], rtol=1e-12
    )


def test_gravity_works_in_bounded_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression (the phase-3 re-review): the work arrays were 64 stations x every
    cell, gigabytes per member on a large grid. Blocks of a few pairs give the same."""
    from apex.algorithms.geophysics import gravity

    rng = np.random.default_rng(8)
    stations = np.column_stack((rng.uniform(-500, 500, (9, 2)), np.zeros(9)))
    centres = np.column_stack((rng.uniform(-400, 400, (13, 2)), rng.uniform(900, 1100, 13)))
    bounds = np.column_stack(
        (centres[:, 0] - 20, centres[:, 0] + 20, centres[:, 1] - 20, centres[:, 1] + 20,
         centres[:, 2] - 5, centres[:, 2] + 5)
    )  # fmt: skip
    mass = rng.normal(size=13)
    whole = (gravity.point_mass_gravity(stations, centres, mass),
             gravity.prism_gravity(stations, bounds, mass))  # fmt: skip
    for budget in (1, 5, 13, 40):  # cells split, one station a block, several stations
        monkeypatch.setattr(gravity, "_ELEMENTS", budget)
        blocks = gravity._blocks(9, 13)
        assert all(
            (r.stop - r.start) * (min(c.stop, 13) - c.start) <= max(budget, 13) for r, c in blocks
        )
        np.testing.assert_allclose(
            gravity.point_mass_gravity(stations, centres, mass), whole[0], rtol=1e-13
        )
        np.testing.assert_allclose(
            gravity.prism_gravity(stations, bounds, mass), whole[1], rtol=1e-13
        )
