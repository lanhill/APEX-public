"""Gravity, displacement and stations against SimulatorWrap.

The references (``tests/data/reference/geophysics/{stations,gravity,seafloor_geertsma}.npz``)
are ``measurement_locations``, ``flow_grav.calc_grav`` and ``compute_deformation_transfer``
at ``d29dc57``. APEX's displacement is Geertsma's nucleus of strain in closed form, where the
reference integrates it numerically up to a cut-off; where that cut-off leaves a negligible
tail the two agree to its ``quad`` tolerance, and where it does not the reference's error is
measured (G12).
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import j0

from apex.algorithms.geophysics import Geertsma, point_mass_gravity, regular_stations

pytestmark = pytest.mark.pet_parity

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference" / "geophysics"
type Floats = NDArray[np.float64]


def _load(name: str) -> dict[str, Floats]:
    path = REFERENCE / f"{name}.npz"
    if not path.exists():
        pytest.fail(f"missing {path}; run tools/geophysics_reference/generate.py {name}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def test_stations_are_the_references() -> None:
    ref = _load("stations")
    x, y = ref["in_centres"][0], ref["in_centres"][1]
    plain = regular_stations(x, y, spacing=250.0, padding=700.0, depth=300.0)
    np.testing.assert_array_equal(plain, ref["out_plain"])
    refined = regular_stations(
        x, y, spacing=250.0, padding=700.0, depth=300.0, refine=(330.0, 210.0, 400.0)
    )
    np.testing.assert_array_equal(refined, ref["out_refined"])
    table = ref["in_seabed"]
    seabed = regular_stations(
        x, y, spacing=300.0, padding=500.0, seabed=(table[:, 0], table[:, 1], table[:, 2])
    )
    np.testing.assert_array_equal(seabed, ref["out_seabed"])


def test_point_mass_gravity_is_the_references() -> None:
    """In µGal with the reference's G of 6.67e-11 (APEX's default is CODATA's 6.67430e-11)."""
    ref = _load("gravity")
    centres = ref["out_centres"].T  # the reference's own cell centres, (n_active, 3)
    dg = point_mass_gravity(ref["in_stations"], centres, ref["in_dm"], 6.67e-11) / 1e-8
    np.testing.assert_allclose(dg, ref["out_dg"], rtol=1e-12)


def _geertsma(ref: dict[str, Floats], name: str, kernel: Geertsma) -> tuple[Floats, Floats]:
    """APEX's and the reference's u_z per unit ΔV, ``(n_cell, n_station)``."""
    depths = ref[f"in_{name}_cell_depth"]
    z = ref[f"in_{name}_station_depth"]
    rho = ref[f"in_{name}_rho"]
    ours = np.array(
        [
            kernel.matrix(np.column_stack((rho[j], np.zeros_like(z), z)), [[0.0, 0.0, c]])[:, 0]
            for j, c in enumerate(depths)
        ]
    )
    return ours, ref[f"out_{name}_u"]


def test_geertsma_is_the_references_integral_where_its_cut_off_is_negligible() -> None:
    """rtol 1e-6: the reference's ``quad`` tolerance; its λ_max leaves a tail below 1e-12."""
    ref = _load("seafloor_geertsma")
    kernel = Geertsma(float(ref["in_poisson"]), truncation_factor=3.0, free_surface="datum")
    ours, theirs = _geertsma(ref, "close", kernel)
    np.testing.assert_allclose(ours, theirs, rtol=1e-6)
    # Stations up to 6 times the depth: the same ones beyond 3 times are zeroed (the values
    # of the others carry the reference's cut-off error, G12, below).
    ours, theirs = _geertsma(ref, "truncated", kernel)
    assert np.any(theirs == 0)
    assert np.any(theirs != 0)
    np.testing.assert_array_equal(ours == 0, theirs == 0)


def _integral(nu: float, z: float, c: float, rho: float, lam_max: float) -> float:
    """The reference's ``uHH_integrand`` integrated from 0 to ``lam_max`` (km units)."""
    eps = np.sign(c - z)

    def f(lam: float) -> float:
        terms = eps * np.exp(lam * eps * (z - c)) + (3 - 4 * nu + 2 * z * lam) * np.exp(
            -lam * (z + c)
        )
        return float(lam * terms * j0(lam * rho))

    return float(quad(f, 0, lam_max)[0])


def test_the_references_cut_off_loses_part_of_the_integral_g12() -> None:
    """The reference integrates to λ_max = 15/max rho (km⁻¹) only: that integral reproduces
    it (to its own rounding), so the cut-off is the whole of its difference from the closed
    form, which is exact (checked against the integral to infinity in the unit tests)."""
    ref = _load("seafloor_geertsma")
    nu = float(ref["in_poisson"])
    kernel = Geertsma(nu, 3.0, free_surface="datum")
    for name in ("close", "wide", "truncated"):
        depths = ref[f"in_{name}_cell_depth"] / 1e3
        z = ref[f"in_{name}_station_depth"] / 1e3
        rho = ref[f"in_{name}_rho"] / 1e3
        cut = np.zeros_like(rho)
        for j, c in enumerate(depths):
            # λ_max over every station passed, also those the truncation then zeroes.
            lam_max = 15 / rho[j].max()
            for i in np.flatnonzero(rho[j] <= 3 * np.abs(c - z)):  # the reference's truncation
                cut[j, i] = _integral(nu, z[i], c, rho[j][i], lam_max) / (4 * np.pi) * 1e-6
        np.testing.assert_allclose(cut, ref[f"out_{name}_u"], rtol=1e-10, atol=0)
    # Measured: up to 2.5 % of a cell's largest u_z with stations to the truncation's 3
    # depths (its default receiver_trunc_factor), 15 % with a station list to 6 depths.
    for name, low, high in (("wide", 0.01, 0.04), ("truncated", 0.1, 0.2)):
        ours, theirs = _geertsma(ref, name, kernel)
        error = np.abs(theirs - ours).max() / np.abs(ours).max()
        assert low < error < high, (name, error)
