"""Rock-physics building blocks of the petro-elastic models.

Each function is a closed form on arrays of any common shape (cells first), vectorised,
with no defaults of its own: the models in :mod:`.petro_elastic` hold the parameters.
Moduli and pressures are in the units the caller passes (the models use MPa, as the
reference does, so that the published fits keep their coefficients); temperatures are
in °C; densities in kg/m³ unless stated.

References
----------
Batzle, M. & Wang, Z. (1992). Seismic properties of pore fluids. Geophysics, 57(11),
1396-1408.
Brie, A., Pampuri, F., Marsala, A. F. & Meazza, O. (1995). Shear sonic interpretation in
gas-bearing sands. SPE 30595.
Dvorkin, J. & Nur, A. (1996). Elasticity of high-porosity sandstones: theory for two
North Sea data sets. Geophysics, 61(5), 1363-1370 (the soft-sand model).
Gassmann, F. (1951). Über die Elastizität poröser Medien. Vierteljahrsschrift der
Naturforschenden Gesellschaft in Zürich, 96, 1-23.
Hashin, Z. & Shtrikman, S. (1963). A variational approach to the theory of the elastic
behaviour of multiphase materials. J. Mech. Phys. Solids, 11, 127-140.
Mavko, G., Mukerji, T. & Dvorkin, J. (2009). The Rock Physics Handbook, 2nd ed.,
Cambridge (the Hashin-Shtrikman bounds, Hertz-Mindlin, Gassmann and the soft-sand model).
Wood, A. W. (1955). A Textbook of Sound. Bell.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Array = NDArray[np.float64]


def _f(x: ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def _temperature(t: float | ArrayLike) -> float | Array:
    """Return one temperature as a Python float, or one per cell as an array.

    A scalar keeps Python's float arithmetic (``t**2`` is ``pow``, not NumPy's square), so
    configurations with one temperature give the same bits as before revision 2.
    """
    if np.ndim(t) == 0:
        return float(t)  # type: ignore[arg-type]
    return _f(t)


def hashin_shtrikman_bounds(
    k1: ArrayLike, g1: ArrayLike, k2: ArrayLike, g2: ArrayLike, f2: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    """Return the Hashin-Shtrikman bounds ``(K_lower, K_upper, G_lower, G_upper)``.

    Two isotropic phases, 1 the softer (``K1 ≤ K2``, ``G1 ≤ G2``), with volume fraction
    ``f2`` of phase 2 (Mavko et al. 2009)::

        K_HS = K_a + f_b / (1/(K_b - K_a) + f_a/(K_a + 4/3 G_a))
        G_HS = G_a + f_b / (1/(G_b - G_a) + 2 f_a (K_a + 2 G_a)/(5 G_a (K_a + 4/3 G_a)))

    with ``a`` the softer phase for the lower bound and the stiffer for the upper.
    """
    k1, g1, k2, g2, f2 = _f(k1), _f(g1), _f(k2), _f(g2), _f(f2)
    f1 = 1.0 - f2
    # Identical phases give 1/0 = inf in a denominator and the right answer (the phase's
    # modulus); the warning is silenced, not the arithmetic.

    def bound(
        ka: Array, ga: Array, fa: Array, kb: Array, gb: Array, fb: Array
    ) -> tuple[Array, Array]:
        m = ka + 4.0 * ga / 3.0
        k = ka + fb / (1.0 / (kb - ka) + fa / m)
        g = ga + fb / (1.0 / (gb - ga) + 2.0 * fa * (ka + 2.0 * ga) / (5.0 * ga * m))
        return k, g

    with np.errstate(divide="ignore"):
        k_lo, g_lo = bound(k1, g1, f1, k2, g2, f2)
        k_hi, g_hi = bound(k2, g2, f2, k1, g1, f1)
    return k_lo, k_hi, g_lo, g_hi


def hertz_mindlin_power_law(
    effective_pressure: ArrayLike,
    bulk_max: float,
    shear_max: float,
    reference_pressure: float,
    exponent: float,
) -> tuple[Array, Array]:
    """Return Hertz-Mindlin-type moduli ``M_max (P_eff / P_ref)^κ`` (bulk, shear).

    The empirical form of SimulatorWrap's ``standardrp`` (Fahimuddin 2010, Report 1).
    """
    common = (_f(effective_pressure) / reference_pressure) ** exponent
    return bulk_max * common, shear_max * common


def poisson_ratio(bulk: ArrayLike, shear: ArrayLike) -> Array:
    """Return ``nu = (3K - 2G) / (6K + 2G)``."""
    k, g = _f(bulk), _f(shear)
    out: Array = (3.0 * k - 2.0 * g) / (6.0 * k + 2.0 * g)
    return out


def hertz_mindlin(
    effective_pressure: ArrayLike,
    mineral_bulk: ArrayLike,
    mineral_shear: ArrayLike,
    coordination_number: ArrayLike,
    critical_porosity: ArrayLike,
) -> tuple[Array, Array]:
    """Return the Hertz-Mindlin moduli of a dense random pack at critical porosity.

    Mavko et al. (2009), with ``nu`` the mineral's Poisson's ratio::

        K_HM = [n² (1 - φc)² G² P / (18 π² (1 - nu)²)]^(1/3)
        G_HM = (5 - 4nu)/(5 (2 - nu)) [3 n² (1 - φc)² G² P / (2 π² (1 - nu)²)]^(1/3)
    """
    p, g = _f(effective_pressure), _f(mineral_shear)
    n, phi_c = _f(coordination_number), _f(critical_porosity)
    nu = poisson_ratio(mineral_bulk, mineral_shear)
    common = n**2 * (1.0 - phi_c) ** 2 * g**2 * p
    bulk = np.cbrt(common / (18.0 * np.pi**2 * (1.0 - nu) ** 2))
    shear = (
        (5.0 - 4.0 * nu)
        / (5.0 * (2.0 - nu))
        * np.cbrt(3.0 * common / (2.0 * np.pi**2 * (1.0 - nu) ** 2))
    )
    return bulk, shear


def soft_sand_dry_moduli(
    porosity_ratio: ArrayLike,
    hm_bulk: ArrayLike,
    hm_shear: ArrayLike,
    mineral_bulk: ArrayLike,
    mineral_shear: ArrayLike,
) -> tuple[Array, Array]:
    """Return dry-rock moduli between the grain pack at φc and the mineral (soft sand).

    The lower Hashin-Shtrikman bound between the two end members (Dvorkin & Nur 1996;
    Mavko et al. 2009), with ``q = φ/φc``::

        K_dry = [q/(K_HM + 4/3 G_HM) + (1 - q)/(K + 4/3 G_HM)]^-1 - 4/3 G_HM
        G_dry = [q/(G_HM + z) + (1 - q)/(G + z)]^-1 - z,
        z = G_HM/6 · (9 K_HM + 8 G_HM)/(K_HM + 2 G_HM)

    Both reference models use it: ``standardrp`` with its power-law pack moduli,
    ``softsandrp`` with Hertz-Mindlin's.
    """
    q, khm, ghm = _f(porosity_ratio), _f(hm_bulk), _f(hm_shear)
    k, g = _f(mineral_bulk), _f(mineral_shear)
    zk = 4.0 * ghm / 3.0
    bulk = 1.0 / (q / (khm + zk) + (1.0 - q) / (k + zk)) - zk
    zg = ghm / 6.0 * (9.0 * khm + 8.0 * ghm) / (khm + 2.0 * ghm)
    shear = 1.0 / (q / (ghm + zg) + (1.0 - q) / (g + zg)) - zg
    return bulk, shear


def gassmann(
    dry_bulk: ArrayLike, mineral_bulk: ArrayLike, fluid_bulk: ArrayLike, porosity: ArrayLike
) -> Array:
    """Return the saturated bulk modulus (Gassmann 1951; Mavko et al. 2009).

    ``K_sat = K_dry + (1 - K_dry/K)² / (φ/K_f + (1 - φ)/K - K_dry/K²)``; the shear
    modulus is unchanged.
    """
    kd, k, kf, phi = _f(dry_bulk), _f(mineral_bulk), _f(fluid_bulk), _f(porosity)
    out: Array = kd + (1.0 - kd / k) ** 2 / (phi / kf + (1.0 - phi) / k - kd / k**2)
    return out


def wood(saturations: list[Array], bulk_moduli: list[Array]) -> Array:
    """Return the Reuss (Wood 1955) average ``(Σ S_i / K_i)^-1`` of a fluid mixture."""
    inverse = sum((s / k for s, k in zip(saturations, bulk_moduli, strict=True)), np.zeros(()))
    out: Array = 1.0 / np.asarray(inverse, dtype=np.float64)
    return out


def brie(
    liquid_saturation: ArrayLike, liquid_bulk: ArrayLike, gas_bulk: ArrayLike, exponent: float
) -> Array:
    """Return Brie et al.'s (1995) patchy-mixing fluid modulus.

    ``K_f = (K_liquid - K_gas) S_liquid^e + K_gas``; ``e = 1`` is the Voigt average, and
    large ``e`` approaches ``K_gas``. For ``e > 1`` it falls below Wood's (the Reuss
    bound) as ``S_liquid`` approaches 0, so it is not a bounded mixing law.
    """
    s, kl, kg = _f(liquid_saturation), _f(liquid_bulk), _f(gas_bulk)
    out: Array = (kl - kg) * s**exponent + kg
    return out


# ------------------------------------------------------------------ Batzle & Wang (1992)

#: Batzle & Wang's (1992) Table 1: pure-water velocity ``Σ w_ij T^i P^j`` (m/s, T in °C,
#: P in MPa), valid to about 100 °C and 100 MPa.
WATER_VELOCITY_COEFFICIENTS = np.array(
    [
        [1.40285e03, 1.52400e00, 3.43700e-03, -1.19700e-05],
        [4.87100e00, -1.11000e-02, 1.73900e-04, -1.62800e-06],
        [-4.78300e-02, 2.74700e-04, -2.13500e-06, 1.23700e-08],
        [1.48700e-04, -6.50300e-07, -1.45500e-08, 1.32700e-10],
        [-2.19700e-07, 7.98700e-10, 5.23000e-11, -4.61400e-13],
    ]
)


def water_velocity(temperature_c: float | ArrayLike, pressure_mpa: ArrayLike) -> Array:
    """Return pure water's P velocity (m/s), Batzle & Wang (1992) eq. 28.

    ``temperature_c`` is one temperature, or one per cell, shaped
    as the pressures.
    """
    p = _f(pressure_mpa)
    p_powers = np.stack([p**j for j in range(4)], axis=-1)
    if np.ndim(temperature_c) == 0:
        # One temperature: the matrix products as before, so results stay bit-identical.
        t_powers = float(temperature_c) ** np.arange(5.0)  # type: ignore[arg-type]
        out: Array = p_powers @ (t_powers @ WATER_VELOCITY_COEFFICIENTS)
        return out
    t = np.asarray(temperature_c, dtype=np.float64)
    coefficients = (t[..., np.newaxis] ** np.arange(5.0)) @ WATER_VELOCITY_COEFFICIENTS
    cellwise: Array = np.sum(p_powers * coefficients, axis=-1)
    return cellwise


def brine_velocity(
    temperature_c: float | ArrayLike, pressure_mpa: ArrayLike, salinity: float
) -> Array:
    """Return brine's P velocity (m/s), Batzle & Wang (1992) eq. 29.

    ``salinity`` is the weight fraction of NaCl (35,000 ppm is 0.035).
    """
    p, s = _f(pressure_mpa), salinity
    t = _temperature(temperature_c)
    s1 = 1170 - 9.6 * t + 0.055 * t**2 - 8.5e-5 * t**3 + 2.6 * p - 0.0029 * t * p
    s1 = s1 - 0.0476 * p**2
    s15 = 780 - 10 * p + 0.16 * p**2
    out: Array = water_velocity(t, p) + s1 * s + s15 * s**1.5 - 820 * s**2
    return out


def gas_bulk_modulus(
    temperature_c: float | ArrayLike, pressure_mpa: ArrayLike, gravity: float
) -> Array:
    """Return a gas's adiabatic bulk modulus (MPa), Batzle & Wang (1992) eqs. 9-11.

    ``gravity`` is the gas's specific gravity relative to air (CO₂ 1.5189, methane
    0.5537, as the reference takes them). With the pseudo-reduced pressure ``P_pr`` and
    temperature ``T_pr`` (Thomas et al. 1970)::

        K = P γ₀ / (1 - P_pr/Z ∂Z/∂P_pr)
    """
    p = _f(pressure_mpa)
    t_abs = _temperature(temperature_c) + 273.15
    p_pr = p / (4.892 - 0.4048 * gravity)
    t_pr = t_abs / (94.72 + 170.75 * gravity)
    a = 0.45 + 8.0 * (0.56 - 1.0 / t_pr) ** 2
    e = 0.109 * (3.85 - t_pr) ** 2 * np.exp(-a * p_pr**1.2 / t_pr)
    z = (0.03 + 0.00527 * (3.5 - t_pr) ** 3) * p_pr + (0.642 * t_pr - 0.007 * t_pr**4 - 0.52) + e
    dz_dp = 0.03 + 0.00527 * (3.5 - t_pr) ** 3 - e * a * 1.2 * p_pr**0.2 / t_pr
    gamma0 = 0.85 + 5.6 / (p_pr + 2) + 27.1 / (p_pr + 3.5) ** 2 - 8.7 * np.exp(-0.65 * (p_pr + 1))
    out: Array = p / (1.0 - p_pr / z * dz_dp) * gamma0
    return out


def dead_oil_velocity(
    temperature_c: float | ArrayLike, pressure_mpa: ArrayLike, density_kg_m3: ArrayLike
) -> Array:
    """Return a dead oil's P velocity (m/s), Batzle & Wang (1992) eq. 20a.

    ``density_kg_m3`` is the oil's density at standard conditions, ``ρ₀`` (g/cm³ in the
    paper, converted here). For a live oil, Batzle & Wang (eq. 22) use a pseudo-density
    instead; the in-situ density is a further approximation.
    """
    p, rho0 = _f(pressure_mpa), _f(density_kg_m3) / 1000.0
    t = _temperature(temperature_c)
    out: Array = (
        2096.0 * np.sqrt(rho0 / (2.6 - rho0))
        - 3.7 * t
        + 4.64 * p
        + 0.0115 * (4.12 * np.sqrt(1.08 / rho0 - 1.0) - 1.0) * t * p
    )
    return out
