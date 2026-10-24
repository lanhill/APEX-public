"""PP reflection coefficients of a planar elastic interface.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``flow_avo``: ``pp_func``, a PyLops
             function chosen by name, L1676, called per angle at L1896)
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
References : Aki, K. & Richards, P. G. (1980). Quantitative Seismology, eqs. 5.40 (exact PP)
             and 5.44 (linearised). W. H. Freeman.
             Fatti, J. L., Smith, G. C., Vail, P. J., Strauss, P. J. & Levitt, P. R. (1994).
             Detection of gas in sandstone reservoirs using AVO analysis. Geophysics, 59(9),
             1362-1376.
             Shuey, R. T. (1985). A simplification of the Zoeppritz equations. Geophysics,
             50(4), 609-614 (the three-term form of Avseth et al. 2005, Quantitative
             Seismic Interpretation, Cambridge).
             Mavko, G., Mukerji, T. & Dvorkin, J. (2009). The Rock Physics Handbook, 2nd ed.,
             §3.4. Cambridge.
Behaviour  : the reference calls PyLops (``pylops.avo.avo``). Of its four names only
             ``approx_zoeppritz_pp`` accepts the reference's arrays (``zoeppritz_pp``
             takes scalars only; ``akirichards`` and ``fatti`` are coefficient functions
             of (angle, Vs/Vp) with another signature).
             ``approx_zoeppritz_pp`` is, despite its name, Aki & Richards' explicit exact
             solution (eq. 5.40), the same numbers as the 4 x 4 scattering matrix; here it
             is an alias of ``zoeppritz``. Checked against PyLops 2.8.0 and bruges 0.5.4
             (stored references, ``tools/geophysics_reference/generate.py``).
Differences: every model refuses evanescent waves (:class:`EvanescentWaveError`), where
             PyLops' arcsin returns NaN with a warning.

Shapes: the six elastic arrays broadcast together to the interfaces' shape ``S`` (e.g.
``(n_trace, n_interface)``), and ``angles_deg`` is ``(n_angle,)``; every model returns
``(*S, n_angle)``. Velocities are in m/s and densities in kg/m³ (any consistent units
serve: the coefficients depend on ratios only). Vs must be positive: a fluid layer (a water
column, Vs = 0) is outside these models (Zoeppritz divides by Vs), so a water-bottom
interface would need a fluid-solid form. Angles are incidence angles in degrees,
the same at every interface (no ray tracing, as the reference).
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geophysics.errors import GeophysicsModelError
from apex.algorithms.geophysics.registries import reflectivity

type Array = NDArray[np.float64]

#: Relative slack on ``sin(angle) = 1`` of a transmitted or converted wave: a critical
#: angle computed in floating point lands within a few ulps of it.
_CRITICAL_SLACK = 1e-12


class EvanescentWaveError(GeophysicsModelError):
    """An interface's transmitted (or converted) wave is evanescent at a configured angle.

    Exact PP reflectivity is complex there, and PyLops and the textbook Aki-Richards
    return NaN (through arcsin). Fatti, Shuey and the linear Aki-Richards stay finite,
    but past a critical angle the linearisations mean nothing physically, so every model
    refuses, as the exact one must: the member fails rather than giving NaN or a
    coefficient no rock could produce.

    Attributes
    ----------
    index
        The first such interface, as an index into the interfaces' shape ``S``.
    angle_deg
        The incidence angle (degrees).
    wave
        ``"transmitted P"``, ``"transmitted S"`` or ``"reflected S"``.
    count
        How many (interface, angle) pairs are evanescent.
    """

    def __init__(self, index: tuple[int, ...], angle_deg: float, wave: str, count: int) -> None:
        self.index = index
        self.angle_deg = angle_deg
        self.wave = wave
        self.count = count
        super().__init__(
            f"the {wave} wave is evanescent at interface {index} for an incidence angle of "
            f"{angle_deg:g} degrees ({count} interface-angle pair(s) in all): its velocity "
            "exceeds the upper P velocity / sin(angle). Reduce the angles, or check the "
            "velocities"
        )


@dataclass(frozen=True)
class _Interfaces:
    """Validated inputs, each ``(*S, 1)`` so that they broadcast against the angles."""

    vp1: Array
    vs1: Array
    rho1: Array
    vp2: Array
    vs2: Array
    rho2: Array
    theta: Array  # (n_angle,), radians
    p: Array  # ray parameter (*S, n_angle), s/m
    cos_p2: Array  # cosine of the transmitted P angle (*S, n_angle)
    cos_s1: Array  # cosine of the reflected S angle
    cos_s2: Array  # cosine of the transmitted S angle


def _interfaces(
    vp1: ArrayLike,
    vs1: ArrayLike,
    rho1: ArrayLike,
    vp2: ArrayLike,
    vs2: ArrayLike,
    rho2: ArrayLike,
    angles_deg: ArrayLike,
) -> _Interfaces:
    names = ("vp1", "vs1", "rho1", "vp2", "vs2", "rho2")
    raw = [np.asarray(x, dtype=np.float64) for x in (vp1, vs1, rho1, vp2, vs2, rho2)]
    try:
        arrays = np.broadcast_arrays(*raw)
    except ValueError as exc:
        shapes = ", ".join(f"{n} {a.shape}" for n, a in zip(names, raw, strict=True))
        raise ValueError(f"the elastic arrays do not broadcast together: {shapes}") from exc
    for name, a in zip(names, arrays, strict=True):
        bad = ~(np.isfinite(a) & (a > 0))
        if bad.any():
            where = tuple(int(i) for i in np.argwhere(bad)[0])
            raise GeophysicsModelError(
                f"{name} must be finite and positive at every interface; at {where} it is "
                f"{a[where]!r} (a fluid layer, Vs = 0, is outside these models)"
            )
    # A positive bulk modulus, K = rho (Vp² - 4/3 Vs²) > 0, needs Vs < (√3/2) Vp.
    for side, vp, vs in (("upper", arrays[0], arrays[1]), ("lower", arrays[3], arrays[4])):
        bad = vs >= np.sqrt(3.0) / 2.0 * vp
        if bad.any():
            where = tuple(int(i) for i in np.argwhere(bad)[0])
            raise GeophysicsModelError(
                f"the {side} medium at {where} has Vs = {vs[where]:g} >= (√3/2) Vp = "
                f"{np.sqrt(3.0) / 2.0 * vp[where]:g}: a negative bulk modulus"
            )
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError(f"angles_deg must be a non-empty 1-D array, got shape {angles.shape}")
    if not (np.all(np.isfinite(angles)) and np.all(angles >= 0) and np.all(angles < 90)):
        raise ValueError(f"incidence angles must be in [0, 90) degrees, got {angles.tolist()}")
    e = [a[..., np.newaxis] for a in arrays]
    theta = np.radians(angles)
    p = np.sin(theta) / e[0]
    shape = p.shape[:-1]
    cosines = []
    for wave, v in (("transmitted P", e[3]), ("reflected S", e[1]), ("transmitted S", e[4])):
        s = p * v
        over = s > 1.0 + _CRITICAL_SLACK
        if over.any():
            first = np.argwhere(over)[0]
            raise EvanescentWaveError(
                index=tuple(int(i) for i in first[:-1]) if shape else (),
                angle_deg=float(angles[first[-1]]),
                wave=wave,
                count=int(over.sum()),
            )
        cosines.append(np.sqrt(np.maximum(1.0 - s * s, 0.0)))
    return _Interfaces(
        vp1=e[0], vs1=e[1], rho1=e[2], vp2=e[3], vs2=e[4], rho2=e[5], theta=theta, p=p,
        cos_p2=cosines[0], cos_s1=cosines[1], cos_s2=cosines[2],
    )  # fmt: skip


def _means(i: _Interfaces) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return the means and relative contrasts ``Δx / x̄`` of Vp, Vs and rho."""
    vp, vs, rho = (i.vp1 + i.vp2) / 2, (i.vs1 + i.vs2) / 2, (i.rho1 + i.rho2) / 2
    return vp, vs, rho, (i.vp2 - i.vp1) / vp, (i.vs2 - i.vs1) / vs, (i.rho2 - i.rho1) / rho


class PPReflectivity(Protocol):
    """A PP reflection-coefficient model (registry ``reflectivity``)."""

    def pp(
        self,
        vp1: ArrayLike,
        vs1: ArrayLike,
        rho1: ArrayLike,
        vp2: ArrayLike,
        vs2: ArrayLike,
        rho2: ArrayLike,
        angles_deg: ArrayLike,
    ) -> Array:
        """Return R_PP ``(*S, n_angle)`` of the interfaces from medium 1 (above) to 2.

        Raises
        ------
        EvanescentWaveError
            If a transmitted or converted wave is evanescent at an angle.
        ValueError
            If an elastic value is not finite and positive, or an angle is outside
            [0, 90).
        """
        ...


@reflectivity.register("zoeppritz")
class Zoeppritz:
    """Exact PP reflectivity: Aki & Richards' (1980) explicit solution, eq. 5.40.

    The cosines of the transmitted and converted angles are ``sqrt(1 - (p v)^2)`` from the
    ray parameter ``p = sin(θ₁)/Vp₁``, not ``cos(arcsin(.))``, which loses accuracy near
    a critical angle.
    """

    def pp(
        self,
        vp1: ArrayLike,
        vs1: ArrayLike,
        rho1: ArrayLike,
        vp2: ArrayLike,
        vs2: ArrayLike,
        rho2: ArrayLike,
        angles_deg: ArrayLike,
    ) -> Array:
        """Return R_PP ``(*S, n_angle)``; see :class:`PPReflectivity`."""
        i = _interfaces(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg)
        p2 = i.p * i.p
        # 1 - 2 sin²φ of the S waves, from sin φ = p Vs.
        q1 = 1.0 - 2.0 * i.vs1**2 * p2
        q2 = 1.0 - 2.0 * i.vs2**2 * p2
        a = i.rho2 * q2 - i.rho1 * q1
        b = i.rho2 * q2 + 2.0 * i.rho1 * i.vs1**2 * p2
        c = i.rho1 * q1 + 2.0 * i.rho2 * i.vs2**2 * p2
        d = 2.0 * (i.rho2 * i.vs2**2 - i.rho1 * i.vs1**2)
        ci1 = np.cos(i.theta) / i.vp1  # cos(i1)/Vp1
        ci2 = i.cos_p2 / i.vp2
        cj1 = i.cos_s1 / i.vs1
        cj2 = i.cos_s2 / i.vs2
        e = b * ci1 + c * ci2
        f = b * cj1 + c * cj2
        g = a - d * ci1 * cj2
        h = a - d * ci2 * cj1
        den = e * f + g * h * p2
        out: Array = ((b * ci1 - c * ci2) * f - (a + d * ci1 * cj2) * h * p2) / den
        return out


@reflectivity.register("approx_zoeppritz_pp")
class ApproxZoeppritzPP(Zoeppritz):
    """PyLops' ``approx_zoeppritz_pp``, the 3dBox example's choice: the same as ``zoeppritz``.

    PyLops names it approximate, but its formula is Aki & Richards' explicit exact
    solution (eq. 5.40, as Dvorkin et al. 2014 give it); the stored PyLops references
    agree with ``zoeppritz`` to rounding.
    """


@reflectivity.register("aki_richards")
class AkiRichards:
    """Aki & Richards' (1980) linearised PP reflectivity, eq. 5.44.

    Parameters
    ----------
    form
        ``"textbook"`` (the default; bruges' ``akirichards``)::

            R = ½(1 - 4Vs_m²p²) Δrho/rho_m + ΔVp/(2Vp_m cos²θ̄) - 4Vs_m²p² ΔVs/Vs_m

        with ``θ̄`` the mean of the incidence and transmission angles and ``p`` the ray
        parameter. ``"linear"`` (PyLops' ``akirichards`` coefficients, linear in the
        contrasts, as its AVO modelling operator uses)::

            R = ΔVp/Vp_m / (2 cos²θ₁) - 4 r² sin²θ₁ ΔVs/Vs_m
                + (½ - 2 r² sin²θ₁) Δrho/rho_m,          r = Vs_m/Vp_m

        Both agree to first order in the contrasts.
    """

    def __init__(self, form: Literal["textbook", "linear"] = "textbook") -> None:
        if form not in ("textbook", "linear"):
            raise ValueError(f"form must be 'textbook' or 'linear', got {form!r}")
        self.form = form

    def pp(
        self,
        vp1: ArrayLike,
        vs1: ArrayLike,
        rho1: ArrayLike,
        vp2: ArrayLike,
        vs2: ArrayLike,
        rho2: ArrayLike,
        angles_deg: ArrayLike,
    ) -> Array:
        """Return R_PP ``(*S, n_angle)``; see :class:`PPReflectivity`."""
        i = _interfaces(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg)
        vp, vs, _, dvp, dvs, drho = _means(i)
        out: Array
        if self.form == "textbook":
            theta2 = np.arcsin(np.minimum(i.p * i.vp2, 1.0))
            mean_theta = (i.theta + theta2) / 2
            k = 4.0 * vs**2 * i.p**2  # 4 Vs_m² p²
            out = 0.5 * (1.0 - k) * drho + dvp / (2.0 * np.cos(mean_theta) ** 2) - k * dvs
            return out
        sin2 = np.sin(i.theta) ** 2
        r2 = (vs / vp) ** 2
        out = dvp / (2.0 * np.cos(i.theta) ** 2) - 4.0 * r2 * sin2 * dvs
        out = out + (0.5 - 2.0 * r2 * sin2) * drho
        return out


@reflectivity.register("fatti")
class Fatti:
    """Fatti et al.'s (1994) form of the linearised reflectivity, in impedance contrasts.

    ``R = ½(1 + tan²θ) ΔI_P/Ī_P - 4r² sin²θ ΔI_S/Ī_S - (½tan²θ - 2r² sin²θ) Δrho/rho_m``,
    ``r = Vs_m/Vp_m``, ``θ`` the incidence angle (bruges' ``fatti``; PyLops' ``fatti``
    coefficients with the same contrasts).
    """

    def pp(
        self,
        vp1: ArrayLike,
        vs1: ArrayLike,
        rho1: ArrayLike,
        vp2: ArrayLike,
        vs2: ArrayLike,
        rho2: ArrayLike,
        angles_deg: ArrayLike,
    ) -> Array:
        """Return R_PP ``(*S, n_angle)``; see :class:`PPReflectivity`."""
        i = _interfaces(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg)
        vp, vs, _, _, _, drho = _means(i)
        ip1, ip2 = i.vp1 * i.rho1, i.vp2 * i.rho2
        is1, is2 = i.vs1 * i.rho1, i.vs2 * i.rho2
        dip = 2.0 * (ip2 - ip1) / (ip2 + ip1)
        dis = 2.0 * (is2 - is1) / (is2 + is1)
        sin2 = np.sin(i.theta) ** 2
        tan2 = np.tan(i.theta) ** 2
        r2 = (vs / vp) ** 2
        out: Array = 0.5 * (1.0 + tan2) * dip - 4.0 * r2 * sin2 * dis
        out = out - (0.5 * tan2 - 2.0 * r2 * sin2) * drho
        return out


@reflectivity.register("shuey")
class Shuey:
    """Shuey's (1985) three-term approximation, ``R₀ + G sin²θ + F (tan²θ - sin²θ)``.

    ``R₀ = ½(ΔVp/Vp_m + Δrho/rho_m)``, ``G = ½ΔVp/Vp_m - 2r²(Δrho/rho_m + 2ΔVs/Vs_m)``,
    ``F = ½ΔVp/Vp_m``, ``r = Vs_m/Vp_m``, ``θ`` the incidence angle (bruges' ``shuey``).
    ``_m`` is the mean of the two media.
    """

    def pp(
        self,
        vp1: ArrayLike,
        vs1: ArrayLike,
        rho1: ArrayLike,
        vp2: ArrayLike,
        vs2: ArrayLike,
        rho2: ArrayLike,
        angles_deg: ArrayLike,
    ) -> Array:
        """Return R_PP ``(*S, n_angle)``; see :class:`PPReflectivity`."""
        i = _interfaces(vp1, vs1, rho1, vp2, vs2, rho2, angles_deg)
        vp, vs, _, dvp, dvs, drho = _means(i)
        sin2 = np.sin(i.theta) ** 2
        tan2 = np.tan(i.theta) ** 2
        r0 = 0.5 * (dvp + drho)
        g = 0.5 * dvp - 2.0 * (vs / vp) ** 2 * (drho + 2.0 * dvs)
        out: Array = r0 + g * sin2 + 0.5 * dvp * (tan2 - sin2)
        return out
