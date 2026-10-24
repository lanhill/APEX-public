"""Generate the reference outputs of the geophysics kernels' tests.

Run with the PET interpreter, never the APEX environment: ``.venv-pet`` has
SimulatorWrap (``subsurface``) at ``d29dc57``, and PyLops, bruges and CoolProp, none of
which APEX depends on::

    uv pip install --python .venv-pet/bin/python "pylops==2.8.0" "bruges==0.5.4" \
        "CoolProp==8.0.0" "setuptools<81"   # bruges 0.5.4 imports pkg_resources
    env -u PYTHONPATH .venv-pet/bin/python tools/geophysics_reference/generate.py [<case> ...]

Each case writes ``tests/data/reference/geophysics/<case>.npz``: the seeded inputs
(``in_*``), the libraries' outputs (``out_*``) and provenance (``meta_*``). The libraries'
own functions are called, so the tests check APEX's kernels against them, not against a
copy. SimulatorWrap's modules are checked byte for byte against the reference commit
before any case runs.

This script is outside the ``apex`` package: it is the only place that imports
SimulatorWrap, PyLops or bruges.
"""

import argparse
import contextlib
import hashlib
import io
import os
import sys
import warnings
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "geophysics"

#: SimulatorWrap (``subsurface``) at d29dc57: sha256 of the modules the cases call.
SIMULATORWRAP_COMMIT = "d29dc57"
SIMULATORWRAP_SHA256 = {
    "multphaseflow/flow_rock.py": (
        "d42e3c33126c38b08dec78debc739fa95d9778db7e6523f863f80980284aed40"
    ),
    "rockphysics/standardrp.py": "5362fd972003919ef2bd3811643942f639b1715ca231e9aa3b54fadc8af335a7",
    "rockphysics/softsandrp.py": "a38aba82563d78669fc8679bc81c46fbad4b5f2d835d49bcd6267d7052b27c9f",
}
VERSIONS = {"pylops": "2.8.0", "bruges": "0.5.4"}

type Arrays = dict[str, np.ndarray]
CASES: dict[str, tuple[int, Callable[[np.random.Generator], Arrays]]] = {}


def case(name: str, seed: int) -> Callable[[Callable[[np.random.Generator], Arrays]], object]:
    """Register a case: a function of a seeded generator returning ``in_``/``out_`` arrays."""

    def decorator(function: Callable[[np.random.Generator], Arrays]) -> object:
        CASES[name] = (seed, function)
        return function

    return decorator


def _check_environment() -> None:
    if os.environ.get("PYTHONPATH"):
        sys.exit("PYTHONPATH is set; run with `env -u PYTHONPATH`")
    for package, version in VERSIONS.items():
        found = metadata.version(package)
        if found != version:
            sys.exit(f"{package} {version} is the reference version; this is {found}")
    import subsurface

    root = Path(subsurface.__file__).resolve().parent
    for relative, digest in SIMULATORWRAP_SHA256.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != digest:
            sys.exit(f"subsurface/{relative} is not SimulatorWrap {SIMULATORWRAP_COMMIT}")


def _interfaces(rng: np.random.Generator, n: int, angles: np.ndarray) -> Arrays:
    """Random sedimentary interfaces, none evanescent at the largest angle.

    Vp 1500-4500 m/s, Vs/Vp 0.3-0.65, density 1800-2700 kg/m³; a third of them with
    contrasts of a few per cent, where the approximations approach the exact solution.
    """
    out: Arrays = {}
    rows: list[np.ndarray] = []
    sin_max = np.sin(np.radians(angles.max()))
    while len(rows) < n:
        vp1, vp2 = rng.uniform(1500, 4500, 2)
        if len(rows) % 3 == 0:
            vp2 = vp1 * rng.uniform(0.97, 1.03)
        r1, r2 = rng.uniform(0.3, 0.65, 2)
        d1, d2 = rng.uniform(1800, 2700, 2)
        vs1, vs2 = r1 * vp1, r2 * vp2
        if max(vp2, vs2) * sin_max / vp1 < 0.95:
            rows.append(np.array([vp1, vs1, d1, vp2, vs2, d2]))
    table = np.array(rows)
    for k, name in enumerate(("vp1", "vs1", "rho1", "vp2", "vs2", "rho2")):
        out[f"in_{name}"] = table[:, k].copy()
    out["in_angles"] = angles
    return out


@case("reflectivity", seed=20261013)
def reflectivity(rng: np.random.Generator) -> Arrays:
    """PyLops' and bruges' PP reflectivities on 240 interfaces at 0-40 degrees.

    - ``out_pylops_approx_zoeppritz_pp`` ``(n_angle, n)``: as SimulatorWrap calls it, one
      angle at a time on arrays (``flow_rock.py`` L1896);
    - ``out_pylops_zoeppritz_pp`` ``(n_angle, n)``: the 4 x 4 scattering matrix, one
      interface at a time (it takes scalars only);
    - ``out_pylops_{akirichards,fatti}_g{1,2,3}`` ``(n_angle, n)``: PyLops' coefficients
      with Vs/Vp the ratio of the means;
    - ``out_bruges_*`` ``(n_angle, n)``: ``zoeppritz_rpp`` (explicit), ``zoeppritz``
      (scattering matrix, one interface and angle at a time), ``akirichards``, ``fatti``,
      ``shuey`` (real parts; their imaginary parts are checked to be 0 here).
    """
    import bruges.reflection as br
    from pylops.avo import avo

    angles = np.array([0.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    out = _interfaces(rng, 240, angles)
    vp1, vs1, rho1 = out["in_vp1"], out["in_vs1"], out["in_rho1"]
    vp2, vs2, rho2 = out["in_vp2"], out["in_vs2"], out["in_rho2"]
    n = vp1.size
    out["out_pylops_approx_zoeppritz_pp"] = np.array(
        [
            avo.approx_zoeppritz_pp(vp1, vs1, rho1, vp2, vs2, rho2, np.atleast_1d(a))[0]
            for a in angles
        ]
    )
    out["out_pylops_zoeppritz_pp"] = np.array(
        [
            [
                avo.zoeppritz_pp(vp1[j], vs1[j], rho1[j], vp2[j], vs2[j], rho2[j], float(a))
                for j in range(n)
            ]
            for a in angles
        ]
    )
    ratio = ((vs1 + vs2) / 2) / ((vp1 + vp2) / 2)
    for name in ("akirichards", "fatti"):
        g = getattr(avo, name)(angles, ratio)
        for k in range(3):
            out[f"out_pylops_{name}_g{k + 1}"] = np.asarray(g[k]) * np.ones((angles.size, n))
    for name in ("zoeppritz_rpp", "zoeppritz", "akirichards", "fatti", "shuey"):
        if name == "zoeppritz":
            # bruges' scattering matrix fails to broadcast arrays (an in-place ``theta1 *=``),
            # so it is called one interface and angle at a time.
            value = np.array(
                [
                    [
                        br.zoeppritz(
                            vp1[j], vs1[j], rho1[j], vp2[j], vs2[j], rho2[j], np.array([a])
                        )
                        for j in range(n)
                    ]
                    for a in angles
                ]
            )
        else:
            value = np.asarray(getattr(br, name)(vp1, vs1, rho1, vp2, vs2, rho2, angles))
        value = value.reshape(angles.size, n)
        if np.max(np.abs(np.imag(value))) > 1e-12 * np.max(np.abs(np.real(value))):
            raise RuntimeError(f"bruges {name} has an imaginary part on subcritical interfaces")
        out[f"out_bruges_{name}"] = np.real(value).astype(np.float64)
    return out


def _flow_cells(rng: np.random.Generator, n: int, phases: tuple[str, ...]) -> Arrays:
    """Return random cells for the petro-elastic cases.

    Porosity 0.05-0.35, pressure 10-40 MPa, overburden 40-60 MPa (four below the
    pressure, where the reference clamps the effective pressure), and random saturations
    of ``phases`` (Dirichlet, so they sum to 1).
    """
    porosity = rng.uniform(0.05, 0.35, n)
    pressure = rng.uniform(10.0, 40.0, n)
    overburden = rng.uniform(40.0, 60.0, n)
    overburden[:4] = pressure[:4] - rng.uniform(0.5, 2.0, 4)  # negative effective pressure
    saturations = rng.dirichlet(np.ones(len(phases)), n).T
    out: Arrays = {
        "in_porosity": porosity,
        "in_pressure_mpa": pressure,
        "in_overburden_mpa": overburden,
        "in_phases": np.array(phases),
        "in_saturations": saturations,
    }
    return out


def _elastic_out(rock: object) -> Arrays:
    return {
        f"out_{name}": np.asarray(getattr(rock, f"get{getter}")(), dtype=np.float64)
        for name, getter in (
            ("vp", "BulkVel"), ("vs", "ShearVel"), ("density", "Dens"),
            ("bulk_modulus_mpa", "BulkMod"), ("shear_modulus_mpa", "ShearMod"),
        )
    }  # fmt: skip


@case("pem_standard", seed=20261014)
def pem_standard(rng: np.random.Generator) -> Arrays:
    """``standardrp.elasticproperties.calc_props`` on 200 three-phase cells.

    Outputs ``(n_cell,)``: Vp, Vs (m/s), density (kg/m³), saturated bulk and shear moduli
    (MPa). The overburden is set on the object, as ``flow_rock`` does from its file.
    """
    from subsurface.rockphysics import standardrp

    out = _flow_cells(rng, 200, ("OIL", "WAT", "GAS"))
    # Above φc = 0.4 the reference extrapolates, up to the clay limit (0.443).
    out["in_porosity"][-4:] = [0.40, 0.41, 0.42, 0.44]
    rock = standardrp.elasticproperties({})
    rock.overburden = out["in_overburden_mpa"]
    rock.p_init = None
    rock.calc_props(
        ["OIL", "WAT", "GAS"], list(out["in_saturations"]), out["in_pressure_mpa"],
        out["in_porosity"],
    )  # fmt: skip
    out.update(_elastic_out(rock))
    return out


@case("pem_soft_sand", seed=20261015)
def pem_soft_sand(rng: np.random.Generator) -> Arrays:
    """``softsandrp.elasticproperties`` on 200 brine-CO₂ cells, with phase densities given.

    - ``out_vp`` ... ``out_shear_modulus_mpa``: ``calc_props`` (the whole model);
    - ``out_fluid_density``, ``out_fluid_bulk_mpa``: the reference's pore fluid per cell
      (``_fluid_dens``, ``_fluidprops_Brie``), so that the dry rock and Gassmann can be
      compared alone (APEX's fluids differ: pet_parity.md G3-G6);
    - ``out_hm_bulk_mpa``, ``out_hm_shear_mpa``, ``out_dry_bulk_mpa``,
      ``out_dry_shear_mpa``: ``_hertzmindlin_Mavko`` and ``_dryrockmoduli_Smeaheia`` at
      the clamped effective pressure;
    - ``out_co2_bulk_mpa``, ``out_methane_bulk_mpa``: the gas branch of
      ``_phaseprops_Smeaheia`` (``CO2=True``; and ``CO2=None``, its methane branch);
    - ``out_water_bulk_mpa``: its pure-water branch (``CO2=False``) at the cell's fluid
      density, and ``out_brine_bulk_mpa`` its brine branch (``CO2=True``);
    - ``out_oil_bulk_pa``: its oil branch at the cell's fluid density (in Pa: G5).
    """
    from subsurface.rockphysics import softsandrp

    out = _flow_cells(rng, 200, ("WAT", "GAS"))
    out["in_porosity"] = rng.uniform(0.05, 0.355, out["in_porosity"].size)
    # At and above φc = 0.36, where the reference's cap φ/φc <= 0.9 holds the rock.
    out["in_porosity"][-6:] = [0.36, 0.38, 0.40, 0.45, 0.5, 0.6]
    n = out["in_porosity"].size
    densities = np.stack([rng.uniform(1000.0, 1100.0, n), rng.uniform(150.0, 750.0, n)])
    out["in_densities"] = densities
    rock = softsandrp.elasticproperties({})
    rock.overburden = out["in_overburden_mpa"]
    rock.p_init = None
    phases = ["WAT", "GAS"]
    sats, p, phi = out["in_saturations"], out["in_pressure_mpa"], out["in_porosity"]
    with contextlib.redirect_stdout(io.StringIO()):  # its clamp prints per cell
        rock.calc_props(phases, list(sats), p, phi, dens=list(densities))
    out.update(_elastic_out(rock))
    k_s, g_s = 37000.0, 44000.0  # the reference's quartz (MPa)
    keys = (
        "fluid_density", "fluid_bulk_mpa", "hm_bulk_mpa", "hm_shear_mpa", "dry_bulk_mpa",
        "dry_shear_mpa", "co2_bulk_mpa", "methane_bulk_mpa", "water_bulk_mpa",
        "brine_bulk_mpa", "effective_mpa", "oil_bulk_pa",
    )  # fmt: skip
    rows: dict[str, list[float]] = {k: [] for k in keys}
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(n):
            densf = rock._fluid_dens(sats[:, i], densities[:, i])
            peff = rock._effective_pressure(out["in_overburden_mpa"][i], p[i])
            hm_k, hm_g = rock._hertzmindlin_Mavko(peff, k_s, g_s, 9, 0.36)
            dry_k, dry_g = rock._dryrockmoduli_Smeaheia(9, 0.36, phi[i], peff, k_s, g_s)
            values = (
                densf, rock._fluidprops_Brie(phases, sats[:, i], p[i], densf), hm_k, hm_g,
                dry_k, dry_g, rock._phaseprops_Smeaheia("gas", p[i], densf, CO2=True),
                rock._phaseprops_Smeaheia("gas", p[i], densf, CO2=None),
                rock._phaseprops_Smeaheia("wat", p[i], densf, CO2=False),
                rock._phaseprops_Smeaheia("wat", p[i], densf, CO2=True), peff,
                rock._phaseprops_Smeaheia("oil", p[i], densf),
            )  # fmt: skip
            for k, v in zip(keys, values, strict=True):
                rows[k].append(float(v))
    out.update({f"out_{k}": np.array(v) for k, v in rows.items()})
    return out


@case("rock_physics_bruges", seed=20261016)
def rock_physics_bruges(rng: np.random.Generator) -> Arrays:
    """Return bruges' rock-physics building blocks on 100 random inputs (an independent library).

    Moduli in GPa as bruges takes them, effective stress in MPa, temperature in °C,
    pressure in Pa for the fluids:

    - ``hashin_shtrikman`` (``bounds``), bulk and shear, lower and upper, of two minerals;
    - ``hertz_mindlin`` and ``soft_sand`` (``rockphysicsmodels``, ``f = 1``);
    - ``smith_gassmann`` (``fluidsub``); ``wood`` (``fluids``);
    - ``v_water`` and ``v_brine`` (``fluids``; Batzle & Wang eqs. 28-29).
    """
    from bruges.rockphysics import bounds, fluids, fluidsub, rockphysicsmodels

    n = 100
    out: Arrays = {
        "in_k1": rng.uniform(10, 25, n), "in_g1": rng.uniform(2, 8, n),
        "in_k2": rng.uniform(30, 75, n), "in_g2": rng.uniform(30, 50, n),
        "in_f2": rng.uniform(0, 1, n), "in_sigma_mpa": rng.uniform(1, 40, n),
        "in_porosity": rng.uniform(0.02, 0.35, n),
        "in_critical_porosity": rng.uniform(0.36, 0.42, n),
        "in_coordination": rng.uniform(6, 12, n), "in_fluid_bulk": rng.uniform(0.02, 3.0, n),
        "in_temperature_c": rng.uniform(10, 100, n),
        "in_pressure_pa": rng.uniform(0.1e6, 60e6, n),
        "in_salinity": rng.uniform(0, 0.2, n), "in_saturation": rng.uniform(0, 1, n),
    }  # fmt: skip
    keys = ("hs_bulk_lower", "hs_bulk_upper", "hs_shear_lower", "hs_shear_upper",
            "hm_bulk", "hm_shear", "soft_bulk", "soft_shear", "gassmann")  # fmt: skip
    rows: dict[str, list[float]] = {k: [] for k in keys}
    for i in range(n):
        f = [1 - out["in_f2"][i], out["in_f2"][i]]
        k, g = [out["in_k1"][i], out["in_k2"][i]], [out["in_g1"][i], out["in_g2"][i]]
        hs_k = bounds.hashin_shtrikman(f, k, g, modulus="bulk")
        hs_g = bounds.hashin_shtrikman(f, k, g, modulus="shear")
        args = (out["in_k2"][i], out["in_g2"][i])
        hm = rockphysicsmodels.hertz_mindlin(
            *args, out["in_sigma_mpa"][i], out["in_critical_porosity"][i],
            out["in_coordination"][i], 1,
        )  # fmt: skip
        soft = rockphysicsmodels.soft_sand(
            *args, out["in_porosity"][i], out["in_sigma_mpa"][i],
            out["in_critical_porosity"][i], out["in_coordination"][i], 1,
        )  # fmt: skip
        gassmann = fluidsub.smith_gassmann(
            soft[0], out["in_k2"][i], out["in_fluid_bulk"][i], out["in_porosity"][i]
        )
        values = (*hs_k, *hs_g, *hm, *soft, gassmann)
        for key, v in zip(keys, values, strict=True):
            rows[key].append(float(v))
    out.update({f"out_{k}": np.array(v) for k, v in rows.items()})
    out["out_wood"] = np.asarray(
        fluids.wood(out["in_k1"], out["in_fluid_bulk"], out["in_saturation"]), dtype=np.float64
    )
    t, p = out["in_temperature_c"], out["in_pressure_pa"]
    out["out_v_water"] = np.asarray(fluids.v_water(t, p), dtype=np.float64)
    out["out_v_brine"] = np.asarray(fluids.v_brine(t, p, out["in_salinity"]), dtype=np.float64)
    return out


@case("avo_traces", seed=20261017)
def avo_traces(rng: np.random.Generator) -> Arrays:
    """``flow_avo._calc_avo_props_active_cells`` on a 7 x 6 x 5 grid, 3dBox's settings.

    Settings of the 3dBox AVO case's PET set-up (``3D_IES.pipt``): 45 Hz Ricker of
    0.05 s, shale 2450/849 m/s and 2300 kg/m³, 1.9224-2.2150 s at 1 ms,
    ``approx_zoeppritz_pp``, 0.5 ms internally. Every cell is active (the reference's
    inactive cells are NaN). The cell thicknesses are random, so no output time ties
    between two internal samples.

    - ``out_traces_<angle>`` ``(ny, nx, n_time)``: one call per angle (single-angle list);
    - ``out_traces_both`` ``(ny, nx, n_time, 2)``: one call with both angles, which
      returns the last angle's traces twice (pet_parity.md G7).
    Inputs: ``in_top`` (ny, nx) top depth, ``in_dz``, ``in_vp``, ``in_vs``, ``in_rho``
    (nz, ny, nx), as the reference takes them.
    """
    from pylops.avo.avo import approx_zoeppritz_pp
    from subsurface.multphaseflow.flow_rock import flow_avo

    nz, ny, nx = 5, 6, 7
    top = rng.uniform(2370.0, 2420.0, (ny, nx))
    dz = rng.uniform(4.0, 25.0, (nz, ny, nx))
    vp = rng.uniform(2300.0, 3600.0, (nz, ny, nx))
    vs = vp * rng.uniform(0.45, 0.6, (nz, ny, nx))
    rho = rng.uniform(2050.0, 2450.0, (nz, ny, nx))
    zcorn = np.zeros((nz, 2, ny, 2, nx, 2))
    zcorn[0, 0, :, 0, :, 0] = top
    out: Arrays = {"in_top": top, "in_dz": dz, "in_vp": vp, "in_vs": vs, "in_rho": rho}
    angles = [10.0, 20.0]

    def run(angle_list: list[float]) -> np.ndarray:
        model = flow_avo.__new__(flow_avo)
        model.avo_config = {
            "angle": angle_list, "frequency": 45.0, "wave_len": 0.05, "vp_shale": 2450.0,
            "vs_shale": 849.0, "den_shale": 2300.0, "t_min": 1.9224, "t_max": 2.2150,
            "t_sampling": 0.001,
        }  # fmt: skip
        model.pp_func = approx_zoeppritz_pp
        model.NX, model.NY, model.NZ = nx, ny, nz
        grid = {"ACTNUM": np.ones((nz, ny, nx), dtype=bool)}
        data = model._calc_avo_props_active_cells(grid, vp, vs, rho, dz, zcorn)[0]
        return np.asarray(data, dtype=np.float64)

    for a in angles:
        out[f"out_traces_{int(a)}"] = run([a])
    out["out_traces_both"] = run(angles)
    out["in_angles"] = np.array(angles)
    return out


def _box_grid(
    rng: np.random.Generator, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small corner-point grid in ``misc.ecl``'s layout, slanted, one cell inactive.

    COORD ``(nj+1, ni+1, 2, 3)`` (each pillar's top then base point), ZCORN
    ``(nk, 2, nj, 2, ni, 2)``, ACTNUM ``(nk, nj, ni)``.
    """
    nk, nj, ni = shape
    x = np.arange(ni + 1) * 120.0
    y = np.arange(nj + 1) * 90.0
    gx, gy = np.meshgrid(x, y)  # (nj+1, ni+1)
    coord = np.zeros((nj + 1, ni + 1, 2, 3))
    coord[..., 0, 0], coord[..., 0, 1], coord[..., 0, 2] = gx, gy, 1800.0
    # Slanted pillars: the base is shifted, so cell centres follow the pillars.
    coord[..., 1, 0], coord[..., 1, 1], coord[..., 1, 2] = gx + 30.0, gy - 20.0, 2300.0
    tops = 2000.0 + rng.uniform(0.0, 15.0, (nj + 1, ni + 1))
    dz = rng.uniform(8.0, 20.0, nk)
    zcorn = np.zeros((nk, 2, nj, 2, ni, 2))
    for k in range(nk):
        top = tops + dz[:k].sum()
        for a in (0, 1):
            for b in (0, 1):
                corner = top[a : a + nj, b : b + ni]
                zcorn[k, 0, :, a, :, b] = corner
                zcorn[k, 1, :, a, :, b] = corner + dz[k]
    actnum = np.ones((nk, nj, ni), dtype=bool)
    actnum[1, 2, 3] = False
    return coord, zcorn, actnum


@case("stations", seed=20261020)
def stations(rng: np.random.Generator) -> Arrays:
    """``measurement_locations`` over a small grid: plain, refined near a well, and a seabed.

    ``out_<variant>`` ``(n_station, 3)``: x, y, z (z positive down), in the reference's
    order. Inputs: the grid (``in_coord``, ``in_zcorn``, ``in_actnum``), ``in_centres``
    (the reference's active cell centres, ``(3, n_active)``), the seabed table.
    """
    import pandas as pd
    from subsurface.multphaseflow.flow_rock import flow_grav

    coord, zcorn, actnum = _box_grid(rng, (3, 5, 6))
    grid = {"COORD": coord, "ZCORN": zcorn, "ACTNUM": actnum}
    model = flow_grav.__new__(flow_grav)
    centres = np.array(model.find_cell_centre(grid))
    seabed = pd.DataFrame(
        {"x": rng.uniform(-900, 1600, 40), "y": rng.uniform(-900, 1400, 40),
         "z": -rng.uniform(250, 350, 40), "column": 0, "row": 0}
    )  # fmt: skip
    out: Arrays = {"in_coord": coord, "in_zcorn": zcorn, "in_actnum": actnum,
                   "in_centres": centres,
                   "in_seabed": seabed[["x", "y", "z"]].to_numpy()}  # fmt: skip

    def stack(pos: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack((pos["x"], pos["y"], pos["z"])).astype(np.float64)

    out["out_plain"] = stack(model.measurement_locations(grid, 300.0, pad=700.0, dxy=250.0))
    out["out_refined"] = stack(
        model.measurement_locations(grid, 300.0, pad=700.0, dxy=250.0,
                                    well_coord=[330.0, 210.0], r0=400.0)
    )  # fmt: skip
    out["out_seabed"] = stack(model.measurement_locations(grid, seabed, pad=500.0, dxy=300.0))
    return out


@case("gravity", seed=20261021)
def gravity(rng: np.random.Generator) -> Arrays:
    """``flow_grav.calc_grav``: Δg (µGal) at stations of the cells' mass changes.

    A 6 x 5 x 3 grid (one cell inactive), three phases; ``in_dm`` ``(n_active,)`` the
    total mass change (kg) per active cell in the reference's active order, ``in_stations``
    ``(n, 3)``; ``out_dg`` ``(n,)``; ``out_centres`` ``(3, n_active)``.
    """
    from subsurface.multphaseflow.flow_rock import flow_grav

    coord, zcorn, actnum = _box_grid(rng, (3, 5, 6))
    grid = {"COORD": coord, "ZCORN": zcorn, "ACTNUM": actnum}
    n = int(actnum.sum())
    masses = {p: rng.uniform(-4e6, 4e6, n) for p in ("OIL", "WAT", "GAS")}
    base = {f"{p}_mass": np.zeros(n) for p in masses}
    repeat = {f"{p}_mass": m for p, m in masses.items()}
    stations = np.column_stack(
        (rng.uniform(-500, 1200, 25), rng.uniform(-400, 900, 25), rng.uniform(0, 300, 25))
    )
    model = flow_grav.__new__(flow_grav)
    model.pem = type("Pem", (), {"input": {"phases": "OIL WAT GAS"}})()
    pos = {"x": stations[:, 0], "y": stations[:, 1], "z": stations[:, 2]}
    dg = model.calc_grav(grid, base, repeat, pos)
    return {
        "in_coord": coord, "in_zcorn": zcorn, "in_actnum": actnum, "in_stations": stations,
        "in_dm": sum(masses.values()), "out_dg": np.asarray(dg, dtype=np.float64),
        "out_centres": np.array(model.find_cell_centre(grid)),
    }  # fmt: skip


@case("seafloor_geertsma", seed=20261022)
def seafloor_geertsma(rng: np.random.Generator) -> Arrays:
    """``compute_deformation_transfer`` (Geertsma, ``integral``): u_z per unit ΔV.

    Per cell ``j`` (depth ``in_cell_depth[j]``, m), the stations ``in_station_depth``,
    ``in_rho[j]`` (m); ``out_u[j]`` (m per m³), the reference's ``THH`` with ΔV = 1. The
    stations lie within 3 times the depth difference (its truncation) and close enough
    that its λ_max = 15/max rho (km⁻¹) leaves a negligible tail; ``truncated`` adds stations
    beyond the truncation (0 there), and ``wide`` stations up to it, where λ_max cuts the
    integral short (G12).
    """
    from subsurface.multphaseflow.flow_rock import flow_seafloor_disp

    model = flow_seafloor_disp.__new__(flow_seafloor_disp)
    nu, compressibility = 0.25, 1e-4
    e_mod = ((1 + nu) * (1 - 2 * nu)) / ((1 - nu) * compressibility)
    out: Arrays = {"in_poisson": np.array(nu)}
    for name, max_rho in (("close", 0.12), ("truncated", None), ("wide", 3.0)):
        depths = rng.uniform(1500.0, 2600.0, 6)
        z = rng.uniform(0.0, 350.0, 12)
        rhos, us = [], []
        for c in depths:
            if max_rho is None:
                rho = np.abs(c - z) * rng.uniform(0.0, 6.0, z.size)
            else:
                rho = np.abs(c - z) * rng.uniform(0.0, max_rho, z.size)
            thh, _ = model.compute_deformation_transfer(
                z, c, 0.0, rho, nu, e_mod, 1.0, ["Geertsma_vertical"]
            )
            rhos.append(rho)
            us.append(thh)
        out[f"in_{name}_cell_depth"] = depths
        out[f"in_{name}_station_depth"] = z
        out[f"in_{name}_rho"] = np.array(rhos)
        out[f"out_{name}_u"] = np.array(us)
    return out


def write(name: str) -> Path:
    """Run case ``name`` and write its ``.npz``."""
    seed, function = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # bruges 0.5.4's docstrings
        arrays = function(np.random.default_rng(seed))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.npz"
    np.savez_compressed(
        path,
        **arrays,
        meta_seed=np.array(seed),
        meta_simulatorwrap_commit=np.array(SIMULATORWRAP_COMMIT),
        meta_versions=np.array([f"{k} {v}" for k, v in VERSIONS.items()]),
        meta_numpy_version=np.array(np.__version__),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Generate the named cases, or all of them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help=f"cases to generate (all: {sorted(CASES)})")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        parser.error(f"unknown cases {unknown}; known: {sorted(CASES)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        _check_environment()
    for name in args.cases or sorted(CASES):
        print(f"wrote {write(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
