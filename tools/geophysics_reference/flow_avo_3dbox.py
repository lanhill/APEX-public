"""SimulatorWrap's AVO traces of a 3dBox truth run, for the end-to-end check.

Reads an OPM Flow run of ``examples/3dbox_production_avo``'s truth with SimulatorWrap's own
code, the calls ``flow_avo.get_avo_result`` makes at each vintage: ``calc_pem`` (its
restart reading, ``standardrp``), ``calc_velocities`` and ``_calc_avo_props_active_cells``
with the 3dBox settings, one angle per call (``pet_parity.md``, G7). Deliberately, it
stops at the traces: ``get_avo_result``'s ``step_x``/``step_y`` thinning and its masks
(active, sample, ROI) are not applied, NTG is not read (``standardrp`` does not use it),
and the pressures are in MPa where the case's PET set-up passes bar (below). It writes a
subset of the traces to ``tests/data/reference/geophysics/flow_avo_3dbox.npz``, which
``tests/integration/test_geophysics_3dbox.py`` compares with APEX's ``sampling =
"nearest"`` traces of its own run of the same truth.

Two choices make the comparison one of reading and wiring, not of units: the pressures go
to ``standardrp`` in MPa (``press_conv = 0.1``, from the deck's bar), and the overburden is
the PET set-up's value, 320 bar (32 MPa), in both.

Run the truth with APEX into a folder ``RUN``, with the PET deck's ln PERMX cap
(``prepare.simulate_truth(prepare.HERE, RUN, permx_cap=6.0, deck="deck_pet.mako")``
of ``examples/3dbox_production_avo/prepare.py``), then this with the PET interpreter
::

    env -u PYTHONPATH .venv-pet/bin/python tools/geophysics_reference/flow_avo_3dbox.py RUN

This script is outside the ``apex`` package and imports SimulatorWrap (``.venv-pet``).
"""

import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import OUT, SIMULATORWRAP_COMMIT, VERSIONS, _check_environment

#: The deck's case name (APEX's ``opm_flow`` writes ``MODEL.DATA``).
CASE = "MODEL"
START = dt.datetime(2022, 1, 1)
#: The vintages, days after the start (the PET set-up's ``VINTAGE``), and the angles.
VINTAGE_DAYS = (2000, 4000)
ANGLES = (10.0, 20.0)
#: The PET set-up's overburden pressure (``overburden.npz``: 320 per cell), in MPa.
OVERBURDEN_MPA = 32.0
#: Every TRACE_STEP-th column along I and J is kept (60 x 60 columns: 9 x 9 kept).
TRACE_STEP = 7


def traces(run: Path) -> dict[str, np.ndarray]:
    """Return the kept traces ``(n_kept, n_time)`` per vintage and angle, and their columns."""
    from misc import ecl
    from pylops.avo.avo import approx_zoeppritz_pp
    from subsurface.multphaseflow.flow_rock import flow_avo
    from subsurface.rockphysics import standardrp

    case = ecl.EclipseCase(str(run / f"{CASE}.DATA"))
    grid = case.grid()
    # get_avo_result builds an EclipseInit from the .DATA path (which it cannot open);
    # the case's own INIT gives the same dimensions and DZ.
    nz, ny, nx = case.init.nk, case.init.nj, case.init.ni
    rock = standardrp.elasticproperties({})
    rock.overburden = np.full(int(np.sum(grid["ACTNUM"])), OVERBURDEN_MPA)
    if hasattr(rock, "p_init"):  # calc_pem then takes the vintage's pressure (unused anyway)
        del rock.p_init
    rock.input = {"ntg": "no", "press_conv": 0.1}

    model = flow_avo.__new__(flow_avo)
    model.no_flow = False
    model.ecl_case = case
    model.pem = rock
    model.pem_input = {"phases": ["OIL", "WAT", "GAS"]}
    model.dyn_var = []
    model.ensemble_member = None
    model.pp_func = approx_zoeppritz_pp
    model.NX, model.NY, model.NZ = nx, ny, nz
    dz = case.cell_data("DZ")
    out: dict[str, np.ndarray] = {}
    for days in VINTAGE_DAYS:
        time = START + dt.timedelta(days=days)
        model.calc_pem(time, 1)
        vp, vs, rho = model.calc_velocities(str(run), None, grid, 1, [nz, ny, nx])
        for angle in ANGLES:
            model.avo_config = {
                "angle": [angle], "frequency": 45.0, "wave_len": 0.05, "vp_shale": 2450.0,
                "vs_shale": 849.0, "den_shale": 2300.0, "t_min": 1.9224, "t_max": 2.2150,
                "t_sampling": 0.001,
            }  # fmt: skip
            data = np.asarray(
                model._calc_avo_props_active_cells(grid, vp, vs, rho, dz, grid["ZCORN"])[0],
                dtype=np.float64,
            )
            # (ny, nx, n_time[, 1]): keep every TRACE_STEP-th column, J then I.
            kept = data.reshape(ny, nx, -1)[::TRACE_STEP, ::TRACE_STEP, :]
            out[f"out_{int(angle)}_{days}"] = kept.reshape(-1, kept.shape[-1])
    j, i = np.meshgrid(np.arange(0, ny, TRACE_STEP), np.arange(0, nx, TRACE_STEP), indexing="ij")
    out["out_columns"] = np.column_stack((i.ravel(), j.ravel()))  # (I, J), 0-based
    return out


def main(argv: list[str] | None = None) -> int:
    """Read the run and write the reference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="the folder of an OPM Flow run of the truth")
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        _check_environment()
        arrays = traces(args.run.resolve())
    prt = (args.run / f"{CASE}.PRT").read_text(errors="replace").splitlines()
    flow = next((line.strip() for line in prt if "Flow Version" in line), "unknown")
    path = OUT / "flow_avo_3dbox.npz"
    np.savez_compressed(
        path,
        **arrays,
        meta_simulatorwrap_commit=np.array(SIMULATORWRAP_COMMIT),
        meta_versions=np.array([f"{k} {v}" for k, v in VERSIONS.items()]),
        meta_flow_version=np.array(flow),
        meta_numpy_version=np.array(np.__version__),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
