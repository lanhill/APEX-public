"""SimulatorWrap's gravity and seafloor displacement of a 3dBox truth run.

Reads an OPM Flow run of ``examples/3dbox_production_avo``'s truth (whose deck writes DEN and
RPORV) with SimulatorWrap's own code, whole: ``flow_grav.get_grav_result`` (its stations,
``calc_mass``, ``calc_grav``) and ``flow_seafloor_disp.get_displacement_result`` (its
stations, ``get_pore_volume``, ``map_z_response`` with the Geertsma kernel in its
``integral`` mode), for a fixed baseline at the start and vintages at days 2000 and 4000,
the displacement by both its methods (``pressure``, ``porv``). The pressures go to it in
MPa (``press_conv = 0.1``, from the deck's bar), so that its ``compressibility`` is per MPa.

It writes ``tests/data/reference/geophysics/potential_3dbox.npz``: the stations, Δg (µGal)
and u_z (cm) per vintage, and the per-cell quantities they come from: each active cell's
mass change (the sum of the per-phase mass changes it saves) and volume decrease (computed
here by ``map_z_response``'s formula from the pore volumes and MPa pressures it keeps in
``dyn_var``, which it does not save), which ``tests/integration/test_geophysics_3dbox.py``
compares with APEX's forecast of its own run of the same truth. Whether the reference's
aquifer filter (``filter_rporv``) removed cells is recorded: it must not have, for the
comparison to hold cell for cell.

Run the truth with APEX into a folder ``RUN``, with the PET deck's ln PERMX cap
(``prepare.simulate_truth(prepare.HERE, RUN, permx_cap=6.0, deck="deck_pet.mako")``
of ``examples/3dbox_production_avo/prepare.py``), then this with the PET interpreter;
the displacement kernel's numerical integrals take a few minutes::

    env -u PYTHONPATH .venv-pet/bin/python tools/geophysics_reference/flow_potential_3dbox.py RUN

This script is outside the ``apex`` package and imports SimulatorWrap (``.venv-pet``).
"""

import argparse
import contextlib
import io
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import OUT, SIMULATORWRAP_COMMIT, VERSIONS, _check_environment

#: The deck's case name (APEX's ``opm_flow`` writes ``MODEL.DATA``).
CASE = "MODEL"
START = {"year": 2022, "month": 1, "day": 1}
#: The survey days after the start: the baseline and the vintages (the AVO case's).
BASELINE_DAY, VINTAGE_DAYS = 0, [2000, 4000]
#: The stations: SimulatorWrap's ``measurement_locations`` with these settings (m): 5 x 5
#: stations 742.5 m apart (``ceil(span / meas_spacing)`` per axis). Coarse, as the reference
#: integrates its kernel numerically for every station-cell pair.
STATIONS = {"padding": 600.0, "meas_spacing": 600.0, "water_depth": 300.0}
#: The displacement's settings: Poisson's ratio, the compaction coefficient (1/MPa).
POISSON, COMPRESSIBILITY_PER_MPA = 0.25, 1e-3


def _pem() -> SimpleNamespace:
    return SimpleNamespace(input={"phases": "OIL WAT GAS", "press_conv": 0.1})


def _model(cls: type, config_name: str, config: dict[str, object]) -> object:
    """Return a SimulatorWrap model of ``cls`` set up as its ``__init__`` would, without a run."""
    model = cls.__new__(cls)
    model.no_flow = False
    model.file = CASE
    model.startDate = START
    model.pem = _pem()
    model.dyn_var = []
    setattr(model, config_name, config)
    return model


def potential(run: Path) -> dict[str, np.ndarray]:
    """Return the stations, Δg, u_z and their per-cell quantities of the run."""
    from subsurface.multphaseflow.flow_rock import flow_grav, flow_seafloor_disp

    folder = str(run) + "/"
    out: dict[str, np.ndarray] = {}
    log = io.StringIO()
    with tempfile.TemporaryDirectory() as save, contextlib.redirect_stdout(log):
        grav = _model(
            flow_grav, "grav_config",
            {"baseline": BASELINE_DAY, "vintage": VINTAGE_DAYS, **STATIONS},
        )  # fmt: skip
        grav.get_grav_result(folder, save)
        for v, days in enumerate(VINTAGE_DAYS):
            saved = np.load(Path(save) / f"grav_vint{v}.npz", allow_pickle=True)
            out[f"out_grav_{days}"] = np.asarray(grav.grav_result[v], dtype=np.float64)
            out[f"out_grav_dm_{days}"] = sum(
                np.asarray(saved[f"{phase}_mass"], dtype=np.float64)
                for phase in ("OIL", "WAT", "GAS")
            )
        pos = saved["meas_location"].item()
        out["out_stations"] = np.column_stack((pos["x"], pos["y"], pos["z"]))
        for method in ("pressure", "porv"):
            disp = _model(
                flow_seafloor_disp, "disp_config",
                {"baseline": BASELINE_DAY, "vintage": VINTAGE_DAYS, "method": method,
                 "model": "geertsma", "poisson": POISSON,
                 "compressibility": COMPRESSIBILITY_PER_MPA, "z_base": 0.0,
                 "receiver_trunc_factor": 3.0, "geertsma_mode": "integral", **STATIONS},
            )  # fmt: skip
            disp.get_displacement_result(folder, save)
            base = disp.dyn_var[0]
            for v, days in enumerate(VINTAGE_DAYS):
                out[f"out_disp_{method}_{days}"] = np.asarray(disp.disp_result[v], np.float64)
                repeat = disp.dyn_var[v + 1]
                if method == "pressure":
                    dv = base["RPORV"] * (base["PRESSURE"] - repeat["PRESSURE"])
                    dv = dv * COMPRESSIBILITY_PER_MPA
                else:
                    dv = base["RPORV"] - repeat["RPORV"]
                out[f"out_disp_dv_{method}_{days}"] = np.asarray(dv, dtype=np.float64)
    out["meta_aquifer_filtered"] = np.array("Remove aquifer cells" in log.getvalue())
    out["in_poisson"] = np.array(POISSON)
    out["in_compressibility_per_mpa"] = np.array(COMPRESSIBILITY_PER_MPA)
    for name, value in STATIONS.items():
        out[f"in_{name}"] = np.array(value)
    return out


def main(argv: list[str] | None = None) -> int:
    """Read the run and write the reference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="the folder of an OPM Flow run of the truth")
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        _check_environment()
        arrays = potential(args.run.resolve())
    if bool(arrays["meta_aquifer_filtered"]):
        print("warning: the reference's aquifer filter removed cells", file=sys.stderr)
    prt = (args.run / f"{CASE}.PRT").read_text(errors="replace").splitlines()
    flow = next((line.strip() for line in prt if "Flow Version" in line), "unknown")
    path = OUT / "potential_3dbox.npz"
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
