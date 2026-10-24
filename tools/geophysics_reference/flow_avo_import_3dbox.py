"""SimulatorWrap's own AVO observations of a 3dBox truth run, for ``apex pet-import``.

Runs ``flow_avo.get_avo_result`` (SimulatorWrap ``d29dc57``) whole, as PET runs it, on an
OPM Flow run of ``examples/3dbox_production_avo``'s truth: the PEM at a fixed baseline and the
vintages, the traces, its mask (active columns, ``step_x``/``step_y``, the region of
interest) and the differences from the baseline, flattened as PET assimilates them. Each
variant's vectors are what a PET case would hold as observations;
``tests/integration/test_geophysics_3dbox.py`` imports them with ``apex pet-import`` and
checks APEX's forecast of the same run reproduces them, value by value.

The variants (one angle each: with two, SimulatorWrap gives both the last one's traces,
``pet_parity.md`` G7), each thinned by ``step_x``/``step_y``:

- ``fallback``: the default ellipse; 3dBox's completions are in an included file, which
  SimulatorWrap does not read, so it surrounds every active column;
- ``ellipse``: the default ellipse with a 300 m margin, around the completions written into
  the deck (a copy of the run whose ``MODEL.DATA`` holds the schedule inline);
- ``rectangle``: ``x_range``/``y_range`` with no margin.

The folders are passed with a trailing separator, as PET passes its ``En_<i>/`` (without
it, ``get_avo_result`` reads the grid through ``EclipseInit``, which lacks ``init``), and a
``baseline`` is set (without one it fails on ``base_time``: G10). Pressures go to the PEM in MPa
(``press_conv = 0.1``, from the deck's bar) with PET's 320 bar overburden as 32 MPa.

Run the truth with APEX into a folder ``RUN``, with the PET deck's ln PERMX cap
(``prepare.simulate_truth(prepare.HERE, RUN, permx_cap=6.0, deck="deck_pet.mako")``
of ``examples/3dbox_production_avo/prepare.py``), then this with the PET interpreter
::

    env -u PYTHONPATH .venv-pet/bin/python tools/geophysics_reference/flow_avo_import_3dbox.py RUN

This script is outside the ``apex`` package and imports SimulatorWrap (``.venv-pet``).
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import OUT, SIMULATORWRAP_COMMIT, VERSIONS, _check_environment

#: The deck's case name (APEX's ``opm_flow`` writes ``MODEL.DATA``).
CASE = "MODEL"
#: The case's PET set-up: start 1 Jan 2022, baseline at the start, vintages (days), one angle.
START = {"day": 1, "month": 1, "year": 2022}
BASELINE = 0.0
VINTAGES = [2000.0, 4000.0]
ANGLE = 20.0
#: The PET set-up's overburden pressure (``overburden.npz``: 320 per cell), in MPa.
OVERBURDEN_MPA = 32.0
#: The 3dBox AVO settings (``3D_IES.pipt``), but for the time axis: its first 0.1 s, which
#: holds the reservoir (150 m below 1.9224 s), keeps the reference small.
AVO = {
    "angle": [ANGLE], "frequency": 45.0, "wave_len": 0.05, "vp_shale": 2450.0,
    "vs_shale": 849.0, "den_shale": 2300.0, "t_min": 1.9224, "t_max": 2.0224,
    "t_sampling": 0.001, "pp_func": "approx_zoeppritz_pp",
}  # fmt: skip
#: Each variant's own avo keys, and whether its deck holds the schedule inline; thinned
#: along both axes, unequally, which also keeps the reference small.
VARIANTS: dict[str, tuple[dict[str, object], bool]] = {
    "fallback": ({"step_x": 6, "step_y": 5}, False),
    "ellipse": ({"roi_margin_m": 300.0, "step_x": 4, "step_y": 3}, True),
    "rectangle": (
        {
            "roi_mode": "rectangle",
            "roi_margin_m": 0.0,
            "x_range": [400.0, 1300.0],
            "y_range": [200.0, 900.0],
            "step_x": 2,
            "step_y": 3,
        },
        False,
    ),
}


def _inline_schedule(run: Path, target: Path) -> None:
    """Copy ``run`` to ``target`` with its schedule include written into ``MODEL.DATA``."""
    target.mkdir(parents=True)
    for path in run.iterdir():
        if path.is_file() and path.name != f"{CASE}.DATA":
            os.symlink(path, target / path.name)
    text = (run / f"{CASE}.DATA").read_text()
    pattern = re.compile(r"INCLUDE\s*\n\s*'([^']*Schdl\.sch)'\s*/", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"{run}/{CASE}.DATA has no Schdl.sch include")
    inline = Path(match.group(1)).read_text()
    (target / f"{CASE}.DATA").write_text(text[: match.start()] + inline + text[match.end() :])


def _observations(folder: Path, avo: dict[str, object], work: Path) -> list[np.ndarray]:
    """Run get_avo_result on ``folder`` and return its vectors, one per vintage."""
    from misc import ecl
    from pylops.avo.avo import approx_zoeppritz_pp
    from subsurface.multphaseflow.flow_rock import flow_avo
    from subsurface.rockphysics import standardrp

    case = ecl.EclipseCase(str(folder / f"{CASE}.DATA"))
    rock = standardrp.elasticproperties({})
    rock.overburden = np.full(int(np.sum(case.grid()["ACTNUM"])), OVERBURDEN_MPA)
    if hasattr(rock, "p_init"):  # calc_pem then takes the vintage's pressure (unused anyway)
        del rock.p_init
    rock.input = {"ntg": "no", "press_conv": 0.1}

    model = flow_avo.__new__(flow_avo)
    model.no_flow = False
    model.input_dict = {}
    model.file = CASE
    model.startDate = dict(START)
    model.pem = rock
    model.pem_input = {"phases": ["OIL", "WAT", "GAS"], "baseline": BASELINE,
                       "vintage": list(VINTAGES)}  # fmt: skip
    model.ensemble_member = None
    model.pp_func = approx_zoeppritz_pp
    model.NX, model.NY, model.NZ = case.init.ni, case.init.nj, case.init.nk
    model.avo_config = {**AVO, **avo}
    save = work / "save"
    save.mkdir()
    cwd = os.getcwd()
    os.chdir(work)  # get_avo_result writes its masks in the working directory
    try:
        # With the separator, as PET passes its En_<i>/ folders: get_avo_result then reads
        # the grid and DZ through EclipseCase (without it, EclipseInit, which fails).
        model.get_avo_result(str(folder) + os.sep, str(save) + os.sep)
    finally:
        os.chdir(cwd)
    return [
        np.asarray(np.load(save / f"avo_vint{v}.npz", allow_pickle=True)["avo"], dtype=float)
        for v in range(len(VINTAGES))
    ]


def main(argv: list[str] | None = None) -> int:
    """Run every variant and write the reference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="the folder of an OPM Flow run of the truth")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    arrays: dict[str, np.ndarray] = {}
    with warnings.catch_warnings(), tempfile.TemporaryDirectory(dir=run.parent) as scratch:
        warnings.simplefilter("ignore", SyntaxWarning)
        _check_environment()
        for name, (avo, inline) in VARIANTS.items():
            work = Path(scratch) / name
            work.mkdir()
            folder = run
            if inline:
                folder = work / "run"
                _inline_schedule(run, folder)
            for days, vector in zip(VINTAGES, _observations(folder, avo, work), strict=True):
                arrays[f"out_{name}_{int(days)}"] = vector.ravel()
            arrays[f"in_{name}_avo"] = np.array(json.dumps({**AVO, **avo}))
        shutil.rmtree(scratch, ignore_errors=True)
    prt = (run / f"{CASE}.PRT").read_text(errors="replace").splitlines()
    flow = next((line.strip() for line in prt if "Flow Version" in line), "unknown")
    path = OUT / "flow_avo_import_3dbox.npz"
    np.savez_compressed(
        path,
        **arrays,
        in_start=np.array("1/1/2022"),
        in_baseline=np.array(BASELINE),
        in_vintages=np.array(VINTAGES),
        meta_simulatorwrap_commit=np.array(SIMULATORWRAP_COMMIT),
        meta_versions=np.array([f"{k} {v}" for k, v in VERSIONS.items()]),
        meta_flow_version=np.array(flow),
        meta_numpy_version=np.array(np.__version__),
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
