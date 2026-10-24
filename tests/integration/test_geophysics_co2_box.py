"""A CO₂ store monitored by seismic data on OPM Flow (opt-in).

The CO₂ box (``tests/integration/data/co2store/box.mako``, APEX's own): the 3dBox AVO
case's grid as a CO2STORE gas-water deck, a tight caprock over four reservoir layers whose
log-permeability is the state (the 3dBox truth's layers 2-5 for the truth), one CO₂
injector at the base of the centre column for 5 years, 40 °C. Its data are the injector's
BHP, the near stack's RMS at each monitor minus the baseline's, and the time shifts, with a
baseline at the start and monitors at 2 and 5 years. On the truth's run:

- the plume shows: time shifts that grow from the first monitor to the second and stay
  local, and RMS changes;
- the fluid models order the time shifts as the physics says (a softer pore fluid is a
  slower rock): Wood's uniform saturation is the slowest, the patchiness blend at q = 0.7
  between, patchy saturation the fastest, column by column; Batzle & Wang's extrapolated
  correlation, 2.6-6 times too stiff for CO₂, gives smaller shifts than Span & Wagner;
- a short ES-MDA with the patchiness uncertain (``pem_q``, logit) runs every member and
  changes ``pem_q`` with the permeability (one step of 8 members: no claim about its spread).

Needs OPM Flow and CoolProp (the ``co2`` extra); under a minute.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from apex.core.simulation import MemberInput
from apex.domain.observations import DataKey
from apex.services import composition
from apex.simulation.deck import read_summary
from apex.simulation.opm_flow import OPMFlowSimulator
from apex.workflows.geophysics import GeophysicsModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]
pytest.importorskip("CoolProp")

DECK = Path(__file__).resolve().parent / "data" / "co2store" / "box.mako"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "3dbox_production_avo"
VINTAGES = ("2022-01-01", "2025-01-01")
N_COLUMNS = 3600

CONFIG = """
[run]
name = "co2_box"
seed = 2061

[[state.variables]]
name = "permx"
size = 14400
grid = [60, 60, 4]

[[state.variables]]
name = "pem_q"
size = 1

[prior]
ensemble_size = 8

[prior.generator]
name = "gaussian_field"
options = { grid = [60, 60, 4], mean = 5.0, variance = 1.0, major_range = 10.0 }

[prior.variables.pem_q]
name = "gaussian"
options = { mean = 0.0, std = 1.5 }

[observations]
file = "observations.csv"
#FIELDS
[case]
kind = "synthetic"
truth_file = "truth.npz"

[simulator]
name = "opm_flow"
options = { deck_template_file = "box.mako", timeout_s = 900, keep_files = false }

[execution]
name = "process"
options = { workers = 4 }

[method]
name = "esmda"
options = { inflation = [1.0] }
strategy = { name = "approx", options = { energy = 0.98 } }

[geophysics]
grid_file = "REF.EGRID"
overburden_pressure = { gradient = 22600.0, surface = 101325.0 }

[geophysics.pem]
name = "soft_sand"

[geophysics.pem.options]
temperature_c = 40.0
salinity = 0.035
critical_porosity = 0.4
#FLUID

[[geophysics.data]]
kind = "avo"
datatype = "rms"
vintages = ["2022-01-01", "2025-01-01"]

[geophysics.data.options]
angles = [10]
reflectivity = "zoeppritz"
wavelet = { frequency = 45.0, length = 0.05 }
overburden = { vp = 2450.0, vs = 849.0, density = 2300.0 }
times = { start = 1.9224, end = 2.2150, step = 0.001 }
four_d = "differences"
baseline = "2020-01-01"

[[geophysics.data]]
kind = "avo"
datatype = "tshift"
vintages = ["2022-01-01", "2025-01-01"]

[geophysics.data.options]
angles = [10]
reflectivity = "zoeppritz"
wavelet = { frequency = 45.0, length = 0.05 }
overburden = { vp = 2450.0, vs = 849.0, density = 2300.0 }
times = { start = 1.9224, end = 2.2150, step = 0.001 }
attribute = "time_shift"
baseline = "2020-01-01"
"""

#: Span & Wagner's CO₂ with the patchiness uncertain: the truth's q = 0.7 is the fallback.
UNCERTAIN = (
    'mixing = "patchiness"\npatchiness = { state = "pem_q", transform = "logit", value = 0.7 }'
)


def _case(folder: Path) -> Path:
    """Write the case's deck, porosity and truth into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DECK, folder / "box.mako")
    # The 3dBox example's porosity and true log-permeability, 18,000 cells in F order (I
    # fastest, layers slowest), so layer 1 is the first 3,600 values.
    text = (EXAMPLE / "poro.inc").read_text()
    body = re.search(r"^PORO\s*$(.*?)^\s*/", text, re.M | re.S)
    assert body is not None
    poro = np.array([float(v) for v in body.group(1).split()])
    assert poro.size == 5 * N_COLUMNS
    poro[:N_COLUMNS] = 0.05  # layer 1, the caprock
    values = "\n".join(f"{p:.4f}" for p in poro)
    (folder / "poro.inc").write_text(f"PORO\n{values}\n/\n")
    permx = np.load(EXAMPLE / "truth.npz")["permx"][N_COLUMNS:]  # layers 2-5 (F order)
    np.savez(folder / "truth.npz", permx=permx)
    return folder


def _config(folder: Path, fluid: str = UNCERTAIN, fields: str = "") -> Path:
    path = folder / "config.toml"
    path.write_text(CONFIG.replace("#FLUID", fluid).replace("#FIELDS", fields))
    return path


@pytest.fixture(scope="module")
def truth(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The case's folder and the truth's run (its case path), with the grid and BHP data."""
    folder = _case(tmp_path_factory.mktemp("co2_box"))
    simulator = OPMFlowSimulator(deck_template_file=str(folder / "box.mako"), case_dir=str(folder))
    permx = np.load(folder / "truth.npz")["permx"]
    work = folder / "truth_run"
    work.mkdir()
    case: Path = simulator.simulate(MemberInput(-1, {"permx": permx}, work))
    shutil.copy2(case.with_suffix(".EGRID"), folder / "REF.EGRID")
    from resdata.summary import Summary

    report = [d.isoformat()[:10] for d in Summary(str(case)).report_dates]
    dates = [t for t in report if t != "2020-01-01"]  # the deck's report times after START
    assert len(dates) == 20
    assert set(VINTAGES) <= set(dates)
    bhp = read_summary(case, [DataKey(t, "WBHP:INJ") for t in dates])
    assert np.all(bhp > 240.0)  # injecting above the hydrostatic pressure
    lines = [f"{t},WBHP:INJ,{float(b)!r},1.0" for t, b in zip(dates, bhp, strict=True)]
    (folder / "observations.csv").write_text("time,datatype,value,variance\n" + "\n".join(lines))
    return folder, case


def _model(folder: Path, fluid: str) -> GeophysicsModel:
    model = composition.build_geophysics(composition.load_config(_config(folder, fluid)))
    assert model is not None
    return model.for_state({})


def _maps(model: GeophysicsModel, case: Path, name: str) -> dict[str, np.ndarray]:
    return {
        v: model.forecast(case, [DataKey(v, name, i) for i in range(N_COLUMNS)]) for v in VINTAGES
    }


def test_the_plume_shows_in_the_time_shifts_and_amplitudes(truth: tuple[Path, Path]) -> None:
    """Time shifts that grow and stay local, and RMS changes where they are."""
    folder, case = truth
    model = _model(folder, UNCERTAIN)
    assert model.field_sizes() == {"rms_10": N_COLUMNS, "tshift_10": N_COLUMNS}
    shift, rms = _maps(model, case, "tshift_10"), _maps(model, case, "rms_10")
    early, late = shift[VINTAGES[0]], shift[VINTAGES[1]]
    assert np.isfinite(early).all()
    assert np.isfinite(late).all()
    assert 1e-4 < early.max() < late.max() < 0.05  # a slower plume: 0.1-50 ms
    assert late.sum() > early.sum()
    plume = late > 0.1 * late.max()
    assert plume.mean() < 0.5  # local, not the whole box
    change = np.abs(rms[VINTAGES[1]])
    assert np.isfinite(change).all()
    assert change[plume].mean() > 10.0 * change[~plume].mean()  # the amplitudes see it too


def test_the_fluid_models_order_the_time_shifts(truth: tuple[Path, Path]) -> None:
    """A softer pore fluid is a slower rock, column by column.

    The baseline is all brine, the same in every model, and the densities are the
    simulator's, so the P-wave modulus orders Vp: Wood's bound is below Gassmann-Hill's,
    patchiness blends the two, and Wood's fluid modulus grows with CO₂'s, which Batzle &
    Wang overstate here (not in general: in this pressure-temperature window).
    """
    folder, case = truth
    fluids = {
        "wood": 'mixing = "wood"',
        "q07": UNCERTAIN,
        "patchy": 'mixing = "patchy"',
        "batzle_wang": (
            'gas_model = "batzle_wang"\nextrapolate_gas_correlation = true\nmixing = "wood"'
        ),
    }
    late = {k: _maps(_model(folder, f), case, "tshift_10")[VINTAGES[1]] for k, f in fluids.items()}
    tol = 1e-9
    assert np.all(late["wood"] >= late["q07"] - tol)
    assert np.all(late["q07"] >= late["patchy"] - tol)
    assert np.all(late["batzle_wang"] <= late["wood"] + tol)
    # ...and strictly so under the plume.
    plume = late["wood"] > 0.1 * late["wood"].max()
    assert np.all(late["wood"][plume] > late["q07"][plume])
    assert np.all(late["q07"][plume] > late["patchy"][plume])
    assert np.all(late["batzle_wang"][plume] < late["wood"][plume])


def test_a_short_esmda_updates_the_patchiness_with_the_permeability(
    truth: tuple[Path, Path],
) -> None:
    """Every member runs, the data mismatch falls, and pem_q changes with the permeability."""
    folder, case = truth
    model = _model(folder, UNCERTAIN)
    fields = ""
    rng = np.random.default_rng(2062)
    for name in ("rms_10", "tshift_10"):
        for vintage, values in _maps(model, case, name).items():
            std = 5e-4 if name.startswith("tshift") else 0.05 * float(np.abs(values).mean())
            noisy = values + std * rng.standard_normal(values.size)
            np.save(folder / f"{name}_{vintage}.npy", noisy)
            fields += (
                f'\n[[observations.fields]]\ntime = "{vintage}"\ndatatype = "{name}"\n'
                f'file = "{name}_{vintage}.npy"\nvariance = {std**2!r}\n'
            )
    result = composition.start(_config(folder, UNCERTAIN, fields))

    run = folder / "runs" / "co2_box"
    assert '"member_failed"' not in (run / "events.jsonl").read_text()
    first, last = result.history[0], result.history[-1]
    assert last.data_mismatch_mean < first.data_mismatch_mean
    prior = np.load(run / "checkpoints" / "iter_0.npz")
    posterior = np.load(run / "checkpoints" / f"iter_{len(result.history) - 1}.npz")
    assert prior["state"].shape == (14400 + 1, 8)  # permx, then pem_q
    assert prior["predicted"].shape == (20 + 4 * N_COLUMNS, 8)  # BHP, then 2 x 2 maps
    assert not np.allclose(prior["state"][-1], posterior["state"][-1])  # pem_q changed
    assert not np.allclose(prior["state"][:-1], posterior["state"][:-1])  # with permx
