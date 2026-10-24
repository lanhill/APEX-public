"""The 3dBox AVO case on OPM Flow (opt-in: ``-m integration``, ~2 min).

Runs ``examples/3dbox_production_avo`` for real: the truth's AVO maps are the committed ones,
its traces are SimulatorWrap's own on a run of the same truth,
the restarts OPM Flow writes are the ones the deck's text predicts (also when the restart
control changes mid-schedule), and a short ES-MDA on rates and AVO maps runs end to end.
"""

import datetime as dt
import importlib.util
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from apex.domain.config import GeophysicsSection
from apex.domain.observations import DataKey
from apex.services import composition
from apex.simulation.deck import restart_schedule
from apex.simulation.geophysics import CaseState
from apex.workflows.geophysics import AVOSurvey, GeophysicsModel

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "3dbox_production_avo"
REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference" / "geophysics"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]


def _prepare() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "geophysics_3dbox_prepare", EXAMPLE / "prepare.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def truth(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The true model's run with the PET deck's ln PERMX cap of 6, once: SimulatorWrap's
    stored references below were read from a run of it."""
    case: Path = _prepare().simulate_truth(
        EXAMPLE, tmp_path_factory.mktemp("truth"), permx_cap=6.0, deck="deck_pet.mako"
    )
    return case


@pytest.fixture(scope="module")
def example_truth(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The true model's run as prepare.py makes the example's data (its permx_cap), once."""
    case: Path = _prepare().simulate_truth(EXAMPLE, tmp_path_factory.mktemp("example_truth"))
    return case


def test_the_truths_avo_maps_are_the_committed_ones(example_truth: Path) -> None:
    """The run, the restart reading, the PEM and the synthesis reproduce prepare.py's maps."""
    model = composition.build_geophysics(composition.load_config(EXAMPLE / "config_esmda.toml"))
    assert model is not None
    committed = np.load(EXAMPLE / "true_response.npz")
    for field in committed.files:
        name, vintage = field.split("@")
        n = model.field_sizes()[name]
        values = model.forecast(example_truth, [DataKey(vintage, name, i) for i in range(n)])
        # OPM Flow need not be bit-identical across hosts and thread counts.
        np.testing.assert_allclose(values, committed[field], rtol=1e-6, err_msg=field)


def test_the_traces_are_simulatorwraps_own_on_the_same_run(truth: Path, tmp_path: Path) -> None:
    """End to end against SimulatorWrap's flow_avo reading a run of this truth (restart
    reading, units, PEM, grid, synthesis), with its settings: sampling = "nearest", the
    case's PET set-up's 320 bar overburden. Measured within 3.3e-15 of the peak on the
    reference's OPM Flow build (meta_flow_version); restarts are float32, so another build,
    host or thread count may move them by about 1e-7, hence 1e-6 of the peak, as for the
    committed maps; and a trace on a knife edge of the staircase (an interface time within
    1e-7 s of its 0.5 ms grid) must agree only up to it, as in the import's test below."""
    reference = REFERENCE / "flow_avo_3dbox.npz"
    if not reference.exists():
        pytest.fail(f"missing {reference}; see tools/geophysics_reference/flow_avo_3dbox.py")
    ref = np.load(reference)
    config = composition.load_config(EXAMPLE / "config_esmda.toml")
    assert config.geophysics is not None
    np.save(tmp_path / "overburden.npy", np.full(18000, 320.0))
    section = config.geophysics.model_dump()
    section["overburden_pressure"] = {"file": tmp_path / "overburden.npy", "unit": "bar"}
    options = section["data"][0]["options"]
    options.update(form="traces", sampling="nearest")
    del options["attribute"]  # whole traces
    section["max_field_values"] = 10**8
    nearest = config.model_copy(update={"geophysics": GeophysicsSection.model_validate(section)})
    model = composition.build_geophysics(nearest)
    assert model is not None
    columns = ref["out_columns"]
    vintages = {2000: "2027-06-24", 4000: "2032-12-14"}
    edge = _knife_edge_traces(model, truth, list(vintages.values()), None)
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    times = survey.model.time.times
    reach = survey.model.wavelet.half_length + survey.model.fine_dt
    assert len(edge) <= 0.05 * survey.n_trace * len(vintages), f"{len(edge)} knife edges"
    for days, vintage in vintages.items():
        # (i + 60 j): the trace index of column (I, J) among the 3600, all active.
        first = np.array([edge.get((vintage, int(i + 60 * j)), np.inf) for i, j in columns])
        strict = times[np.newaxis, :] < first[:, np.newaxis] - reach
        for angle in (10, 20):
            expected = ref[f"out_{angle}_{days}"]
            n_time = expected.shape[1]
            keys = [DataKey(vintage, f"avo_{angle}", (i + 60 * j) * n_time + s)
                    for i, j in columns for s in range(n_time)]  # fmt: skip
            got = model.forecast(truth, keys).reshape(len(columns), n_time)
            scale = np.abs(expected).max()
            np.testing.assert_allclose(got[strict], expected[strict], rtol=0, atol=1e-6 * scale)


#: The importer's PET case around SimulatorWrap's observations (tools/geophysics_reference/
#: flow_avo_import_3dbox.py): the case's PET settings, MPa pressures with its 320 bar
#: overburden, a fixed baseline at the start.
PET_CASE = """
[ensemble]
ne = 4
state = "permx"
importstate = "prior.npz"

[dataassim]
scheme = "esmda"
data = "data.csv"
datavar = "var.csv"
[dataassim.mda]
tot_assim_steps = 1

[simulator]
runfile = "RUNFILE"
startdate = "1/1/2022"
pem = [["model", "standardrp elasticproperties"], ["baseline", 0.0],
       ["vintage", [2000.0, 4000.0]], ["overburden", "overburden.npz"],
       ["press_conv", 0.1], ["ntg", "no"]]
"""


@pytest.mark.parametrize("variant", ["fallback", "ellipse", "rectangle"])
def test_pet_import_of_simulatorwraps_own_observations(
    truth: Path, tmp_path: Path, variant: str
) -> None:
    """``apex pet-import`` of the AVO data SimulatorWrap's get_avo_result wrote for this
    truth (its mask: active columns, step_x/step_y, its region of interest; differences
    from a fixed baseline) gives a case whose forecast of the same run is those data,
    value by value, within the tolerance of the traces' test above."""
    import json

    from apex.storage.inputs import read_observation_section
    from apex.storage.pet_config import import_pet_case

    reference = REFERENCE / "flow_avo_import_3dbox.npz"
    if not reference.exists():
        pytest.fail(f"missing {reference}; see tools/geophysics_reference/flow_avo_import_3dbox.py")
    ref = np.load(reference)
    pet = tmp_path / "pet"
    pet.mkdir()
    deck = Path(f"{truth}.DATA").read_text()
    if variant == "ellipse":  # its completions inline, where SimulatorWrap finds them
        include = re.search(r"INCLUDE\s*\n\s*'([^']*Schdl\.sch)'\s*/", deck)
        assert include is not None
        schedule = Path(include.group(1)).read_text()
        deck = deck[: include.start()] + schedule + deck[include.end() :]
    (pet / "RUNFILE.mako").write_text(deck)
    avo = json.loads(str(ref[f"in_{variant}_avo"]))
    pairs = ", ".join(f"[{json.dumps(k)}, {json.dumps(v)}]" for k, v in avo.items())
    (pet / "CONFIG.toml").write_text(PET_CASE + f"avo = [{pairs}]\n")
    np.savez(pet / "overburden.npz", obvalues=np.full(18000, 32.0))
    np.savez(pet / "prior.npz", permx=np.zeros((18000, 4)))
    for days in (2000, 4000):
        np.savez(pet / f"avo_{days}.npz", avo=ref[f"out_{variant}_{days}"])
    (pet / "data.csv").write_text("days,avo\n2000,avo_2000.npz\n4000,avo_4000.npz\n")
    (pet / "var.csv").write_text("days,avo\n2000,\"['abs', 1e-4]\"\n4000,\"['abs', 1e-4]\"\n")

    result = import_pet_case(
        pet / "CONFIG.toml", tmp_path / "apex", grid_file=EXAMPLE / "REF.EGRID"
    )

    config = composition.load_config(result.config)
    model = composition.build_geophysics(config)
    assert model is not None
    observations = read_observation_section(config.observations)
    model.check_observations(observations.keys)
    observed = np.asarray(observations.values)
    got = model.forecast(truth, observations.keys)
    # A trace with an interface time within 1e-7 s of the staircase's 0.5 ms grid is on a
    # knife edge: the restarts are float32, so the two readers' velocities differ by about
    # 1e-7, enough to move it across and shift the samples from there on. Such a trace
    # must still agree before that time, less the wavelet's half-length; every other
    # value must agree throughout.
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    datum = config.geophysics.data[0]  # type: ignore[union-attr]
    edge = _knife_edge_traces(model, truth, datum.vintages, datum.avo().baseline)
    n_time = int(survey.window.sum())
    times = survey.model.time.times[survey.window]
    sample = np.array([k.index for k in observations.keys]) % n_time
    trace = np.array([k.index for k in observations.keys]) // n_time
    vintage = np.array([k.time for k in observations.keys])
    reach = survey.model.wavelet.half_length + survey.model.fine_dt
    strict = np.array(
        [
            (v, t) not in edge or times[s] < edge[v, t] - reach
            for v, t, s in zip(vintage, trace, sample, strict=True)
        ]
    )
    # Few: the baseline's uniform state repeats its interface times over the columns, so
    # one near-tie there marks several traces at once (5 of 240 for the fallback).
    assert len(edge) <= 0.05 * survey.n_trace * len(set(vintage)), f"{len(edge)} knife edges"
    np.testing.assert_allclose(
        got[strict], observed[strict], rtol=0, atol=1e-6 * np.abs(observed).max()
    )
    region = config.geophysics.data[0].avo().region  # type: ignore[union-attr]
    # The fallback keeps every active column; the others leave some out.
    assert (region is None) == (variant == "fallback")


def _knife_edge_traces(
    model: object, truth: Path, vintages: Sequence[str], baseline: str | None
) -> dict[tuple[str, int], float]:
    """Return ``(vintage, trace)`` whose interface times, at the vintage or the baseline,
    lie within 1e-7 s of the nearest sampling's fine grid, and the earliest such time."""

    assert isinstance(model, GeophysicsModel)
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    state = CaseState(truth)
    grid = state.grid
    i, j = survey.columns[:, 0], survey.columns[:, 1]
    time = survey.model.time
    fine = np.arange(time.t_min, time.t_max, survey.model.fine_dt)
    top = survey.model.overburden.two_way_time(grid.top[i, j, 0])

    def earliest(date: dt.date) -> np.ndarray:
        """Each trace's earliest knife-edge interface time, or inf."""
        vp = model._elastic(state, grid, date)[0][i, j, :]
        cum = top[:, None] + np.cumsum(2.0 * (grid.bottom - grid.top)[i, j, :] / vp, axis=1)
        gap = np.abs(cum[:, :, None] - fine[None, None, :]).min(axis=2)  # (n_trace, n_z)
        return np.where(gap < 1e-7, cum, np.inf).min(axis=1)

    base = np.full(len(survey.columns), np.inf)
    if baseline is not None:
        base = earliest(dt.date.fromisoformat(baseline))
    out: dict[tuple[str, int], float] = {}
    for v in vintages:
        first = np.minimum(earliest(dt.date.fromisoformat(v)), base)
        out.update({(v, int(t)): float(first[t]) for t in np.flatnonzero(np.isfinite(first))})
    return out


def test_the_restarts_written_are_those_the_deck_predicts(tmp_path: Path) -> None:
    """The restart control changes after the 4th report time: by opm-common's rule, the
    k-th report time follows the control in force when it is reached (block k - 1)."""
    from apex.core.simulation import MemberInput
    from apex.simulation.opm_flow import OPMFlowSimulator

    deck = (EXAMPLE / "deck_pet.mako").read_text()  # the PET schedule's TSTEP
    changed = deck.replace(
        "TSTEP\n10*400 /", "TSTEP\n4*400 /\n\nRPTRST\n BASIC=3 FREQ=2 /\n\nTSTEP\n6*400 /"
    )
    assert changed != deck
    template = tmp_path / "deck.mako"
    template.write_text(changed)
    predicted = restart_schedule(template, EXAMPLE)
    assert predicted.unresolved == ()

    simulator = OPMFlowSimulator(deck_template_file=str(template), case_dir=str(EXAMPLE))
    work = tmp_path / "run"
    work.mkdir()
    permx = np.load(EXAMPLE / "truth.npz")["permx"]
    case = simulator.simulate(MemberInput(-1, {"permx": permx}, work))

    written = CaseState(case).restart_dates
    assert written == [t.date() for t in predicted.restart_times]
    start = dt.date(2022, 1, 1)
    # Steps 0-4 (BASIC=2), then every second step counted from the start (BASIC=3 FREQ=2).
    assert written == [start + dt.timedelta(days=400 * k) for k in (0, 1, 2, 3, 4, 6, 8, 10)]


def test_a_short_esmda_on_rates_and_avo_maps(tmp_path: Path) -> None:
    """Every member runs (a regression: OPM Flow's round-off saturations of -4.8e-5 failed
    one), and the data mismatch falls. In a copy, so the example folder stays untouched."""
    example = tmp_path / "example"
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("runs", "__pycache__"))
    import tomllib

    import tomli_w

    path = example / "config_esmda.toml"
    raw = tomllib.loads(path.read_text())
    raw["prior"]["ensemble_size"] = 16  # the first 16 of the stored prior's 100 members
    raw["method"]["options"]["inflation"] = [1.0]
    raw["execution"]["options"]["workers"] = 4
    path.write_text(tomli_w.dumps(raw))
    # The rates and the maps' kept wavelet coefficients (the analysis data, not the raw maps).
    data = composition.build_run(composition.load_config(path)).workflow.analysis_observations

    result = composition.start(path)

    first, last = result.history[0], result.history[-1]
    assert last.data_mismatch_mean < first.data_mismatch_mean
    run = example / "runs" / "3dbox_production_avo_esmda"
    events = (run / "events.jsonl").read_text()
    assert '"member_failed"' not in events
    saved = np.load(run / "checkpoints" / "iter_0.npz")
    assert saved["predicted"].shape == (data.n_data, 16)
    assert data.n_data < _rates(example) + 4 * 3600


def _rates(example: Path) -> int:
    """Return the number of rate observations of the example (one per row of its CSV)."""
    with (example / "observations.csv").open() as handle:
        return sum(1 for _ in handle) - 1


# --- gravity and seafloor displacement (phase 3) ------------------------------------------

#: The reference's survey: a baseline at the start, vintages at days 2000 and 4000.
POTENTIAL_VINTAGES = {2000: "2027-06-24", 4000: "2032-12-14"}


def _potential_reference() -> dict[str, np.ndarray]:
    path = REFERENCE / "potential_3dbox.npz"
    if not path.exists():
        pytest.fail(f"missing {path}; see tools/geophysics_reference/flow_potential_3dbox.py")
    with np.load(path) as ref:
        return {k: ref[k] for k in ref.files}


def _potential_section(ref: dict[str, np.ndarray], **displacement: object) -> GeophysicsSection:
    """Gravity and displacement data with the reference's stations and settings."""
    stations = {
        "spacing": float(ref["in_meas_spacing"]),
        "padding": float(ref["in_padding"]),
        "depth": float(ref["in_water_depth"]),
    }
    survey = {"baseline": "2022-01-01", "stations": stations}
    return GeophysicsSection.model_validate(
        {
            "grid_file": EXAMPLE / "REF.EGRID",
            "data": [
                {
                    "kind": "gravity",
                    "datatype": "grav",
                    "vintages": list(POTENTIAL_VINTAGES.values()),
                    "options": {**survey, "gravitational_constant": 6.67e-11},
                },
                {
                    "kind": "seafloor_displacement",
                    "datatype": "subsidence",
                    "vintages": list(POTENTIAL_VINTAGES.values()),
                    "options": {
                        **survey,
                        "kernel": {
                            "name": "geertsma",
                            "options": {
                                "poisson": float(ref["in_poisson"]),
                                "truncation_factor": 3.0,
                                "free_surface": "datum",
                            },
                        },
                        **displacement,
                    },
                },
            ],
        }
    )


def _potential_model(section: GeophysicsSection) -> GeophysicsModel:
    from apex.algorithms.geophysics import displacement_kernels
    from apex.workflows.geophysics import build_geophysics_model

    kernel = section.data[1].displacement().kernel
    created = displacement_kernels.create(kernel.name, **kernel.options)
    return build_geophysics_model(section, None, [None, None], [None, None], None, [None, created])


def test_gravity_and_displacement_are_simulatorwraps_on_the_same_run(truth: Path) -> None:
    """End to end against SimulatorWrap's flow_grav and flow_seafloor_disp reading a run of
    this truth (restart reading, units, stations, masses, volume changes), with its settings:
    point masses and its G of 6.67e-11, the displacement from the baseline's pore volume
    (``volume = "pore"``) or RPORV's change, Geertsma truncated at 3 depths with the free
    surface at sea level. Measured on the reference's OPM Flow build, as shares of the
    peak: the cells' mass changes 6.5e-16, Δg 3.2e-16, the cells' volume decreases to
    rounding, and u_z 1.7e-5, the reference's cut-off of Geertsma's integral
    (``pet_parity.md`` G12), small here as every station is within about a reservoir depth
    of the cells. Restarts are float32, so another build, host or thread count may move
    them by about 1e-7: 1e-6 of the peak for the cell quantities and gravity, and for u_z
    1e-4, the cut-off's 1.7e-5 with headroom."""
    from apex.workflows.geophysics import CaseQuantities, DisplacementSurvey, GravitySurvey

    ref = _potential_reference()
    assert not bool(ref["meta_aquifer_filtered"])  # every active cell counts in both
    per_pa = float(ref["in_compressibility_per_mpa"]) * 1e-6
    by_pressure = _potential_model(_potential_section(ref, compressibility=per_pa, volume="pore"))
    by_rporv = _potential_model(_potential_section(ref, volume_change="pore_volume"))
    gravity = by_pressure.surveys[0]
    assert isinstance(gravity, GravitySurvey)
    np.testing.assert_allclose(gravity.stations, ref["out_stations"], rtol=0, atol=1e-9)
    state = CaseState(truth)
    quantities = CaseQuantities(state, int(by_pressure.reference.actnum.sum()))
    base = dt.date(2022, 1, 1)
    for days, vintage in POTENTIAL_VINTAGES.items():
        date = dt.date.fromisoformat(vintage)
        dm = quantities.fluid_mass(date) - quantities.fluid_mass(base)
        reference_dm = ref[f"out_grav_dm_{days}"]
        scale = np.abs(reference_dm).max()
        np.testing.assert_allclose(dm, reference_dm, rtol=0, atol=1e-6 * scale)
        n = gravity.field_size
        dg = by_pressure.forecast(truth, [DataKey(vintage, "grav", i) for i in range(n)])
        peak = np.abs(ref[f"out_grav_{days}"]).max()
        np.testing.assert_allclose(dg, ref[f"out_grav_{days}"], rtol=0, atol=1e-6 * peak)
        drop = quantities.pressure(base) - quantities.pressure(date)
        decreases = {
            "pressure": per_pa * quantities.pore_volume(base) * drop,
            "porv": quantities.pore_volume(base) - quantities.pore_volume(date),
        }
        for method, model in (("pressure", by_pressure), ("porv", by_rporv)):
            reference_dv = ref[f"out_disp_dv_{method}_{days}"]
            scale = np.abs(reference_dv).max()
            np.testing.assert_allclose(
                decreases[method], reference_dv, rtol=0, atol=1e-6 * scale, err_msg=method
            )
            survey = model.surveys[1]
            assert isinstance(survey, DisplacementSurvey)
            keys = [DataKey(vintage, "subsidence", i) for i in range(survey.field_size)]
            u = model.forecast(truth, keys)
            expected = ref[f"out_disp_{method}_{days}"]
            peak = np.abs(expected).max()
            assert peak > 0
            np.testing.assert_allclose(u, expected, rtol=0, atol=1e-4 * peak, err_msg=method)


def test_a_short_esmda_on_rates_gravity_and_displacement(
    example_truth: Path, tmp_path: Path
) -> None:
    """The whole path with OPM Flow (composition, validation of the deck's DEN and RPORV,
    the members' restarts, the data): every member runs, and the gravity and displacement
    data mismatch falls. The
    gravity and displacement observations are APEX's own forecasts of the truth (default
    settings: CODATA's G, the bulk volume, seabed stations as the free surface) with 2 %
    noise, in a copy of the example."""
    import tomllib

    import tomli_w

    example = tmp_path / "example"
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("runs", "__pycache__"))
    raw = tomllib.loads((example / "config_esmda.toml").read_text())
    raw["prior"]["ensemble_size"] = 16
    raw["method"]["options"]["inflation"] = [1.0]
    raw["execution"]["options"]["workers"] = 4
    del raw["observations"]["fields"]
    del raw["data_transforms"]  # the AVO maps' wavelet compression; this test has no maps
    survey = {"baseline": "2022-01-01", "stations": {"spacing": 600.0, "padding": 600.0,
                                                      "depth": 300.0}}  # fmt: skip
    raw["geophysics"] = {
        "grid_file": "REF.EGRID",
        "data": [
            {"kind": "gravity", "datatype": "grav", "vintages": list(POTENTIAL_VINTAGES.values()),
             "options": survey},
            {"kind": "seafloor_displacement", "datatype": "subsidence",
             "vintages": list(POTENTIAL_VINTAGES.values()),
             "options": {**survey, "compressibility": 1e-9}},
        ],
    }  # fmt: skip
    path = example / "config_esmda.toml"
    path.write_text(tomli_w.dumps(raw))
    model = composition.build_geophysics(composition.load_config(path))
    assert model is not None
    rng = np.random.default_rng(61)
    fields = []
    for name, size in model.field_sizes().items():
        for vintage in POTENTIAL_VINTAGES.values():
            values = model.forecast(example_truth, [DataKey(vintage, name, i) for i in range(size)])
            noise = 0.02 * float(np.abs(values).mean())
            file = f"{name}_{vintage}.npy"
            np.save(example / file, values + noise * rng.standard_normal(size))
            fields.append({"time": vintage, "datatype": name, "file": file, "variance": noise**2})
    raw["observations"]["fields"] = fields
    path.write_text(tomli_w.dumps(raw))

    from apex.services.validation import validate_history_match

    validation = validate_history_match(path)
    assert validation.valid, validation.errors
    result = composition.start(path)

    first, last = result.history[0], result.history[-1]
    # The gravity and displacement data's share falls; the total is the monthly rates' (over
    # 99.9 % of it), which one step of 16 members (alpha = 1) need not improve.
    assert first.metrics is not None
    assert last.metrics is not None
    for datatype in ("grav", "subsidence"):
        before = first.metrics["data_mismatch"][datatype]
        assert last.metrics["data_mismatch"][datatype] < before, datatype
    run = example / "runs" / "3dbox_production_avo_esmda"
    assert '"member_failed"' not in (run / "events.jsonl").read_text()
    saved = np.load(run / "checkpoints" / "iter_0.npz")
    assert saved["predicted"].shape == (_rates(example) + 4 * 25, 16)


@pytest.mark.parametrize("method", ["pressure", "porv"])
def test_pet_import_of_simulatorwraps_own_gravity_and_displacement(
    truth: Path, tmp_path: Path, method: str
) -> None:
    """``apex pet-import`` of a PET case whose grav and sea_disp settings are the
    reference's above, with SimulatorWrap's own data of this truth as observations, gives
    a case whose forecast of the same run is those data, within the tolerances of the
    comparison above."""
    from apex.storage.inputs import read_observation_section
    from apex.storage.pet_config import import_pet_case

    ref = _potential_reference()
    pet = tmp_path / "pet"
    pet.mkdir()
    (pet / "RUNFILE.mako").write_text(Path(f"{truth}.DATA").read_text())
    stations = (
        f'["meas_spacing", {float(ref["in_meas_spacing"])}], '
        f'["padding", {float(ref["in_padding"])}], '
        f'["water_depth", {float(ref["in_water_depth"])}]'
    )
    survey = f'["baseline", 0], ["vintage", [2000, 4000]], {stations}'
    (pet / "CONFIG.toml").write_text(
        PET_CASE.split("pem = ")[0]
        + 'pem = [["model", "standardrp elasticproperties"], ["phases", "OIL WAT GAS"], '
        '["press_conv", 0.1]]\n'
        f"grav = [{survey}]\n"
        f'sea_disp = [{survey}, ["method", "{method}"], ["model", "geertsma"], '
        f'["poisson", {float(ref["in_poisson"])}], '
        f'["compressibility", {float(ref["in_compressibility_per_mpa"])}], ["z_base", 0]]\n'
    )
    np.savez(pet / "prior.npz", permx=np.zeros((18000, 4)))
    for days in POTENTIAL_VINTAGES:
        np.savez(pet / f"grav_{days}.npz", grav=ref[f"out_grav_{days}"])
        np.savez(pet / f"sea_disp_{days}.npz", disp=ref[f"out_disp_{method}_{days}"])
    (pet / "data.csv").write_text(
        "days,grav,sea_disp\n"
        + "".join(f"{d},grav_{d}.npz,sea_disp_{d}.npz\n" for d in POTENTIAL_VINTAGES)
    )
    (pet / "var.csv").write_text(
        "days,grav,sea_disp\n"
        + "".join(f"{d},\"['abs', 1]\",\"['abs', 0.01]\"\n" for d in POTENTIAL_VINTAGES)
    )

    result = import_pet_case(
        pet / "CONFIG.toml", tmp_path / "apex", grid_file=EXAMPLE / "REF.EGRID"
    )

    config = composition.load_config(result.config)
    model = composition.build_geophysics(config)
    assert model is not None
    observations = read_observation_section(config.observations)
    model.check_observations(observations.keys)
    got = model.forecast(truth, observations.keys)
    observed = np.asarray(observations.values)
    for datatype, tolerance in (("grav", 1e-6), ("sea_disp", 1e-4)):
        rows = np.array([k.datatype == datatype for k in observations.keys])
        assert rows.sum() == 2 * 25
        peak = np.abs(observed[rows]).max()
        np.testing.assert_allclose(
            got[rows], observed[rows], rtol=0, atol=tolerance * peak, err_msg=datatype
        )
