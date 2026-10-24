"""History matching with AVO data: field keys through the whole workflow.

A stand-in deck simulator writes what a run of a small box would leave (EGRID, INIT, a
restart at the vintage) with each member's water saturation, and a scalar summary datum.
The real AVO forward model reads the files, so the field keys flow through the methods,
the checkpoints and the composition's wrappers as they would with OPM Flow.
"""

import datetime as dt
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.core.plugins import model_error_models, simulators
from apex.core.simulation import MemberInput
from apex.domain.config import (
    FileWaveletSection,
    HistoryMatchConfig,
    OverburdenPressureFile,
    PluginSection,
    RegionAroundWellsSection,
    RegionFileSection,
)
from apex.domain.observations import DataKey, ObservationSet
from apex.services import composition
from apex.simulation.geophysics import GeophysicsSimulator, Grid
from apex.simulation.model_error import RESIDUAL_PREFIX, ModelErrorSimulator
from apex.workflows.geophysics import AVOSurvey, GeophysicsModel


class _Writers:
    """The conftest's case writers, for helpers that are not tests (set per test)."""

    write_case: Callable[..., Path]
    write_grid: Callable[..., Path]


_cases = _Writers()


@pytest.fixture(autouse=True)
def _case_writers(write_case: Callable[..., Path], write_grid: Callable[..., Path]) -> None:
    _cases.write_case, _cases.write_grid = write_case, write_grid


SHAPE = (4, 3, 3)
N_CELLS = 36
VINTAGE = dt.date(2020, 7, 1)
TRUTH = np.clip(0.5 + 0.3 * np.sin(np.arange(N_CELLS) / 3.0), 0.15, 0.85)


class BoxFlow:
    """A deck simulator stand-in: the member's ``swat`` is the restart's SWAT at VINTAGE."""

    needs_work_dir = True

    def simulate(self, member: MemberInput) -> Path:
        assert member.work_dir is not None
        assert set(member.state) == {"swat"}  # model-error parameters stay out of the deck
        swat = np.clip(np.asarray(member.state["swat"]), 0.05, 0.95)
        restarts = {VINTAGE: {"PRESSURE": 250.0, "SWAT": swat}}
        return _cases.write_case(member.work_dir / "MODEL", SHAPE, restarts, porosity=0.25)

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        from apex.simulation.geophysics import CaseState

        _, saturations, _ = CaseState(case).flow_state(VINTAGE, densities=False)
        assert all(k.index is None and k.datatype == "SWMEAN" for k in keys)
        return np.full(len(keys), saturations["water"].mean())

    def clean(self, case: Path) -> None:
        for path in case.parent.glob(f"{case.name}.*"):
            path.unlink()

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        case = self.simulate(member)
        values = self.read(case, keys)
        self.clean(case)
        return values


class ScalarOffset:
    """``r = theta`` on scalar data, 0 elsewhere (pg_lstm's rule for field keys)."""

    n_params = 1
    state_variable = "nn_params"

    def fit(self, observations: ObservationSet) -> None:
        pass

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        return [DataKey(k.time, RESIDUAL_PREFIX + k.datatype) for k in keys if k.index is None]

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return np.array([0.0 if k.index is not None else theta[0] for k in keys])


@pytest.fixture(autouse=True)
def plugins() -> Iterator[None]:
    simulators.register("test_box_flow")(BoxFlow)
    model_error_models.register("test_scalar_offset")(ScalarOffset)
    try:
        yield
    finally:
        simulators._factories.pop("test_box_flow")
        model_error_models._factories.pop("test_scalar_offset")


GEOPHYSICS = """
[geophysics]
pem = { name = "standard" }
grid_file = "REF.EGRID"
# 2300 kg/m3 * 9.81 m/s2 over the cell centre's depth: 45.3 MPa at 2000 m.
overburden_pressure = { gradient = 22600.0, surface = 101325.0 }

[[geophysics.data]]
kind = "avo"
datatype = "avo"
vintages = ["2020-07-01"]

[geophysics.data.options]
angles = [10, 25]
wavelet = { frequency = 45.0, length = 0.05 }
overburden = { vp = 2450.0, vs = 849.0, density = 2300.0 }
times = { start = 1.62, end = 1.67, step = 0.002 }
"""


def _case(directory: Path, *, mec: bool = False, geophysics: bool = True) -> Path:
    """Write the case: the truth's AVO maps and mean saturation, observed with noise."""
    directory.mkdir(parents=True, exist_ok=True)
    _cases.write_grid(directory / "REF.EGRID", SHAPE)
    truth_case = BoxFlow().simulate(MemberInput(0, {"swat": TRUTH}, directory))
    rng = np.random.default_rng(5)
    config_text = _config(mec, geophysics)
    if geophysics:
        raw = _raw(config_text.replace('"REF.EGRID"', f'"{directory / "REF.EGRID"}"'))
        model = composition.build_geophysics(HistoryMatchConfig.model_validate(raw))
        assert model is not None
        for angle in ("10", "25"):
            n = model.field_sizes()[f"avo_{angle}"]
            keys = [DataKey(VINTAGE.isoformat(), f"avo_{angle}", i) for i in range(n)]
            values = model.forecast(truth_case, keys)
            noise = 0.02 * np.abs(values).mean()
            np.save(directory / f"avo_{angle}.npy", values + noise * rng.standard_normal(n))
            (directory / f"avo_{angle}.var").write_text(repr(float(noise**2)))
    mean = float(TRUTH.mean()) + 0.005 * rng.standard_normal()
    (directory / "observations.csv").write_text(
        f"time,datatype,value,variance\n{VINTAGE.isoformat()},SWMEAN,{mean!r},2.5e-5\n"
    )
    np.savez(directory / "truth.npz", swat=TRUTH)
    path = directory / "config.toml"
    fields = "".join(
        f'\n[[observations.fields]]\ntime = "{VINTAGE.isoformat()}"\ndatatype = "avo_{a}"\n'
        f'file = "avo_{a}.npy"\nvariance = {(directory / f"avo_{a}.var").read_text()}\n'
        for a in ("10", "25")
        if geophysics
    )
    path.write_text(config_text.replace("#FIELDS", fields))
    return path


def _raw(text: str) -> dict[str, object]:
    import tomllib

    return tomllib.loads(text.replace("#FIELDS", ""))


def _config(mec: bool, geophysics: bool) -> str:
    theta = """
[[state.variables]]
name = "nn_params"
size = 1

[prior.variables.nn_params]
name = "gaussian"
options = { mean = 0.0, std = 0.01 }

[model_error]
name = "test_scalar_offset"
"""
    return f"""
[run]
name = "run"
seed = 3

[[state.variables]]
name = "swat"
size = {N_CELLS}
lower = 0.05
upper = 0.95
{theta if mec else ""}
[prior]
ensemble_size = 40
generator = {{ name = "gaussian", options = {{ mean = 0.5, std = 0.15 }} }}

[observations]
file = "observations.csv"
#FIELDS
[case]
truth_file = "truth.npz"

[simulator]
name = "test_box_flow"

[method]
name = "esmda"
options = {{ inflation = [2.0, 2.0] }}
{GEOPHYSICS if geophysics else ""}
"""


def test_avo_maps_are_matched_and_checkpointed(tmp_path: Path) -> None:
    result = composition.start(_case(tmp_path))

    first, last = result.history[0], result.history[-1]
    assert last.data_mismatch_mean < 0.2 * first.data_mismatch_mean
    saved = np.load(tmp_path / "runs" / "run" / "checkpoints" / "iter_0.npz")
    assert saved["predicted"].shape[0] == 1 + 2 * 12  # SWMEAN and two maps of 12 columns


def test_model_error_wraps_geophysics_and_passes_the_fields_through(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path, mec=True))
    run = composition.build_run(config)
    simulator = run.workflow.simulator
    assert isinstance(simulator, ModelErrorSimulator)
    assert isinstance(simulator.inner, GeophysicsSimulator)
    assert isinstance(simulator.inner.inner, BoxFlow)

    state = {"swat": TRUTH, "nn_params": np.array([0.25])}
    keys = [DataKey("2020-07-01", "SWMEAN"), DataKey("2020-07-01", "avo_10", 3)]
    work = tmp_path / "member"
    work.mkdir()
    values = simulator.run(MemberInput(0, state, work), keys)
    expected = np.load(tmp_path / "avo_10.npy")[3]  # observed: the truth's plus noise
    assert values[0] == pytest.approx(TRUTH.mean() + 0.25, abs=1e-6)
    assert values[1] == pytest.approx(expected, rel=0.2)


def test_what_geophysics_cannot_support_is_refused(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    assert config.geophysics.grid_file == (tmp_path / "REF.EGRID").resolve()
    levels = config.model_copy(
        update={"method": config.method.model_copy(update={"options": {"levels": []}})}
    )
    with pytest.raises(ConfigError, match="cannot be combined with fidelity levels"):
        composition.build_geophysics(levels)

    from apex.integrations.ert.export import _check_exportable

    with pytest.raises(ConfigError, match=r"geophysical data \(\[geophysics\]"):
        _check_exportable(config)

    assert config.geophysics.pem is not None
    unknown = config.model_copy(
        update={
            "geophysics": config.geophysics.model_copy(
                update={"pem": config.geophysics.pem.model_copy(update={"name": "nope"})}
            )
        }
    )
    with pytest.raises(ConfigError, match=r"\[geophysics\]: .*nope"):
        composition.build_geophysics(unknown)


class InMemory:
    """A simulator whose runs leave no files."""

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.zeros(len(keys))


def test_a_simulator_without_files_is_refused_with_geophysics(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    in_memory = config.model_copy(
        update={"simulator": config.simulator.model_copy(update={"name": "test_in_memory"})}
    )
    simulators.register("test_in_memory")(InMemory)
    try:
        with pytest.raises(ConfigError, match="runs leave files"):
            composition.build_run(in_memory)
    finally:
        simulators._factories.pop("test_in_memory")


def test_the_geophysics_plugins_are_checked_and_listed(tmp_path: Path) -> None:
    from apex.services.plugins import plugin_sections

    config = composition.load_config(_case(tmp_path))
    names = {(r, s.name) for r, s in plugin_sections(config)}
    assert {("petro_elastic_models", "standard"), ("reflectivity", "zoeppritz"),
            ("avo_attributes", "rms")} <= names  # fmt: skip


def test_a_guest_needs_the_geophysics_plugins_allowed(tmp_path: Path) -> None:
    from apex.services.access import Access, UIAccess

    config = composition.load_config(_case(tmp_path))
    rules = UIAccess.model_validate({
        "plugins": {"methods": ["esmda"], "simulators": ["test_box_flow"],
                    "executors": ["serial"], "priors": ["gaussian"],
                    "reflectivity": ["zoeppritz"]},
    })  # fmt: skip
    assert Access(rules, [], "alice").hidden_plugins(config) == [
        "avo_attributes:rms",
        "petro_elastic_models:standard",
    ]


# --- validation (phase 2, step 3) ----------------------------------------


def _with_deck(config: HistoryMatchConfig, tmp_path: Path, schedule: str) -> HistoryMatchConfig:
    """``config`` with a deck template whose SCHEDULE is ``schedule`` (only its text is read)."""
    deck = tmp_path / "deck.mako"
    deck.write_text(
        f"RUNSPEC\nSTART\n 1 JAN 2020 /\nGRID\nSOLUTION\nSUMMARY\nSCHEDULE\n{schedule}\nEND\n"
    )
    options = {"deck_template_file": str(deck)}
    return config.model_copy(
        update={"simulator": config.simulator.model_copy(update={"options": options})}
    )


def test_the_decks_restarts_at_the_vintages_are_checked_before_the_run(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    ok = _with_deck(config, tmp_path, "RPTRST\n BASIC=2 /\nDATES\n 1 JUL 2020 /\n/")
    assert composition.geophysics_restart_warnings(ok) == []

    off_report = _with_deck(config, tmp_path, "RPTRST\n BASIC=2 /\nDATES\n 1 JUN 2020 /\n/")
    with pytest.raises(ConfigError, match=r"2020-07-01 \(not a report time\).*RPTRST BASIC=2"):
        composition.geophysics_restart_warnings(off_report)

    no_restart = _with_deck(
        config, tmp_path, "RPTRST\n BASIC=3 FREQ=2 /\nDATES\n 1 JUL 2020 /\n 1 AUG 2020 /\n/"
    )
    with pytest.raises(ConfigError, match=r"2020-07-01 \(a report time without a restart\)"):
        composition.geophysics_restart_warnings(no_restart)

    templated = _with_deck(
        config, tmp_path, "RPTRST\n BASIC=2 /\n% for d in dates:\nDATES\n ${d} /\n/\n% endfor"
    )
    (warning,) = composition.geophysics_restart_warnings(templated)
    assert "as far as its text tells" in warning
    assert "Mako lines around DATES" in warning


def test_untransformed_field_data_beyond_max_field_values_are_refused(tmp_path: Path) -> None:
    """Only untransformed fields count; a wavelet restricted to some times covers those."""
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    small = config.model_copy(
        update={"geophysics": config.geophysics.model_copy(update={"max_field_values": 20})}
    )
    with pytest.raises(
        ConfigError, match=r"24 untransformed field values per member \(avo_10, avo_25\)"
    ):
        composition.build_geophysics(small)

    def wavelet(datatype: str, **options: object) -> PluginSection:
        return PluginSection(name="wavelet", options={"datatype": datatype, **options})

    one = small.model_copy(update={"data_transforms": [wavelet("avo_10")]})
    assert composition.build_geophysics(one) is not None  # avo_25's 12 values fit
    other_time = small.model_copy(
        update={"data_transforms": [wavelet("avo_10", times=["2030-01-01"])]}
    )
    with pytest.raises(ConfigError, match=r"24 untransformed"):
        composition.build_geophysics(other_time)
    at_vintage = small.model_copy(
        update={"data_transforms": [wavelet("avo_10", times=["2020-07-01"])]}
    )
    assert composition.build_geophysics(at_vintage) is not None
    # Dates in another ISO form ("20200701") are the same vintages (the review of the
    # coder's fixes): the observed fields count, and a wavelet's times cover them.
    basic = [f.model_copy(update={"time": "20200701"}) for f in small.observations.fields]
    compact = small.model_copy(
        update={"observations": small.observations.model_copy(update={"fields": basic})}
    )
    with pytest.raises(ConfigError, match=r"24 untransformed"):
        composition.build_geophysics(compact)
    compact_wavelet = compact.model_copy(
        update={"data_transforms": [wavelet("avo_10", times=["20200701"])]}
    )
    assert composition.build_geophysics(compact_wavelet) is not None


def test_includes_under_case_dir_are_read_for_the_restart_check(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "schedule.inc").write_text("RPTRST\n BASIC=2 /\nDATES\n 1 JUL 2020 /\n/\n")
    deck = _with_deck(config, tmp_path, "INCLUDE\n '${case_dir}/schedule.inc' /")
    options = {**deck.simulator.options, "case_dir": str(folder)}
    with_case_dir = deck.model_copy(
        update={"simulator": deck.simulator.model_copy(update={"options": options})}
    )
    assert composition.geophysics_restart_warnings(with_case_dir) == []
    before = _with_deck(config, tmp_path, "RPTRST\n BASIC=2 /\nDATES\n 1 JUL 2020 /\n/")
    early = before.model_copy(
        update={"geophysics": before.geophysics.model_copy(  # type: ignore[union-attr]
            update={"data": [before.geophysics.data[0].model_copy(  # type: ignore[union-attr]
                update={"vintages": ["2019-07-01"]})]})}
    )  # fmt: skip
    with pytest.raises(ConfigError, match=r"2019-07-01 \(before START\)"):
        composition.geophysics_restart_warnings(early)


def test_the_plan_states_sizes_and_the_geophysics_warnings(tmp_path: Path) -> None:
    from apex.services.validation import validate_history_match

    path = _case(tmp_path)
    path.write_text(path.read_text().replace("step = 0.002", "step = 0.005"))
    validation = validate_history_match(path)
    # The maps' size does not depend on the time step, so the observations still fit.
    assert validation.valid, validation.errors
    plan = validation.plan
    assert plan is not None
    n_data = 1 + 2 * 12  # SWMEAN and two maps of 12 columns
    mb = 8 * 40 / 1e6
    # ES-MDA saves each step's perturbed observations with it.
    assert plan.checkpoint_mb == round((N_CELLS + 2 * n_data) * mb, 1)
    assert plan.forecast_memory_mb == round(2 * n_data * mb, 1)
    text = " | ".join(plan.warnings)
    assert "has no deck_template_file" in text
    assert "above the wavelet's Nyquist interval" in text


def test_the_sizes_follow_the_method_and_the_kept_raw_forecasts(tmp_path: Path) -> None:
    """GIES saves its perturbations once; raw forecasts and transforms add their rows."""
    from apex.services.validation import validate_history_match

    path = _case(tmp_path)
    text = path.read_text().replace(
        'name = "esmda"\noptions = { inflation = [2.0, 2.0] }',
        'name = "gies"\nmax_iterations = 2',
    )
    text += (
        '\n[[data_transforms]]\nname = "wavelet"\n'
        'options = { datatype = "avo_10", grid = [4, 3, 1], wavelet = "haar", level = 1 }\n'
        "\n[output]\nsave_raw_forecasts = true\n"
    )
    path.write_text(text)
    validation = validate_history_match(path)
    assert validation.valid, validation.errors
    plan = validation.plan
    assert plan is not None
    run = composition.build_run(composition.load_config(path))
    n_raw, n_data = 25, run.workflow.observations.n_data
    mb = 8 * 40 / 1e6
    assert plan.checkpoint_mb == round((N_CELLS + n_data + n_raw) * mb, 1)
    assert plan.forecast_memory_mb == round((2 * n_raw + n_data) * mb, 1)


def test_a_region_around_wells_reads_the_decks_completions(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    datum = config.geophysics.data[0]
    region = RegionAroundWellsSection(shape="rectangle", margin=0.0)
    options = datum.avo().model_copy(update={"region": region})
    geophysics = config.geophysics.model_copy(
        update={"data": [datum.model_copy(update={"options": options})]}
    )
    wells = (
        "WELSPECS\n 'P1' 'G' 1 1 1* OIL /\n 'I1' 'G' 3 1 1* WATER /\n/\n"
        "COMPDAT\n 'P1' 2* 1 1 OPEN /\n 'P1' 2 1 1 1 OPEN /\n 'I1' 0 0 1 1 OPEN /\n/"
    )
    deck = _with_deck(config, tmp_path, wells).model_copy(update={"geophysics": geophysics})
    model = composition.build_geophysics(deck)
    assert model is not None
    # The completions (1, 1) (WELSPECS' head for 2*), (2, 1) and (3, 1) (the head for 0):
    # the columns between them.
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    columns = [tuple(int(v) for v in c) for c in survey.columns]
    assert columns == [(0, 0), (1, 0), (2, 0)]

    mako = _with_deck(config, tmp_path, "COMPDAT\n 'P1' ${i} 1 1 1 OPEN /\n/")
    with pytest.raises(ConfigError, match="cannot be read from the deck's text"):
        composition.build_geophysics(mako.model_copy(update={"geophysics": geophysics}))


def test_a_differences_baseline_needs_a_restart_too(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    datum = config.geophysics.data[0]
    options = datum.avo().model_copy(update={"four_d": "differences", "baseline": "2020-06-01"})
    geophysics = config.geophysics.model_copy(
        update={"data": [datum.model_copy(update={"options": options})]}
    )
    deck = _with_deck(config, tmp_path, "RPTRST\n BASIC=2 /\nDATES\n 1 JUL 2020 /\n/")
    with pytest.raises(ConfigError, match=r"2020-06-01 \(not a report time\)"):
        composition.geophysics_restart_warnings(deck.model_copy(update={"geophysics": geophysics}))


def _with_runspec(
    config: HistoryMatchConfig, tmp_path: Path, runspec: str, schedule: str, **update: object
) -> HistoryMatchConfig:
    """``config`` with a deck whose RUNSPEC and SCHEDULE are given (only its text is read)."""
    deck = tmp_path / "deck.mako"
    deck.write_text(
        f"RUNSPEC\n{runspec}\nGRID\nSOLUTION\nSUMMARY\nSCHEDULE\nSTART\n 1 JAN 2020 /\n"
        f"{schedule}\nDATES\n 1 JUL 2020 /\n/\nEND\n"
    )
    simulator = config.simulator.model_copy(
        update={"options": {"deck_template_file": str(deck)}, **update}
    )
    return config.model_copy(update={"simulator": simulator})


def test_what_every_member_would_fail_on_stops_the_run_before_it_starts(
    tmp_path: Path,
) -> None:
    """A regression (review of phase 2): these failed each member's read, after a whole
    ensemble had run."""
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    soft = config.model_copy(
        update={
            "geophysics": config.geophysics.model_copy(
                update={
                    "pem": PluginSection(name="soft_sand", options={"gas_model": "batzle_wang"})
                }
            )
        }
    )
    model = composition.build_geophysics(soft)
    assert model is not None
    nx, ny, nz = SHAPE
    rpt = "RPTRST\n BASIC=2 /"

    def check(runspec: str, schedule: str = rpt, **update: object) -> list[str]:
        deck = _with_runspec(soft, tmp_path, runspec, schedule, **update)
        return composition.geophysics_deck_warnings(deck, model)

    assert check(f"DIMENS\n {nx} {ny} {nz} /\nOIL\nWATER", "RPTRST\n BASIC=2 DEN /") == []
    with pytest.raises(ConfigError, match=r"grid is \(4, 3, 4\) \(DIMENS\)"):
        check(f"DIMENS\n {nx} {ny} {nz + 1} /")
    with pytest.raises(ConfigError, match="FMTOUT"):
        check("OIL\nWATER\nFMTOUT")
    with pytest.raises(ConfigError, match="extrapolate_gas_correlation"):
        check("OIL\nWATER\nGAS")  # CO2 at 37 C, outside Batzle & Wang's gas fit
    with pytest.raises(ConfigError, match="pem needs DEN in the restarts, but no RPTRST"):
        check("OIL\nWATER")
    (warning,) = check("OIL\nWATER", "RPTRST\n BASIC=2 ${extra} /")
    assert "cannot confirm" in warning
    eclipse = {"name": "eclipse"}
    with pytest.raises(ConfigError, match="keeps only the last restart"):
        check("OIL\nWATER", "RPTRST\n BASIC=1 DEN /", **eclipse)
    assert check("OIL\nWATER\nUNIFOUT", "RPTRST\n BASIC=1 DEN /", **eclipse) == []
    # A templated RUNSPEC tells nothing certain: warned, not refused.
    templated = check(
        "OIL\nWATER\n% if text:\nFMTOUT\n% endif", "RPTRST\n BASIC=1 DEN /", **eclipse
    )
    assert len(templated) == 3
    text = " | ".join(templated)
    assert "FMTOUT" in text
    assert "cannot confirm UNIFOUT" in text
    assert "CO2STORE's oil-gas form" in text  # soft_sand's CO₂: is the oil brine?


def test_a_stopped_geophysics_run_resumes_bit_identically(tmp_path: Path) -> None:
    whole = composition.start(_case(tmp_path / "whole"))
    composition.start(_case(tmp_path / "split"), max_steps=1)
    resumed = composition.resume(tmp_path / "split" / "runs" / "run")
    assert [h.data_mismatch_mean for h in resumed.history] == [
        h.data_mismatch_mean for h in whole.history
    ]
    last = f"iter_{len(whole.history) - 1}.npz"
    a = np.load(tmp_path / "whole" / "runs" / "run" / "checkpoints" / last)
    b = np.load(tmp_path / "split" / "runs" / "run" / "checkpoints" / last)
    for name in a.files:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_provenance_lists_the_geophysics_files(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    pressure = OverburdenPressureFile(file=tmp_path / "ob.npy", unit="bar")
    datum = config.geophysics.data[0]
    options = datum.avo().model_copy(
        update={
            "wavelet": FileWaveletSection(name="file", file=tmp_path / "w.npy", dt=0.001),
            "region": RegionFileSection(file=tmp_path / "r.npy"),
        }
    )
    geophysics = config.geophysics.model_copy(
        update={
            "overburden_pressure": pressure,
            "data": [datum.model_copy(update={"options": options})],
        }
    )
    files = composition._input_files(config.model_copy(update={"geophysics": geophysics}))
    assert files["geophysics.grid_file"] == tmp_path / "REF.EGRID"
    assert files["geophysics.overburden_pressure"] == tmp_path / "ob.npy"
    assert files["geophysics.avo.wavelet"] == tmp_path / "w.npy"
    assert files["geophysics.avo.region"] == tmp_path / "r.npy"


JAN = dt.date(2020, 1, 1)


class BoxPotential(BoxFlow):
    """A stand-in whose member's ``swat`` is SWAT at VINTAGE, which lowers the pressure."""

    def simulate(self, member: MemberInput) -> Path:
        assert member.work_dir is not None
        swat = np.clip(np.asarray(member.state["swat"]), 0.05, 0.95)
        fluids = {"OIL_DEN": 800.0, "WAT_DEN": 1020.0, "RPORV": 12500.0}
        restarts = {
            JAN: {"PRESSURE": 250.0, "SWAT": 0.2, **fluids},
            VINTAGE: {"PRESSURE": 250.0 - 40.0 * swat, "SWAT": swat, **fluids},
        }
        return _cases.write_case(member.work_dir / "MODEL", SHAPE, restarts, porosity=0.25)


POTENTIAL = """
[geophysics]
grid_file = "REF.EGRID"

[[geophysics.data]]
kind = "gravity"
datatype = "grav"
vintages = ["2020-07-01"]
options = { baseline = "2020-01-01", stations = { spacing = 100.0, depth = 100.0 } }

[[geophysics.data]]
kind = "seafloor_displacement"
datatype = "subsidence"
vintages = ["2020-07-01"]
options = { baseline = "2020-01-01", compressibility = 1e-8, kernel = "geertsma", \
stations = { spacing = 100.0, depth = 100.0 } }
"""


def test_gravity_and_displacement_are_matched(tmp_path: Path) -> None:
    simulators.register("test_box_potential")(BoxPotential)
    try:
        _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
        truth = BoxPotential().simulate(MemberInput(0, {"swat": TRUTH}, tmp_path))
        text = _config(mec=False, geophysics=False).replace("test_box_flow", "test_box_potential")
        raw = _raw(text + POTENTIAL.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
        model = composition.build_geophysics(HistoryMatchConfig.model_validate(raw))
        assert model is not None
        assert model.field_sizes() == {"grav": 6, "subsidence": 6}
        rng = np.random.default_rng(5)
        fields = ""
        for name in ("grav", "subsidence"):
            keys = [DataKey(VINTAGE.isoformat(), name, i) for i in range(6)]
            values = model.forecast(truth, keys)
            assert np.all(values > 0)  # water replaces oil (heavier); the pressure drops
            noise = 0.01 * np.abs(values).mean()
            np.save(tmp_path / f"{name}.npy", values + noise * rng.standard_normal(6))
            fields += (
                f'\n[[observations.fields]]\ntime = "{VINTAGE.isoformat()}"\n'
                f'datatype = "{name}"\nfile = "{name}.npy"\nvariance = {float(noise**2)!r}\n'
            )
        (tmp_path / "observations.csv").write_text(
            f"time,datatype,value,variance\n{VINTAGE.isoformat()},SWMEAN,"
            f"{float(TRUTH.mean())!r},1.0\n"
        )
        np.savez(tmp_path / "truth.npz", swat=TRUTH)
        path = tmp_path / "config.toml"
        path.write_text(text.replace("#FIELDS", fields) + POTENTIAL)
        config = composition.load_config(path)
        assert config.geophysics is not None
        files = composition._input_files(config)
        assert files["geophysics.grid_file"] == tmp_path / "REF.EGRID"
        from apex.services.access import Access, UIAccess
        from apex.services.validation import validate_history_match

        validation = validate_history_match(path)
        assert validation.valid, validation.errors
        rules = UIAccess.model_validate({
            "plugins": {"methods": ["esmda"], "simulators": ["test_box_potential"],
                        "executors": ["serial"], "priors": ["gaussian"]},
        })  # fmt: skip
        assert Access(rules, [], "alice").hidden_plugins(config) == [
            "displacement_kernels:geertsma"
        ]
        result = composition.start(path)
    finally:
        simulators._factories.pop("test_box_potential")
    first, last = result.history[0], result.history[-1]
    assert last.data_mismatch_mean < 0.2 * first.data_mismatch_mean


def test_the_deck_must_write_what_gravity_reads(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    raw = _raw(
        _config(mec=False, geophysics=False)
        + POTENTIAL.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"')
    )
    gravity = HistoryMatchConfig.model_validate(raw)
    model = composition.build_geophysics(gravity)
    assert model is not None
    assert model.restart_needs() == {"DEN": "grav", "RPORV": "grav"}
    nx, ny, nz = SHAPE
    runspec = f"DIMENS\n {nx} {ny} {nz} /\nOIL\nWATER"

    def check(schedule: str) -> list[str]:
        deck = _with_runspec(config, tmp_path, runspec, schedule)
        deck = deck.model_copy(update={"geophysics": gravity.geophysics})
        return composition.geophysics_deck_warnings(deck, model)

    assert check("RPTRST\n BASIC=2 DEN RPORV /") == []
    with pytest.raises(ConfigError, match="grav needs RPORV in the restarts"):
        check("RPTRST\n BASIC=2 DEN /")
    with pytest.raises(ConfigError, match="grav needs DEN in the restarts"):
        check("RPTRST\n BASIC=2 RPORV /")


def test_co2s_temperature_must_be_the_decks(tmp_path: Path) -> None:
    """With CO₂'s equation of state, a temperature_c the deck's RTEMP contradicts is
    refused before the run; a deck without RTEMP is warned of."""
    pytest.importorskip("CoolProp")
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    soft = config.model_copy(
        update={
            "geophysics": config.geophysics.model_copy(
                update={"pem": PluginSection(name="soft_sand", options={"temperature_c": 37.0})}
            )
        }
    )
    model = composition.build_geophysics(soft)
    assert model is not None
    nx, ny, nz = SHAPE
    runspec = f"DIMENS\n {nx} {ny} {nz} /\nWATER\nGAS"

    def check(schedule: str) -> list[str]:
        deck = _with_runspec(soft, tmp_path, runspec, schedule)
        return composition.geophysics_deck_warnings(deck, model)

    assert check("RTEMP\n 37 /\nRPTRST\n BASIC=2 DEN /") == []
    with pytest.raises(ConfigError, match="reservoir temperature is 40"):
        check("RTEMP\n 40 /\nRPTRST\n BASIC=2 DEN /")
    with pytest.raises(ConfigError, match="varies the temperature"):
        check("RTEMPVD\n 1000 30\n 2000 60 /\nRPTRST\n BASIC=2 DEN /")
    (warning,) = check("RPTRST\n BASIC=2 DEN /")
    assert "states no readable reservoir temperature" in warning
    # CO2STORE without RTEMP runs at OPM Flow's default, 60 °F (measured): checked too
    # (review of phase 2b: OPM's CO₂ at 15.56 °C, the PEM's at 37 °C).
    runspec = f"DIMENS\n {nx} {ny} {nz} /\nWATER\nGAS\nCO2STORE"
    with pytest.raises(ConfigError, match=r"reservoir temperature is 15\.56"):
        check("RPTRST\n BASIC=2 DEN /")


def test_a_co2store_deck_is_read_in_its_form(tmp_path: Path) -> None:
    """The oil-gas form's oil is brine; Eclipse 300's CO2STORE is refused."""
    config = composition.load_config(_case(tmp_path))
    nx, ny, nz = SHAPE
    rpt = "RPTRST\n BASIC=2 DEN /"
    for runspec, brine in (("OIL\nGAS\nCO2STORE", True), ("WATER\nGAS\nCO2STORE", False),
                           ("OIL\nGAS", False)):  # fmt: skip
        deck = _with_runspec(config, tmp_path, f"DIMENS\n {nx} {ny} {nz} /\n{runspec}", rpt)
        model = composition.build_geophysics(deck)
        assert model is not None
        assert model.brine_as_oil is brine, runspec
    eclipse = _with_runspec(
        config, tmp_path, f"DIMENS\n {nx} {ny} {nz} /\nWATER\nGAS\nCO2STORE", rpt,
        name="eclipse",
    )  # fmt: skip
    model = composition.build_geophysics(eclipse)
    assert model is not None
    with pytest.raises(ConfigError, match="Eclipse 300's CO2STORE"):
        composition.geophysics_deck_warnings(eclipse, model)


def test_the_mixing_law_is_configured_through_the_pem_options(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path))
    assert config.geophysics is not None
    options = {"mixing": "patchiness", "patchiness": 0.3, "gas_model": "batzle_wang"}
    patchy = config.model_copy(
        update={
            "geophysics": config.geophysics.model_copy(
                update={"pem": PluginSection(name="soft_sand", options=options)}
            )
        }
    )
    model = composition.build_geophysics(patchy)
    assert model is not None
    assert (model.pem.mixing, model.pem.patchiness) == ("patchiness", 0.3)  # type: ignore[union-attr]
    bad = patchy.model_copy(
        update={
            "geophysics": config.geophysics.model_copy(
                update={
                    "pem": PluginSection(name="soft_sand", options={**options, "patchiness": 2.0})
                }
            )
        }
    )
    with pytest.raises(ConfigError, match="patchiness must be in"):
        composition.build_geophysics(bad)


class BoxDense(BoxFlow):
    """A stand-in whose restarts hold the phases' densities, for soft_sand."""

    def simulate(self, member: MemberInput) -> Path:
        assert member.work_dir is not None
        assert set(member.state) == {"swat"}  # the petro-elastic parameter stays out
        swat = np.clip(np.asarray(member.state["swat"]), 0.05, 0.95)
        restarts = {VINTAGE: {"PRESSURE": 250.0, "SWAT": swat, "OIL_DEN": 800.0,
                              "WAT_DEN": 1030.0}}  # fmt: skip
        return _cases.write_case(member.work_dir / "MODEL", SHAPE, restarts, porosity=0.25)


UNCERTAIN = """
[[state.variables]]
name = "pem_q"
size = 1

[prior.variables.pem_q]
name = "gaussian"
options = { mean = 0.0, std = 1.5 }
"""


def test_patchiness_is_updated_with_the_reservoir_model(tmp_path: Path) -> None:
    """soft_sand's patchiness from the state variable pem_q (logit), auxiliary
    (no true value: the truth takes the fallback 0.85), updated by ES-MDA."""
    simulators.register("test_box_dense")(BoxDense)
    try:
        _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
        truth = BoxDense().simulate(MemberInput(0, {"swat": TRUTH}, tmp_path))
        pem = (
            'pem = { name = "soft_sand", options = { gas_model = "batzle_wang", '
            'mixing = "patchiness", patchiness = { state = "pem_q", transform = "logit", '
            "value = 0.85 } } }"
        )
        geophysics = GEOPHYSICS.replace('pem = { name = "standard" }', pem)
        text = _config(mec=False, geophysics=False).replace("test_box_flow", "test_box_dense")
        text = text.replace("[prior]", UNCERTAIN + "\n[prior]") + geophysics
        raw = _raw(text.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
        model = composition.build_geophysics(HistoryMatchConfig.model_validate(raw))
        assert model is not None
        assert model.state_variables == ("pem_q",)
        rng = np.random.default_rng(5)
        fields = ""
        for angle in ("10", "25"):
            n = model.field_sizes()[f"avo_{angle}"]
            keys = [DataKey(VINTAGE.isoformat(), f"avo_{angle}", i) for i in range(n)]
            values = model.for_state({}).forecast(truth, keys)
            noise = 0.01 * float(np.abs(values).mean())
            np.save(tmp_path / f"avo_{angle}.npy", values + noise * rng.standard_normal(n))
            fields += (
                f'\n[[observations.fields]]\ntime = "{VINTAGE.isoformat()}"\n'
                f'datatype = "avo_{angle}"\nfile = "avo_{angle}.npy"\n'
                f"variance = {noise**2!r}\n"
            )
        (tmp_path / "observations.csv").write_text(
            f"time,datatype,value,variance\n{VINTAGE.isoformat()},SWMEAN,"
            f"{float(TRUTH.mean())!r},2.5e-5\n"
        )
        np.savez(tmp_path / "truth.npz", swat=TRUTH)  # no pem_q: it has no true value
        path = tmp_path / "config.toml"
        path.write_text(text.replace("#FIELDS", fields))
        run = composition.build_run(composition.load_config(path))
        assert "pem_q" in run.workflow.auxiliary_variables
        result = composition.start(path)
    finally:
        simulators._factories.pop("test_box_dense")
    first, last = result.history[0], result.history[-1]
    assert last.data_mismatch_mean < first.data_mismatch_mean
    checkpoints = tmp_path / "runs" / "run" / "checkpoints"
    prior = np.load(checkpoints / "iter_0.npz")
    posterior = np.load(checkpoints / f"iter_{len(result.history) - 1}.npz")
    n_swat = N_CELLS  # the state is swat then pem_q
    assert prior["state"].shape[0] == n_swat + 1
    assert not np.allclose(prior["state"][n_swat], posterior["state"][n_swat])  # updated


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ('{ state = "pem_q", transform = "probit" }', "a state reference is"),
        ('{ state = "pem_q", bound = 1 }', "a state reference is"),
        # An update could push q out of [0, 1]: only the logistic keeps it there.
        ('{ state = "pem_q", transform = "none" }', "can leave patchiness's valid range"),
        ('{ state = "pem_q", transform = "log" }', "can leave patchiness's valid range"),
        ('{ state = "pem_q", value = true }', "value must be a number"),
        ('{ state = "nope" }', "'nope' must be declared, of size 1"),
        ('{ state = "swat" }', "'swat' must be declared, of size 1"),
    ],
)
def test_a_state_reference_must_name_a_scalar_variable(
    tmp_path: Path, options: str, message: str
) -> None:
    pem = f'pem = {{ name = "soft_sand", options = {{ patchiness = {options} }} }}'
    text = _config(mec=False, geophysics=False).replace("[prior]", UNCERTAIN + "\n[prior]")
    text += GEOPHYSICS.replace('pem = { name = "standard" }', pem)
    _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
    raw = _raw(text.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
    with pytest.raises(ConfigError, match=message):
        composition.build_geophysics(HistoryMatchConfig.model_validate(raw))


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("temperature_c", '{ state = "pem_q" }', "a state variable may set only"),
        ("brie_shape", '{ state = "pem_q" }', "a state variable may set only"),
        ("patchiness", '{ state = "pem_q", transform = "logit", value = 2.0 }',
         "patchiness must be in"),
    ],
)  # fmt: skip
def test_only_the_options_a_model_declares_may_be_uncertain(
    tmp_path: Path, option: str, value: str, message: str
) -> None:
    """A regression (review of phase 2b, step 4): an option the setup depends on (the
    temperature, CO₂'s table) cannot change per member; the fallback is checked at setup."""
    pem = f'pem = {{ name = "soft_sand", options = {{ {option} = {value} }} }}'
    text = _config(mec=False, geophysics=False).replace("[prior]", UNCERTAIN + "\n[prior]")
    text += GEOPHYSICS.replace('pem = { name = "standard" }', pem)
    _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
    raw = _raw(text.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
    with pytest.raises(ConfigError, match=message):
        composition.build_geophysics(HistoryMatchConfig.model_validate(raw))


def test_a_table_option_without_state_is_an_ordinary_option(tmp_path: Path) -> None:
    """A regression (review of phase 2b): only a table with ``state`` is a state reference,
    so standard's fluid_fits table still reaches the model and the option checks."""
    from apex.services.plugins import geophysics_sections

    fits = "fluid_fits = { water = [1067.3, 0.3, 2807.6, 9.0] }"
    pem = f'pem = {{ name = "standard", options = {{ {fits} }} }}'
    text = _config(mec=False, geophysics=False) + GEOPHYSICS.replace(
        'pem = { name = "standard" }', pem
    )
    _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
    raw = _raw(text.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
    config = HistoryMatchConfig.model_validate(raw)
    options, parameters = composition._pem_parameters(config)
    assert parameters == ()
    assert options["fluid_fits"] == {"water": [1067.3, 0.3, 2807.6, 9.0]}
    assert composition.build_geophysics(config) is not None
    (pem_section,) = [s for r, s in geophysics_sections(config) if r == "petro_elastic_models"]
    assert "fluid_fits" in pem_section.options


def test_a_templated_runspec_leaves_the_co2store_form_unknown(tmp_path: Path) -> None:
    """``brine_as_oil`` is ``None`` behind a templated RUNSPEC (the form cannot be read),
    so an oil-gas run with a CO₂ model is refused when read, not misread (review of
    phase 2b)."""
    config = composition.load_config(_case(tmp_path))
    nx, ny, nz = SHAPE
    rpt = "RPTRST\n BASIC=2 DEN /"
    literal = _with_runspec(config, tmp_path, f"DIMENS\n {nx} {ny} {nz} /\nOIL\nGAS\nCO2STORE", rpt)
    assert composition._co2store_oil_gas(literal) is True
    templated = _with_runspec(
        config, tmp_path, f"DIMENS\n {nx} {ny} {nz} /\n% if True:\nOIL\n% endif\nGAS\nCO2STORE", rpt
    )
    assert composition._co2store_oil_gas(templated) is None


def test_a_misspelt_model_with_a_state_reference_is_an_unknown_plugin(tmp_path: Path) -> None:
    """A regression (the coder's review of phase 2b): it was reported as a model with no
    option a state variable may set."""
    pem = 'pem = { name = "soft_sandd", options = { patchiness = { state = "pem_q" } } }'
    text = _config(mec=False, geophysics=False).replace("[prior]", UNCERTAIN + "\n[prior]")
    text += GEOPHYSICS.replace('pem = { name = "standard" }', pem)
    _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
    raw = _raw(text.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
    with pytest.raises(ConfigError) as caught:
        composition.build_geophysics(HistoryMatchConfig.model_validate(raw))
    assert "soft_sandd" in str(caught.value)
    assert "state variable may set" not in str(caught.value)


def test_a_wavelet_transform_of_avo_maps_takes_the_traces_lattice(tmp_path: Path) -> None:
    """[[data_transforms]] wavelet of an AVO field needs no grid or mask
    (the traces' lattice and kept columns); its compressed observation is the observed map's
    own coefficients, as a forecast equal to it gives."""
    from apex.algorithms.data_compression.transform import WaveletCompression
    from apex.storage.inputs import read_observation_section

    path = _case(tmp_path)
    wavelet = (
        '\n[[data_transforms]]\nname = "wavelet"\n'
        'options = { datatype = "avo_10", wavelet = "haar", level = 1 }\n'
    )
    path.write_text(path.read_text() + wavelet)
    config = composition.load_config(path)
    # The coder's review of phase 4: validation (and so every submission) accepts it.
    from apex.services.validation import validate_history_match

    validation = validate_history_match(path)
    assert validation.valid, validation.errors
    (transform,) = composition.build_transforms(config)
    assert isinstance(transform, WaveletCompression)
    assert transform.options.grid == (4, 3)  # the 4 x 3 columns, every one traced
    observations = read_observation_section(config.observations)
    compressed = transform.fit(observations)
    assert any(k.datatype == "wavelet(avo_10)" for k in compressed.keys)
    observed = np.asarray(observations.values, dtype=float)[:, np.newaxis]
    applied = transform.apply(observed)[:, 0]
    np.testing.assert_allclose(applied, compressed.values, rtol=1e-12, atol=1e-15)
    # The whole loop on the compressed map: the data mismatch falls.
    result = composition.start(path)
    assert result.history[-1].data_mismatch_mean < result.history[0].data_mismatch_mean
    # A partial layout is refused; a layout given in full is used as it is.
    partial = path.read_text().replace("level = 1 }", 'level = 1, order = "F" }')
    path.write_text(partial)
    with pytest.raises(ConfigError, match="give its layout in full"):
        composition.build_transforms(composition.load_config(path))
    path.write_text(partial.replace(', order = "F" }', " }"))
    path.write_text(path.read_text().replace("level = 1 }", "level = 1, grid = [4, 3, 1] }"))
    (given,) = composition.build_transforms(composition.load_config(path))
    assert isinstance(given, WaveletCompression)
    assert given.options.grid == (4, 3, 1)


@pytest.mark.parametrize(
    ("model", "option", "expected"),
    [
        ("soft_sand", "mineral_bulk_modulus", 1.0),  # the exponential of 0
        ("soft_sand", "mineral_shear_modulus", 1.0),
        ("soft_sand", "mineral_density", 1.0),
    ],
)
def test_more_rock_physics_parameters_may_be_uncertain(
    tmp_path: Path, model: str, option: str, expected: float
) -> None:
    """soft_sand's mineral moduli and density, through the
    exponential (any positive moduli are valid), set per member."""
    pem = f'pem = {{ name = "{model}", options = {{ {option} = {{ state = "pem_q" }} }} }}'
    text = _config(mec=False, geophysics=False).replace("[prior]", UNCERTAIN + "\n[prior]")
    text += GEOPHYSICS.replace('pem = { name = "standard" }', pem)
    _cases.write_grid(tmp_path / "REF.EGRID", SHAPE)
    raw = _raw(text.replace('"REF.EGRID"', f'"{tmp_path / "REF.EGRID"}"'))
    built = composition.build_geophysics(HistoryMatchConfig.model_validate(raw))
    assert built is not None
    assert built.state_variables == ("pem_q",)
    assert getattr(built.for_state({"pem_q": [0.0]}).pem, option) == pytest.approx(expected)


def test_a_sparse_lattice_is_refused_for_the_default_wavelet_layout(tmp_path: Path) -> None:
    """The review of phase 4, step C: traces covering under half their (cropped) lattice
    leave mostly filled cells, whose zero coefficients collapse the noise estimate."""
    path = _case(tmp_path)
    region = np.zeros(SHAPE[:2])
    region[0, 0] = region[-1, -1] = 1  # two corners: the whole lattice, 2 of 12 traced
    np.save(tmp_path / "region.npy", region)
    for angle in ("10", "25"):  # observations of the two traced columns only
        np.save(tmp_path / f"avo_{angle}.npy", np.load(tmp_path / f"avo_{angle}.npy")[:2])
    text = path.read_text().replace(
        "[geophysics.data.options]", '[geophysics.data.options]\nregion = { file = "region.npy" }'
    )
    wavelet = (
        '\n[[data_transforms]]\nname = "wavelet"\n'
        'options = { datatype = "avo_10", wavelet = "haar", level = 1 }\n'
    )
    path.write_text(text + wavelet)
    with pytest.raises(ConfigError, match="cover 17 % of their 4 x 3 lattice"):
        composition.build_transforms(composition.load_config(path))
    from apex.services.validation import validate_history_match

    validation = validate_history_match(path)  # said at validate time too
    assert not validation.valid
    assert any("cover 17 %" in e for e in validation.errors)


def test_a_runs_avo_maps_are_drawn(tmp_path: Path) -> None:
    """The stored run's view places each map value at its column, and
    field_maps draws the observed, prior, posterior and mismatch maps of every angle."""
    pytest.importorskip("matplotlib")
    import apex.reporting.figures  # noqa: F401  (registers the plugins)
    from apex.core.plugins import figures
    from apex.services.figure_view import run_view

    composition.start(_case(tmp_path))
    view = run_view(tmp_path / "runs" / "run")
    layout = view.field_layout("avo_10")
    assert layout is not None
    assert layout.kind == "columns"
    assert layout.lateral == SHAPE[:2]
    assert layout.positions.shape == (SHAPE[0] * SHAPE[1], 2)
    assert view.field_layout("SWMEAN") is None
    figure = figures.create("field_maps")
    assert figure.applies(view) is None
    written = figure.draw(view, tmp_path / "maps")
    assert sorted(p.stem for p in written) == sorted(
        f"avo_{a}_{VINTAGE.isoformat()}_{kind}"
        for a in ("10", "25")
        for kind in ("obs", "prior", "post", "mismatch")
    )


def _soft(config: HistoryMatchConfig, **options: object) -> HistoryMatchConfig:
    assert config.geophysics is not None
    return config.model_copy(
        update={
            "geophysics": config.geophysics.model_copy(
                update={"pem": PluginSection(name="soft_sand", options=dict(options))}
            )
        }
    )


def test_the_decks_temperature_is_resolved_per_cell(tmp_path: Path) -> None:
    """Temperature = "deck" takes one temperature as temperature_c
    (the scalar path, as before), depth tables per cell with their range, a thermal run
    per cell with a temperature_range of its own; what the text cannot tell is refused."""
    pytest.importorskip("CoolProp")
    config = composition.load_config(_case(tmp_path))
    nx, ny, nz = SHAPE
    runspec = f"DIMENS\n {nx} {ny} {nz} /\nWATER\nGAS\nMETRIC"
    rpt = "RPTRST\n BASIC=2 DEN /"

    def build(schedule: str, extra: str = "", **options: object) -> GeophysicsModel | None:
        soft = _soft(config, temperature="deck", **options)
        return composition.build_geophysics(
            _with_runspec(soft, tmp_path, runspec + extra, f"{schedule}\n{rpt}")
        )

    constant = build("RTEMP\n 41.5 /")
    # Without gas no CO₂ table is needed: a thermal run then takes no range.
    runspec_oil = f"DIMENS\n {nx} {ny} {nz} /\nOIL\nWATER\nMETRIC\nTHERMAL"
    oil = _with_runspec(_soft(config, temperature="deck"), tmp_path, runspec_oil, rpt)
    oil_model = composition.build_geophysics(oil)
    assert oil_model is not None
    assert oil_model.temperature is not None
    assert constant is not None
    assert constant.temperature is None
    assert (constant.pem.temperature_c, constant.pem.temperature) == (41.5, None)  # type: ignore[union-attr]
    # The default range: the tables' temperatures over grid_file's cells, tops to bases.
    assert config.geophysics is not None
    grid = Grid.read(config.geophysics.grid_file)
    shallow, deep = float(grid.top[grid.actnum].min()), float(grid.bottom[grid.actnum].max())
    gradient = f"RTEMPVD\n {shallow - 100:g} 35\n {deep + 100:g} 65 /"
    tables = build(gradient)
    assert tables is not None
    assert tables.temperature is not None
    assert tables.temperature.tables == (((shallow - 100, 35.0), (deep + 100, 65.0)),)
    expected = np.interp([shallow, deep], [shallow - 100, deep + 100], [35.0, 65.0])
    assert tables.pem.temperature_range == pytest.approx(tuple(expected))  # type: ignore[union-attr]
    # A long gradient (from the surface, beyond 100 °C deep down) takes only the cells' part.
    long = build(f"RTEMPVD\n 0 10\n {4 * deep:g} 160 /")
    assert long is not None
    assert long.pem.temperature_range[1] < 100.0  # type: ignore[union-attr]
    with pytest.raises(ConfigError, match="give temperature_range"):
        build(f"RTEMPVD\n {shallow:g} 120\n {deep:g} 130 /")
    # One table: a given range must cover the cells; a covering one is taken.
    with pytest.raises(ConfigError, match="misses the deck's"):
        build(gradient, temperature_range=[expected[0] + 1.0, 65.0])
    wider = build(gradient, temperature_range=[30.0, 70.0])
    assert wider is not None
    assert wider.pem.temperature_range == (30.0, 70.0)  # type: ignore[union-attr]
    with pytest.raises(ConfigError, match=r"must be \[low, high\]"):
        build("RPTRST\n BASIC=2 /", "\nTHERMAL", temperature_range=[20.0])
    thermal = build("RPTRST\n BASIC=2 /", "\nTHERMAL", temperature_range=[20.0, 80.0])
    assert thermal is not None
    assert thermal.temperature is not None
    assert thermal.temperature.thermal
    # CO₂'s table is built at setup: a thermal run with gas needs a range then, not later.
    with pytest.raises(ConfigError, match="needs temperature_range"):
        build("RPTRST\n BASIC=2 /", "\nTHERMAL")
    with pytest.raises(ConfigError, match="outside temperature_range"):
        build("RTEMP\n 41.5 /", temperature_range=[20.0, 40.0])
    with pytest.raises(ConfigError, match="cannot be read from its text"):
        build("RTEMPVD\n 1000 ${t}\n 2000 65 /")
    with pytest.raises(ConfigError, match="temperature_c or 'deck', not both"):
        build("RTEMP\n 41.5 /", temperature_c=40.0)
    # The varying table is no longer refused under "deck".
    soft = _soft(config, temperature="deck")
    deck = _with_runspec(soft, tmp_path, runspec, f"RTEMPVD\n 1000 35\n 2000 65 /\n{rpt}")
    model = composition.build_geophysics(deck)
    assert model is not None
    assert composition.geophysics_deck_warnings(deck, model) == []


class _Case:
    """A stand-in for CaseState: depths, INIT arrays and one restart."""

    def __init__(self, depth: list[float], units: str = "METRIC", **arrays: list[float]) -> None:
        self.case = Path("MEMBER")
        self.depth = np.array(depth)
        self.units = type("U", (), {"name": units})()
        self.init = {k: np.array(v) for k, v in arrays.items() if k == "EQLNUM"}
        self._restart = {k: np.array(v) for k, v in arrays.items() if k != "EQLNUM"}

    def static(self, name: str) -> np.ndarray:
        return self.init[name]

    def restart(self, date: object, names: list[str]) -> dict[str, np.ndarray]:
        return {n: self._restart[n] for n in names if n in self._restart}


def test_each_cell_takes_its_regions_table_at_its_depth_or_its_restart_temp() -> None:
    from apex.workflows.geophysics import CellTemperature

    date = dt.date(2025, 1, 1)
    one = CellTemperature(tables=(((1000.0, 30.0), (2000.0, 60.0)),))
    cells = one.cells(_Case([500.0, 1500.0, 2500.0]), date)  # type: ignore[arg-type]
    np.testing.assert_allclose(cells, [30.0, 45.0, 60.0])  # the ends beyond the table
    two = CellTemperature(tables=(((0.0, 30.0), (1000.0, 40.0)), ((0.0, 50.0), (1000.0, 70.0))))
    cells = two.cells(_Case([500.0, 500.0], EQLNUM=[1, 2]), date)  # type: ignore[arg-type]
    np.testing.assert_allclose(cells, [35.0, 60.0])
    # Without EQLNUM in INIT every cell is in region 1, as a defaulted EQLNUM is.
    np.testing.assert_allclose(two.cells(_Case([500.0]), date), [35.0])  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="EQLNUM 3 has no temperature table"):
        two.cells(_Case([500.0], EQLNUM=[3]), date)  # type: ignore[arg-type]
    thermal = CellTemperature(thermal=True)
    field = thermal.cells(_Case([0.0, 0.0], "FIELD", TEMP=[104.0, 140.0]), date)  # type: ignore[arg-type]
    np.testing.assert_allclose(field, [40.0, 60.0])
    with pytest.raises(ConfigError, match="has no TEMP"):
        thermal.cells(_Case([0.0]), date)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one of them"):
        CellTemperature()
