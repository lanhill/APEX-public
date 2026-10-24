"""Typed configuration of a history-matching run (the native APEX TOML schema).

Example::

    [run]
    name = "linear_demo"
    seed = 2026

    [[state.variables]]
    name = "x"
    size = 3

    [prior]
    ensemble_size = 100
    generator = { name = "gaussian", options = { mean = 0.0, std = 1.0 } }

    [prior.variables.x]          # optional per-variable generator
    name = "gaussian_field"
    options = { grid = [3, 1, 1], major_range = 2.0, variogram = "spherical" }

    [observations]
    file = "observations.csv"

    [[observations.fields]]       # optional field data, e.g. a seismic vintage
    time = "2024-01-01"
    datatype = "avo"
    file = "avo_2024.npz"
    variance = 0.01

    [simulator]
    name = "linear"
    options = { forward_file = "forward.npz" }

    [execution]
    name = "process"
    options = { workers = 8 }

    [method]
    name = "esmda"
    options = { inflation = [4, 4, 4, 4] }
    strategy = { name = "approx", options = { energy = 0.98 } }

    [case]                        # synthetic (default; needs the true model) or real
    kind = "synthetic"
    truth_file = "truth.npz"

    [[metrics]]                   # optional; default normalized_data_mismatch (+ rmse if synthetic)
    name = "ssim"
    options = { grid = [10, 10, 2], per_layer = true, win_size = 3 }

    [[data_transforms]]           # optional; e.g. wavelet compression of the avo fields
    name = "wavelet"
    options = { datatype = "avo", grid = [16, 12, 6], wavelet = "db2", level = 2 }

    [geophysics]                  # optional; AVO data from a deck simulator's runs
    pem = { name = "standard" }
    grid_file = "truth/MODEL.EGRID"
    # Overburden (lithostatic) pressure per cell, in Pa: p_ob = surface + gradient * depth,
    # with depth the cell centre's true vertical depth (m, positive down, the run's INIT DEPTH).
    # Example: rock above the reservoir with an average density of 2300 kg/m³ gives
    #   gradient = 2300 kg/m³ * 9.81 m/s² ≈ 22,600 Pa/m  (≈ 1 psi/ft),
    # so a cell at 2000 m has p_ob ≈ 101,325 + 22,600 * 2000 ≈ 45.3 MPa.
    overburden_pressure = { gradient = 22600.0, surface = 101325.0 }   # Pa/m, Pa
    # or one value per active cell from a file, as in the reference:
    # overburden_pressure = { file = "overburden.npz", key = "overburden" }

    [[geophysics.data]]           # fields avo_10 and avo_20 at each vintage
    kind = "avo"
    datatype = "avo"
    vintages = ["2025-06-01", "2030-01-01"]

    [geophysics.data.options]
    angles = [10, 20]
    wavelet = { frequency = 45, length = 0.05 }               # Ricker
    overburden = { vp = 2450, vs = 849, density = 2300 }      # m/s, m/s, kg/m3
    times = { start = 1.9, end = 2.2, step = 0.001 }          # s, two-way

Relative file paths are resolved against the configuration file's directory.
"""

from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema


class _Strict(BaseModel):
    # Unknown keys are errors: a misspelt option must not be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSection(_Strict):
    """``[run]``: identity and reproducibility."""

    name: str = Field(min_length=1)
    seed: int = Field(ge=0, description="Root seed for all random streams.")
    output_dir: Path | None = Field(
        default=None, description="Run directory; default `runs/<name>` next to the config."
    )

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        # The name places the run (runs/<name>, the web UI's UI/<user>/runs/<name>): a
        # separator or "." / ".." would put it elsewhere.
        if "/" in value or "\\" in value or value in (".", ".."):
            raise ValueError(
                f"run name {value!r}: one folder name, no '/' or '\\', not '.' or '..'"
            )
        return value


class StateVariableSection(_Strict):
    """``[[state.variables]]``: one state block."""

    name: str = Field(min_length=1)
    size: int = Field(ge=1)
    # None means unbounded; kept as None (not +-inf) so the JSON snapshot round-trips.
    lower: float | None = None
    upper: float | None = None
    grid: list[int] | None = Field(
        default=None,
        description="Field shape [nx, ny, nz] (Eclipse cell order) when the variable is a grid "
        "field; used by SSIM and parameter maps.",
    )


class StateSection(_Strict):
    """``[state]``."""

    variables: list[StateVariableSection] = Field(min_length=1)


class PluginSection(_Strict):
    """A registry name plus keyword options for its factory.

    ``diagnostics``, for plugins that produce diagnostics (the method, its
    strategy and localization), switches this plugin's items on or off and **overrides**
    ``[output] diagnostics`` for it: ``{ taper = true }`` saves the taper even if
    ``[output]`` does not list it, ``{ taper = false }`` suppresses it even if it does.
    ``max_mb`` and ``every`` override ``[output] diagnostics_max_mb`` and
    ``diagnostics_every`` for this plugin. The run logs the resulting plan and warns when
    a plugin switch contradicts ``[output]``.
    """

    name: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, bool | float] | None = Field(
        default=None,
        description="Per-plugin diagnostics switches (name = true/false, max_mb, every); "
        "they override [output] diagnostics for this plugin.",
    )


def _bare_name(data: object) -> object:
    # A plugin with no options is written as its name: the reference offered
    # ``stopping = ... or "select"``, which was refused (the release tester, 2026-10-19).
    return {"name": data} if isinstance(data, str) else data


class _NameOrTable:
    """JSON Schema of a :data:`NamedPlugin` field: a plugin's name, or its table."""

    def __get_pydantic_json_schema__(
        self, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        table = handler(core_schema)
        name = {"type": "string", "minLength": 1, "description": "The plugin's name alone."}
        return {"anyOf": [name, table]}


#: A plugin section that may also be written as its name alone (``stopping = "select"`` is
#: ``stopping = { name = "select" }``). Only for the auxiliary parts the documentation
#: writes so (strategy, localization, damping, stopping rule, prior generators, the
#: geophysics models, model error): not ``[method]``, ``[simulator]``, ``[execution]`` or an
#: optimizer, whose tables readers (the web UI's assistant, studies' raw-TOML patches)
#: expect to be tables.
NamedPlugin = Annotated[PluginSection, BeforeValidator(_bare_name), _NameOrTable()]


class PriorGroupSection(PluginSection):
    """``[prior.groups.<name>]``: a joint generator and the state variables it draws."""

    variables: list[str] = Field(min_length=1)


class PriorSection(_Strict):
    """``[prior]``: how the initial ensemble is generated.

    ``generator`` applies to every state variable unless ``[prior.variables.<name>]``
    overrides it. Generators are plugins: ``gaussian`` (independent normals), ``file``
    (stored ``.npz``), ``gaussian_field`` (variogram + Cholesky, PET's sampler) and
    ``geostatspy_sgsim`` (GeostatsPy). Realisations are clipped to the state bounds.
    """

    ensemble_size: int | None = Field(
        default=None, ge=2, description="Members; optional when every variable is read from a file."
    )
    generator: NamedPlugin = Field(default_factory=lambda: PluginSection(name="gaussian"))
    variables: dict[str, NamedPlugin] = Field(default_factory=dict)
    groups: dict[str, PriorGroupSection] = Field(
        default_factory=dict,
        description="Variables drawn together by one joint generator, e.g. the "
        "latent fields of a plurigaussian simulation (`gstools_pgs`).",
    )

    def generator_for(self, variable: str) -> PluginSection:
        """Return the generator section that applies to ``variable``."""
        return self.variables.get(variable, self.generator)

    @model_validator(mode="after")
    def _groups_are_disjoint(self) -> "PriorSection":
        seen: dict[str, str] = {}
        for group, section in self.groups.items():
            for name in section.variables:
                if name in seen:
                    raise ValueError(
                        f"state variable {name!r} is in prior groups {seen[name]!r} and {group!r}"
                    )
                if name in self.variables:
                    raise ValueError(
                        f"state variable {name!r} has a [prior.variables] generator and is in "
                        f"prior group {group!r}"
                    )
                seen[name] = group
        return self


class FieldObservationSection(_Strict):
    """``[[observations.fields]]``: one field datum (e.g. a seismic vintage) from a file.

    The file (``.npz`` or ``.npy``) holds the field's values over its active cells; they
    become the keys ``(time, datatype, 0..n-1)``.
    """

    time: str = Field(min_length=1)
    datatype: str = Field(min_length=1)
    file: Path
    key: str | None = Field(default=None, description="Array name in an .npz; default its first.")
    variance: float = Field(
        gt=0,
        description="Error variance of every value; a data transform (e.g. wavelet) may "
        "replace it with its own estimate.",
    )


class ObservationsSection(_Strict):
    """``[observations]``: scalar data from a long-format CSV and/or field data from files.

    The CSV has columns ``time,datatype,value,variance``. Fields follow the CSV's data.
    """

    file: Path | None = None
    fields: list[FieldObservationSection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _some_data(self) -> Self:
        if self.file is None and not self.fields:
            raise ValueError("[observations] needs a file, fields, or both")
        return self


class MethodSection(PluginSection):
    """``[method]``: an ensemble algorithm from the ``methods`` registry and its strategy.

    Iterative methods also take ``max_iterations`` and an optional ``schedule`` table
    (``DampingSchedule`` fields).
    """

    strategy: NamedPlugin | None = None
    localization: NamedPlugin | None = Field(
        default=None,
        description="Localization plugin passed to the strategy, e.g. `adaptive` (prior-only, "
        "alias `adaptive_prior`), `adaptive_per_update` or `distance`; its `block_sizes` come "
        "from the state layout. Unset (the default): no localization, except the local-analysis "
        "strategies (local_analysis, local_rlmmac, adaptive_rows), which select data by "
        "prior correlation.",
    )
    schedule: dict[str, Any] | None = None
    damping: NamedPlugin | None = Field(
        default=None,
        description="Damping scheme from the damping_schemes registry: rlmmac "
        "(GIES's default) or lm (LM-EnRML's default, PET's rule), with its options; for GIES "
        "and LM-EnRML. Without it, [method.schedule] gives the default scheme's options.",
    )
    max_iterations: int = Field(default=10, ge=1)

    @field_validator("schedule", mode="before")
    @classmethod
    def _renamed_tolerance(cls, value: object) -> object:
        # ``misfit_tolerance`` was renamed ``data_mismatch_tolerance`` on 2026-09-30 (ADR
        # 0027); configurations and run snapshots with the old name keep working.
        if isinstance(value, dict) and "misfit_tolerance" in value:
            if "data_mismatch_tolerance" in value:
                raise ValueError("give data_mismatch_tolerance or misfit_tolerance, not both")
            value = {
                ("data_mismatch_tolerance" if k == "misfit_tolerance" else k): v
                for k, v in value.items()
            }
        return value


#: ``[method]`` fields that define the algorithm. A study's method entry replaces
#: them all, so one algorithm never inherits another's options, strategy or schedule.
ALGORITHM_FIELDS = frozenset(
    {"name", "options", "diagnostics", "strategy", "schedule", "max_iterations"}
)
#: ``[method]`` fields of auxiliary techniques that strengthen any algorithm (localization,
#: the damping scheme; later inflation and similar schemes): kept from the case and changed
#: by study variants.
#: Every field of :class:`MethodSection` is in exactly one of the two sets.
AUXILIARY_FIELDS = frozenset({"localization", "damping"})


class FailuresSection(_Strict):
    """``[failures]``: what to do when member simulations fail."""

    min_success_fraction: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Drop failed members and continue while at least this fraction of the "
        "current ensemble succeeds; otherwise abort.",
    )


class CaseSection(_Strict):
    """``[case]``: a synthetic case study (known true model) or a real one.

    A synthetic case needs ``truth_file``: an ``.npz`` with one array per state variable,
    the parameters that generated the observations. Metrics against the true model
    (``rmse``, ``ssim``) run only in synthetic cases.
    """

    kind: Literal["synthetic", "real"] = "synthetic"
    truth_file: Path | None = None

    @model_validator(mode="after")
    def _truth_matches_kind(self) -> Self:
        if self.kind == "synthetic" and self.truth_file is None:
            raise ValueError(
                "a synthetic case study needs [case] truth_file (the true model's parameters, "
                'one array per state variable); for a field case set [case] kind = "real"'
            )
        if self.kind == "real" and self.truth_file is not None:
            raise ValueError('a real case study has no truth_file; set kind = "synthetic"')
        return self


class MetricSection(PluginSection):
    """``[[metrics]]``: a performance metric evaluated at every forecast.

    ``label`` names its results (default: ``name``), so one metric can be configured twice.
    """

    label: str | None = None

    @property
    def key(self) -> str:
        """Name under which the metric's results are stored."""
        return self.label or self.name


class OutputSection(_Strict):
    """``[output]``: optional extra results written during a run.

    Every run saves, per step, the ensemble and its forecast (``checkpoints/iter_<k>.npz``,
    for iterative methods the accepted iterates), per-member data mismatches and metrics
    (``data_mismatch_<k>.npz``) and the history. Perturbed observations are saved per step for
    ES-MDA (redrawn each step) and once for iterative methods.

    Diagnostics are saved only on request: ``diagnostics`` lists them run-wide,
    and **a plugin section's own ``diagnostics`` table overrides this list for that
    plugin** (both ways), as do its ``max_mb`` and ``every``. The effective plan is logged
    (event ``diagnostics_plan``, with a warning for each contradiction) and written to
    ``checkpoints/diagnostics_plan.json``.
    """

    save_rejected_trials: bool = Field(
        default=False,
        description="Iterative methods: also save each rejected trial ensemble and forecast "
        "(checkpoints/trial_<k>.npz).",
    )
    keep_simulations: Literal["all", "failed", "last", "failed_and_last", "none"] = Field(
        default="all",
        description="Which simulator work directories (runs/<name>/simulations/iter_<k>/) to "
        "keep: all (default); failed (only failed members'); last (only the latest "
        "forecast's); failed_and_last (the latest forecast's and every failed member's); "
        "none. Results never depend on them.",
    )
    save_raw_forecasts: bool = Field(
        default=False,
        description="With data transforms: also save the untransformed forecasts "
        "(raw_predicted) in each saved step.",
    )
    diagnostics: list[str] = Field(
        default_factory=list,
        description="Diagnostics to save (checkpoints/diagnostics_<k>.npz), e.g. "
        '["taper", "strategy.singular_values"]; default none. A bare name applies to every '
        "plugin producing it, 'role.name' to one role (method, strategy, localization). "
        "A plugin section's own diagnostics table overrides this for that plugin.",
    )
    diagnostics_max_mb: float = Field(
        default=200.0,
        gt=0,
        description="Size limit per item and step (MB); larger items are skipped, noted in "
        "the step's index. A plugin's diagnostics.max_mb overrides it.",
    )
    diagnostics_every: int = Field(
        default=1,
        ge=1,
        description="Save diagnostics every k-th iteration. A plugin's diagnostics.every "
        "overrides it.",
    )


class DataSelection(_Strict):
    """Which scalar data a role of ``[validation]`` takes.

    A datum ``KEYWORD:WELL`` at a report time is selected if it matches any of ``wells``,
    ``datatypes`` and ``keywords`` (and one of ``times``, when given). A datatype without
    ``:WELL`` is its own well for ``wells``. At least one of ``wells``, ``datatypes`` and
    ``keywords`` must be given.
    """

    wells: list[str] = Field(default_factory=list, description="Wells, e.g. ['PRO3'].")
    datatypes: list[str] = Field(
        default_factory=list, description="Exact datatypes, e.g. ['WOPR:PRO3']."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Keywords for every well, e.g. ['WWPR']."
    )
    times: list[str] | None = Field(
        default=None,
        description="Restrict the selection to these times, as the observations name them.",
    )

    @model_validator(mode="after")
    def _selects_something(self) -> Self:
        if not (self.wells or self.datatypes or self.keywords):
            raise ValueError("a data selection needs wells, datatypes or keywords")
        return self


class ValidationSection(_Strict):
    """``[validation]``: cross validation (CV) and test data for iterative methods.

    ``cv`` data are forecast at every iteration but never assimilated; their mean-square
    normalized data mismatch picks the posterior (the iterate where it is lowest) and, with the
    ``patience`` rule, stops the run early. Only iterative methods (e.g. GIES) take CV;
    ES-MDA's fixed schedule does not. ``test`` data are only scored, to check that CV
    works. ``history_matching`` defaults to every datum in neither. **Field data (e.g.
    seismic) are always history-matching data** and cannot be selected for cv or test.
    Without ``cv`` the method runs unchanged.

    ``patience`` assumes a roughly V-shaped CV curve. If the CV mismatch may first rise
    and then fall (the paper's Norne case), use ``select`` or a larger patience, or the
    run stops early and keeps the prior.
    """

    history_matching: DataSelection | None = Field(
        default=None,
        description="The data assimilated (a data selection: wells, datatypes, keywords, "
        "times); default every scalar datum in neither cv nor test.",
    )
    cv: DataSelection | None = Field(
        default=None,
        description="The cross-validation data: forecast at every iteration, never "
        "assimilated, their mismatch picks the posterior. A data selection, e.g. "
        '{ wells = ["PRO3"] } or { keywords = ["WWPR"] }.',
    )
    test: DataSelection | None = Field(
        default=None,
        description="Data only scored, never assimilated nor used to stop (a data "
        "selection, as cv), to check that cross validation works.",
    )
    stopping: NamedPlugin | None = Field(
        default=None,
        description='CV stopping rule; default { name = "patience", options = { patience = 2 } }, '
        'or "select" (short for { name = "select" }: run to the usual criteria). Needs cv.',
    )

    @model_validator(mode="after")
    def _stopping_needs_cv(self) -> Self:
        if self.stopping is not None and self.cv is None:
            raise ValueError("[validation] stopping needs a cv data set")
        return self

    def stopping_rule(self) -> PluginSection:
        """Return the configured stopping rule, or the default (patience 2)."""
        return self.stopping or PluginSection(name="patience", options={"patience": 2})


def avo_datatype(prefix: str, angle: float) -> str:
    """Return an AVO field's datatype, ``<prefix>_<angle>``.

    The angle is written with ``g`` formatting and its decimal point as ``p``: 10 gives
    ``avo_10``, 12.5 gives ``avo_12p5``.
    """
    return f"{prefix}_{angle:g}".replace(".", "p")


class OverburdenPressureFile(_Strict):
    """The overburden pressure of each cell, from a file (the reference's form)."""

    file: Path = Field(
        description="An .npz or .npy file: one value per active cell, or per grid cell in "
        "Eclipse (F) order."
    )
    key: str | None = Field(default=None, description="Array name in an .npz; default its first.")
    unit: Literal["Pa", "kPa", "MPa", "bar", "psi"] = Field(
        default="Pa", description="The values' pressure unit."
    )


class OverburdenPressureGradient(_Strict):
    """The overburden pressure as lithostatic: ``surface + gradient * depth``.

    ``depth`` is each cell centre's true vertical depth (m, positive down, from the run's
    INIT). Example::

        # Rock above the reservoir with an average density of 2300 kg/m³ gives
        #   gradient = 2300 kg/m³ * 9.81 m/s² ≈ 22,600 Pa/m  (≈ 1 psi/ft),
        # so a cell at 2000 m has p_ob ≈ 101,325 + 22,600 * 2000 ≈ 45.3 MPa.
        overburden_pressure = { gradient = 22600.0, surface = 101325.0 }   # Pa/m, Pa
    """

    gradient: float = Field(
        gt=0,
        description="Pa per m of depth: the overburden's mean density times g, e.g. "
        "2300 kg/m³ * 9.81 m/s² ≈ 22600 Pa/m (≈ 1 psi/ft), so that a cell at 2000 m has "
        "101325 + 22600 * 2000 ≈ 45.3 MPa.",
    )
    surface: float = Field(
        default=101325.0,
        ge=0,
        description="Pa at depth 0 (atmospheric by default; offshore, depth 0 is the sea "
        "surface and the water column is part of the gradient's average).",
    )


class RickerWaveletSection(_Strict):
    """A Ricker wavelet."""

    name: Literal["ricker"] = "ricker"
    frequency: float = Field(gt=0, description="Peak frequency (Hz).")
    length: float = Field(
        gt=0,
        description="Half the support (s), the reference's wave_len (for sampling = 'nearest').",
    )


class FileWaveletSection(_Strict):
    """A wavelet sampled in a file (``.npz``/``.npy``), zero-phase or not."""

    name: Literal["file"]
    file: Path
    key: str | None = Field(default=None, description="Array name in an .npz; default its first.")
    dt: float = Field(gt=0, description="The samples' interval (s).")
    centre: int | None = Field(
        default=None, ge=0, description="Index of the zero lag; default the largest |sample|."
    )


class ConstantOverburdenSection(_Strict):
    """A uniform medium above (and, by default, below) the reservoir: the reference's shale."""

    name: Literal["constant"] = "constant"
    vp: float = Field(gt=0, description="m/s")
    vs: float = Field(gt=0, description="m/s")
    density: float = Field(gt=0, description="kg/m³")


class LayeredOverburdenSection(_Strict):
    """Horizontal layers from the surface down; the last extends down."""

    name: Literal["layers"]
    thickness: list[float] = Field(description="m, one fewer than the velocities.")
    vp: list[float] = Field(min_length=1, description="m/s")
    vs: list[float] = Field(min_length=1, description="m/s")
    density: list[float] = Field(min_length=1, description="kg/m³")


class TimeAxisSection(_Strict):
    """The output times ``arange(start, end, step)`` (s, two-way)."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    step: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("times: end must be after start")
        return self


class RegionFileSection(_Strict):
    """A static region of interest from a file: the columns whose traces are kept."""

    file: Path = Field(
        description="An .npz or .npy file of shape (n_x, n_y) (I, J): nonzero keeps the column."
    )
    key: str | None = Field(default=None, description="Array name in an .npz; default its first.")


class RegionAroundWellsSection(_Strict):
    """A static region of interest around the deck's well completions.

    The columns kept are those whose centre (x, y of the grid's pillars' tops: grid
    coordinates before MAPAXES, in m) lies in an ``ellipse`` or ``rectangle`` around the
    completions' columns, widened by ``margin``. The completions are read from the deck
    template and its includes (``COMPDAT``). The shapes are the reference's, but for its
    completions, which it reads from the rendered deck only, and for fewer than two of
    them, where it surrounds every active column (APEX: the completion's own column, with
    ``margin`` around it). Mako control lines (``% if``) are not evaluated and included
    files over 5 MB are not read: when the completions depend on them, give the region as
    a file. Example::

        # Keep the traces within 500 m of the box around every completion, as an ellipse:
        region = { shape = "ellipse", margin = 500.0 }
    """

    shape: Literal["ellipse", "rectangle"]
    margin: float = Field(
        default=500.0, ge=0, description="m added to the completions' extent on each side."
    )
    wells: list[str] | None = Field(
        default=None, min_length=1, description="The wells around which; default every well."
    )
    x_range: tuple[float, float] | None = Field(
        default=None,
        description="[min, max] x (m, grid coordinates before MAPAXES) in place of the "
        "completions' extent.",
    )
    y_range: tuple[float, float] | None = Field(
        default=None,
        description="[min, max] y (m, grid coordinates before MAPAXES) in place of the "
        "completions' extent.",
    )

    @model_validator(mode="after")
    def _ranges(self) -> Self:
        for name, value in (("x_range", self.x_range), ("y_range", self.y_range)):
            if value is not None and not value[0] <= value[1]:
                raise ValueError(f"{name} must be [min, max]")
        return self


class AVOOptions(_Strict):
    """``options`` of an AVO datum (``kind = "avo"``)."""

    angles: list[float] = Field(min_length=1, description="Incidence angles (degrees).")
    reflectivity: NamedPlugin = Field(
        default_factory=lambda: PluginSection(name="zoeppritz"),
        description="A plugin of `reflectivity`: `zoeppritz` (exact, the default), "
        "`aki_richards`, `fatti`, `shuey`, `approx_zoeppritz_pp` (the 3dBox example's).",
    )
    # Plain unions: each member has a required field the other lacks, so a Ricker needs
    # no name (a discriminator would require one).
    wavelet: RickerWaveletSection | FileWaveletSection
    overburden: ConstantOverburdenSection | LayeredOverburdenSection
    underburden: ConstantOverburdenSection | None = Field(
        default=None, description="Below the reservoir; default the overburden's medium there."
    )
    times: TimeAxisSection
    time_window: tuple[float, float] | None = Field(
        default=None,
        description="[start, end] (s) of the samples a map attribute reduces, or a trace keeps; "
        "default every output time.",
    )
    form: Literal["maps", "traces"] = Field(
        default="maps", description="A value per trace (`attribute`) or the whole trace."
    )
    attribute: str = Field(
        default="rms",
        description="A map attribute of `avo_attributes` (`rms`, `mean`, `min`, `max`, "
        "`sum_negative`), for form = 'maps'; or `time_shift`, the change of the two-way "
        "time to the reservoir's base from the baseline (s), which needs a baseline (from the "
        "previous vintage with baseline_mode = 'rolling').",
    )
    sampling: Literal["interfaces", "nearest"] = Field(
        default="interfaces",
        description="'interfaces' (exact interface times, smooth) or 'nearest' (the "
        "reference's 0.5 ms staircase; to reproduce SimulatorWrap only).",
    )
    inactive: Literal["overburden", "above"] = Field(
        default="overburden",
        description="Properties of inactive cells: the overburden's, or the nearest active "
        "cell's above.",
    )
    four_d: Literal["vintages", "differences"] = Field(
        default="vintages",
        description="The data are each vintage's values, or their differences from the "
        "baseline's ('differences', with baseline): a map's attribute at the vintage minus "
        "the attribute at the baseline, a trace's samples minus the baseline's.",
    )
    baseline: str | None = Field(
        default=None,
        description="ISO date of the baseline survey, before every vintage, for four_d = "
        "'differences' or attribute = 'time_shift': each vintage's data are its values minus "
        "the baseline's, or with baseline_mode = 'rolling' minus the previous vintage's (the "
        "first vintage's minus the baseline's); it needs a restart too.",
    )
    baseline_mode: Literal["fixed", "rolling"] = Field(
        default="fixed",
        description="'fixed': every vintage minus the baseline; 'rolling': the first vintage "
        "minus the baseline, each later one minus the vintage before it (the vintages in "
        "increasing order).",
    )
    step_x: int = Field(default=1, ge=1, description="Keep every step_x-th column along I.")
    step_y: int = Field(default=1, ge=1, description="Keep every step_y-th column along J.")
    region: RegionAroundWellsSection | RegionFileSection | None = Field(
        default=None,
        description="A static region of interest: only its active columns' traces are kept "
        '(with step_x/step_y), e.g. { shape = "ellipse", margin = 500.0 } around the '
        'completions, or { file = "region.npy" }; default every active column.',
    )

    @field_validator("reflectivity", mode="before")
    @classmethod
    def _plugin_name(cls, value: object) -> object:
        return {"name": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.angles)) != len(self.angles):
            raise ValueError("angles must be distinct")
        if not all(0 <= a < 90 for a in self.angles):
            raise ValueError("angles must be in [0, 90) degrees")
        if self.time_window is not None and not self.time_window[0] < self.time_window[1]:
            raise ValueError("time_window must be [start, end] with start < end")
        # By value, not model_fields_set: a dumped configuration (config.json, a job's
        # spec) writes every field, and must read back.
        if self.form == "traces" and self.attribute != "rms":
            raise ValueError("form = 'traces' keeps the whole trace; drop attribute")
        if self.attribute == "time_shift" and len(self.angles) != 1:
            # The same at every angle: copies would count as independent data.
            raise ValueError("time_shift takes one angle (vertical incidence), e.g. angles = [0]")
        # The time shift is a difference by nature, whatever four_d says.
        if self.four_d == "differences" or self.attribute == "time_shift":
            if self.baseline is None:
                needs = "time_shift" if self.attribute == "time_shift" else "four_d = 'differences'"
                raise ValueError(f"{needs} needs a baseline (an ISO date)")
            try:
                date.fromisoformat(self.baseline)
            except ValueError as error:
                raise ValueError(f"baseline {self.baseline!r} is not an ISO date") from error
        elif self.baseline is not None:
            raise ValueError("a baseline is for four_d = 'differences' (or time_shift)")
        return self


class RegularStationsSection(_Strict):
    """Survey stations on a regular grid over the cells, the reference's.

    ``ceil(span / spacing)`` stations along each axis of the cells' extent widened by
    ``padding``, x fastest; optionally twice as dense within ``refine``'s radius of a point.
    Coordinates (``refine``, a seabed table, a station file) are the grid's own (its
    COORD, before MAPAXES), in m.
    Example::

        # Seabed stations every 500 m over the field and 1.5 km around it, 300 m down:
        stations = { spacing = 500.0, padding = 1500.0, depth = 300.0 }
    """

    spacing: float = Field(gt=0, description="m: about the stations' spacing.")
    padding: float = Field(default=0.0, ge=0, description="m around the cells' extent.")
    depth: float = Field(
        default=0.0,
        description="m, positive down: the stations' depth (the water depth for seabed "
        "stations; negative onshore, above the datum).",
    )
    seabed_file: Path | None = Field(
        default=None,
        description="An .npy/.npz (n, 3) table of x, y, depth (m): each station takes its "
        "nearest depth, in place of depth.",
    )
    seabed_key: str | None = Field(
        default=None, description="seabed_file's array in an .npz; default its only one."
    )
    refine: tuple[float, float, float] | None = Field(
        default=None, description="[x, y, radius] (m): twice as dense within the circle."
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # By value (a dumped configuration writes every field): depth's default is 0.
        if self.seabed_file is not None and self.depth != 0.0:
            raise ValueError("seabed_file gives the stations' depths: drop depth")
        if self.seabed_key is not None and self.seabed_file is None:
            raise ValueError("seabed_key is for seabed_file")
        if self.refine is not None and not self.refine[2] > 0:
            raise ValueError("refine's radius must be positive")
        if self.refine is not None and round(self.spacing / 2) == 0:
            raise ValueError(
                "refine needs a spacing above 1 m (its stations are spacing / 2 apart, rounded)"
            )
        return self


class StationFileSection(_Strict):
    """Survey stations from a file: an .npy/.npz ``(n, 3)`` array of x, y, depth (m)."""

    file: Path
    key: str | None = Field(default=None, description="Array name in an .npz; default its first.")


def _baseline_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"baseline {value!r} is not an ISO date") from error
    return value


class GravityOptions(_Strict):
    """``options`` of a gravity datum (``kind = "gravity"``).

    The data are Δg (µGal, positive down; free-air corrected for the stations' own movement,
    the rock's mass change from compaction left out) at the stations: the attraction of
    each active cell's fluid-mass change, Σ_phases density * saturation * pore volume
    (RPORV), from the baseline to the vintage. The restarts need RPORV and the phases'
    densities (``RPTRST BASIC=2 DEN RPORV``).
    """

    baseline: str = Field(description="ISO date of the baseline survey (needs a restart).")
    stations: RegularStationsSection | StationFileSection
    model: Literal["point_mass", "prism"] = Field(
        default="point_mass",
        description="'point_mass' (the reference's) or 'prism' (each cell a uniform "
        "rectangular prism over its corners' horizontal extent: exact near the cells of a "
        "grid aligned with x and y).",
    )
    gravitational_constant: float = Field(
        default=6.67430e-11,
        gt=0,
        description="m³ kg⁻¹ s⁻² (CODATA 2018; the reference's 6.67e-11).",
    )

    _baseline = field_validator("baseline")(_baseline_date)


class DisplacementOptions(_Strict):
    """``options`` of a seafloor or surface displacement datum.

    The data are the vertical displacement (cm, positive down: subsidence) at the stations
    from each active cell's volume decrease since the baseline: ``c_m V (P_baseline - P)``
    (``volume_change = "pressure"``; V the cell's bulk volume, or its pore volume at the
    baseline with ``volume = "pore"``, the reference's) or the pore-volume decrease
    ``RPORV_baseline - RPORV`` (``"pore_volume"``, which needs RPORV in the restarts).
    Example::

        # Uniaxial compaction coefficient 1e-4 per MPa = 1e-10 per Pa:
        options = { baseline = "2020-01-01", compressibility = 1e-10,
                    stations = { spacing = 500.0, padding = 1500.0, depth = 300.0 } }
    """

    baseline: str = Field(description="ISO date of the baseline survey (needs a restart).")
    stations: RegularStationsSection | StationFileSection
    kernel: NamedPlugin = Field(
        default_factory=lambda: PluginSection(name="geertsma"),
        description="A plugin of `displacement_kernels`: `geertsma` (a half space, the "
        "default) or `van_opstal` (a rigid basement: basement_depth).",
    )
    volume_change: Literal["pressure", "pore_volume"] = Field(
        default="pressure",
        description="'pressure': c_m V ΔP; 'pore_volume': the change of RPORV.",
    )
    compressibility: float | None = Field(
        default=None, gt=0, description="c_m (1/Pa), the uniaxial compaction coefficient."
    )
    volume: Literal["bulk", "pore"] = Field(
        default="bulk",
        description="V in c_m V ΔP: the cell's bulk volume (Geertsma's; non-net rock "
        "included), or its pore volume at the baseline (the reference's).",
    )

    _baseline = field_validator("baseline")(_baseline_date)

    @field_validator("kernel", mode="before")
    @classmethod
    def _plugin_name(cls, value: object) -> object:
        return {"name": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _compressibility(self) -> Self:
        if self.volume_change == "pressure" and self.compressibility is None:
            raise ValueError("volume_change = 'pressure' needs compressibility (1/Pa)")
        if self.volume_change == "pore_volume" and (
            self.volume != "bulk" or self.compressibility is not None
        ):
            raise ValueError("volume and compressibility are for volume_change = 'pressure'")
        return self


class GeophysicsDataSection(_Strict):
    """``[[geophysics.data]]``: one kind of geophysical data, at its vintages."""

    kind: Literal["avo", "gravity", "seafloor_displacement"] = Field(
        description="'avo', 'gravity' or 'seafloor_displacement' (onshore too)."
    )
    datatype: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        description="The observations' datatype: AVO fields are `<datatype>_<angle>`, e.g. "
        "avo_10, avo_12p5; gravity and displacement fields are `<datatype>`.",
    )
    vintages: list[str] = Field(
        min_length=1, description="ISO dates of the surveys; each needs a restart."
    )
    options: AVOOptions | GravityOptions | DisplacementOptions

    @model_validator(mode="before")
    @classmethod
    def _options_of_the_kind(cls, data: object) -> object:
        # Each kind's options, so that their errors name the kind's own fields.
        if isinstance(data, dict) and isinstance(data.get("options"), dict):
            kinds: dict[str, type[BaseModel]] = {
                "avo": AVOOptions, "gravity": GravityOptions,
                "seafloor_displacement": DisplacementOptions,
            }  # fmt: skip
            kind = str(data.get("kind"))
            model = kinds.get(kind)
            if model is not None:
                try:
                    options = model.model_validate(data["options"])
                except ValidationError as error:
                    # Raised here, the errors would lose their "options" path.
                    problems = "; ".join(
                        f"{'.'.join(('options', *map(str, e['loc'])))}: {e['msg']}"
                        for e in error.errors()
                    )
                    raise ValueError(f"{kind} {problems}") from None
                data = {**data, "options": options}
        return data

    @field_validator("vintages")
    @classmethod
    def _dates(cls, value: list[str]) -> list[str]:
        for vintage in value:
            try:
                date.fromisoformat(vintage)
            except ValueError as error:
                raise ValueError(f"vintage {vintage!r} is not an ISO date") from error
        if len(set(value)) != len(value):
            raise ValueError("vintages must be distinct")
        return value

    def baseline(self) -> str | None:
        """Return the baseline's ISO date: an AVO difference's, gravity's, displacement's."""
        return self.options.baseline

    def avo(self) -> AVOOptions:
        """Return the options of an AVO datum.

        Raises
        ------
        TypeError
            If the datum is not AVO.
        """
        if not isinstance(self.options, AVOOptions):
            raise TypeError(f"{self.datatype} is a {self.kind} datum, not AVO")
        return self.options

    def gravity(self) -> GravityOptions:
        """Return the options of a gravity datum.

        Raises
        ------
        TypeError
            If the datum is not gravity.
        """
        if not isinstance(self.options, GravityOptions):
            raise TypeError(f"{self.datatype} is a {self.kind} datum, not gravity")
        return self.options

    def displacement(self) -> DisplacementOptions:
        """Return the options of a displacement datum.

        Raises
        ------
        TypeError
            If the datum is not displacement.
        """
        if not isinstance(self.options, DisplacementOptions):
            raise TypeError(f"{self.datatype} is a {self.kind} datum, not displacement")
        return self.options

    @model_validator(mode="after")
    def _after_the_baseline(self) -> Self:
        kinds = {"avo": AVOOptions, "gravity": GravityOptions,
                 "seafloor_displacement": DisplacementOptions}  # fmt: skip
        if not isinstance(self.options, kinds[self.kind]):
            raise ValueError(f"options are not those of kind {self.kind!r}")
        baseline = self.options.baseline
        early = [
            v
            for v in self.vintages
            if baseline is not None and date.fromisoformat(v) <= date.fromisoformat(baseline)
        ]
        if early:
            raise ValueError(
                f"vintages {early} are not after the baseline {baseline} (a difference from "
                "the baseline itself is zero)"
            )
        if isinstance(self.options, AVOOptions) and self.options.baseline_mode == "rolling":
            if self.options.baseline is None:
                raise ValueError(
                    "baseline_mode = 'rolling' takes differences: four_d = 'differences' or "
                    "attribute = 'time_shift', with a baseline"
                )
            dates = [date.fromisoformat(v) for v in self.vintages]
            if any(b <= a for a, b in pairwise(dates)):
                raise ValueError(
                    "baseline_mode = 'rolling' takes each vintage minus the one before it: "
                    "list the vintages in increasing order"
                )
        return self

    def datatypes(self) -> list[str]:
        """Return the field datatypes this datum gives: one per angle for AVO."""
        if isinstance(self.options, AVOOptions):
            return [avo_datatype(self.datatype, a) for a in self.options.angles]
        return [self.datatype]


class GeophysicsSection(_Strict):
    """``[geophysics]``: geophysical data forecast from the deck's runs.

    For AVO data a petro-elastic model turns each cell's flow state at a vintage
    (porosity, pressure, saturations and, for some models, the phases' densities, read from
    the run's INIT and restart files) into Vp, Vs and density; gravity and displacement data
    take the cells' fluid masses or volume changes. Needs a deck simulator (``opm_flow``,
    ``eclipse``).
    """

    pem: NamedPlugin | None = Field(
        default=None,
        description="A plugin of `petro_elastic_models`: `standard` or `soft_sand` (for AVO data).",
    )
    overburden_pressure: OverburdenPressureGradient | OverburdenPressureFile | None = Field(
        default=None,
        description="For AVO data, the overburden (lithostatic) pressure, for the effective "
        "pressure: "
        "{ gradient = 22600.0, surface = 101325.0 } (Pa/m, Pa: 2300 kg/m³ * 9.81 m/s² over "
        "the cell centre's depth, so 45.3 MPa at 2000 m), or one value per cell from a file, "
        '{ file = "overburden.npz", key = "obvalues", unit = "bar" }.',
    )
    grid_file: Path = Field(
        description="An EGRID of the deck (e.g. the truth's or a base-case run's): its "
        "active columns fix the traces and its cells the gravity and displacement kernels, "
        "the same for every member."
    )
    data: list[GeophysicsDataSection] = Field(min_length=1)
    max_field_values: int = Field(
        default=1_000_000,
        ge=1,
        description="Most field values per member (every field at every vintage) without a "
        "data transform: beyond it each field needs one (e.g. wavelet), as whole traces "
        "would fill the checkpoints and memory.",
    )

    @model_validator(mode="after")
    def _distinct_datatypes(self) -> Self:
        avo = any(d.kind == "avo" for d in self.data)
        if avo and (self.pem is None or self.overburden_pressure is None):
            raise ValueError("AVO data need [geophysics] pem and overburden_pressure")
        if not avo and (self.pem is not None or self.overburden_pressure is not None):
            raise ValueError("pem and overburden_pressure are used only by AVO data; remove them")
        names = [d for section in self.data for d in section.datatypes()]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"[[geophysics.data]] repeat the datatypes {repeated}")
        return self


class HistoryMatchConfig(_Strict):
    """A complete history-matching run.

    ``data_transforms`` (``[[data_transforms]]``) are applied in order to the
    observations and to every forecast before the analysis, e.g. ``wavelet`` compression.
    """

    run: RunSection
    state: StateSection
    prior: PriorSection
    observations: ObservationsSection
    simulator: PluginSection
    execution: PluginSection = Field(default_factory=lambda: PluginSection(name="serial"))
    method: MethodSection
    failures: FailuresSection = Field(default_factory=FailuresSection)
    data_transforms: list[PluginSection] = Field(default_factory=list)
    parameterizations: list[PluginSection] = Field(
        default_factory=list,
        description="Maps from the state to the properties the simulator reads, applied in "
        "order, e.g. `pgs_truncation`; none: the state is passed as is.",
    )
    model_error: NamedPlugin | None = Field(
        default=None,
        description="A model-error (MEC) model added to the simulator's output, "
        "e.g. `pg_lstm`; its parameters are the state variable it names (default `nn_params`).",
    )
    geophysics: GeophysicsSection | None = Field(
        default=None,
        description="Geophysical data (AVO) forecast from the deck's runs.",
    )
    output: OutputSection = Field(default_factory=OutputSection)
    case: CaseSection = Field(default_factory=CaseSection)
    validation: ValidationSection | None = None
    metrics: list[MetricSection] | None = Field(
        default=None,
        description="Performance metrics; default `normalized_data_mismatch`, plus `rmse` in a "
        "synthetic "
        "case (`ssim` needs its grid, so it must be configured).",
    )
