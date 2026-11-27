"""An optimization run's configuration.

Example::

    [optimization]
    name = "rosenbrock_enopt"
    seed = 1

    [[controls.variables]]
    name = "x"
    initial = [0.5, 0.5, 0.5]
    lower = -2.0
    upper = 2.0
    variance = 0.01            # initial perturbation variance (ensemble optimizers)

    [objective]
    name = "rosenbrock"

    [optimizer]
    name = "enopt"
    options = { ne = 10, maxiter = 50 }

A simulator-backed objective (phase 2) names a response objective and the simulator that
gives its responses; each column of controls becomes one simulation, its controls the
member's state (a deck template reads ``${injbhp[0]}``)::

    [objective]
    name = "npv"
    options = { start = "1994-02-09", times = ["1995-01-01", "1996-01-01"], wop = 283.05,
                wwp = 37.74, wwi = 12.58, disc = 0.08, obj_scaling = -1.0e6 }

    [simulator]
    name = "opm_flow"
    options = { deck_template_file = "3WELL.mako" }

    [execution]
    name = "process"
    options = { workers = 4 }

    [model]                    # optional: fixed model realizations (robust optimization)
    realizations_file = "permx_realizations.npz"

Unknown keys are refused. Relative paths are resolved against the configuration's directory.
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from apex.domain.config import PluginSection, _Strict
from apex.domain.surrogate import QuantityKind


class OptimizationSection(_Strict):
    """``[optimization]``: identity and reproducibility."""

    name: str = Field(min_length=1)
    seed: int = Field(ge=0, description="Root seed of the perturbations' random stream.")
    output_dir: Path | None = Field(
        default=None, description="Run directory; default `runs/<name>` next to the config."
    )

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        # The name places the optimization (runs/<name>, the service's optimizations/<name>,
        # the web UI's UI/<user>/optimizations/<name>): a separator or "." / ".." would put
        # it elsewhere, out of a guest's own folder (the review).
        if "/" in value or "\\" in value or value in (".", ".."):
            raise ValueError(
                f"optimization name {value!r}: one folder name, no '/' or '\\', not '.' or '..'"
            )
        return value


class ControlVariable(_Strict):
    """``[[controls.variables]]``: one named block of controls."""

    name: str = Field(min_length=1)
    initial: float | list[float] = Field(description="Initial value(s) of the block.")
    lower: float | None = Field(default=None, description="Lower bound of every value.")
    upper: float | None = Field(default=None, description="Upper bound of every value.")
    variance: float | None = Field(
        default=None,
        gt=0,
        description="Initial perturbation variance (needed by the ensemble optimizers).",
    )

    @property
    def size(self) -> int:
        """Return the number of controls in the block."""
        return 1 if isinstance(self.initial, float | int) else len(self.initial)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        values = [self.initial] if isinstance(self.initial, float | int) else self.initial
        if not values:
            raise ValueError(f"controls {self.name!r} have no initial value")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(f"controls {self.name!r}: lower must be below upper")
        outside = [
            v
            for v in values
            if (self.lower is not None and v < self.lower)
            or (self.upper is not None and v > self.upper)
        ]
        if outside:
            raise ValueError(f"controls {self.name!r}: initial values {outside} are out of bounds")
        return self


class ControlsSection(_Strict):
    """``[controls]``: the variables and the model realizations of robust optimization."""

    variables: list[ControlVariable] = Field(min_length=1)
    realizations: int = Field(
        default=1, ge=1, description="Model realizations (PET's num_models) the objective averages."
    )

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"control names repeat: {names}")
        return self


class ModelSection(_Strict):
    """``[model]``: what every simulation reads besides the controls (phase 2)."""

    realizations_file: Path | None = Field(
        default=None,
        description="An .npz of model arrays (n_values, n_realizations) by name, e.g. permx: "
        "column r goes into the simulations of realization r (robust optimization; sets "
        "[controls] realizations).",
    )


class OptimizationOutput(_Strict):
    """``[output]``: what an optimization keeps besides its history."""

    keep_simulations: Literal["all", "failed", "none"] = Field(
        default="failed",
        description="Simulation folders kept in simulations/: all, the failed ones, or none.",
    )


class OptimizationSurrogateSection(_Strict):
    """``[surrogate]``: evaluations by a fitted surrogate, checked by the simulator.

    Every evaluation the optimizer asks for is made by the surrogate whose manifest
    ``model_file`` names (built by ``apex surrogate build`` on this optimization), on local
    workers; the ``[simulator]`` runs the checks (every ``check_every`` accepted steps, the
    accepted controls at every realization) and the reported optimum, which is always the
    simulator's.
    """

    model_file: Path = Field(
        description="The surrogate's manifest (`.json`, its arrays beside it), e.g. "
        "`runs/<build>/model/surrogate.json`."
    )
    check_every: int = Field(
        default=3,
        ge=0,
        description="Accepted steps whose controls are also evaluated by the simulator and "
        "gated (steps 3, 6, 9, ... by default); 0: no checks.",
    )
    on_check_failure: Literal["stop", "simulator"] = Field(
        default="stop",
        description="A check that fails: `stop` the optimization at its last checkpoint "
        "(resumable); or start it again with the `simulator` for every evaluation (the "
        "plain run with the same seed; the evaluations already simulated are reused).",
    )
    outside: Literal["stop", "simulator", "surrogate"] = Field(
        default="stop",
        description="Controls outside the surrogate's validated domain: `stop` the "
        "optimization, evaluate them by the `simulator` (recorded as such), or evaluate them "
        "by the `surrogate` anyway (an explicit choice to extrapolate: logged at every batch, "
        "marked in the journal and counted in the result, as in history matching).",
    )

    relative: dict[QuantityKind, float] = Field(
        default_factory=dict,
        description="The checks' relative tolerances by kind of quantity (pressure, rate, "
        "plume, mass, other): may only tighten the model's own gate.",
    )
    workers: int | None = Field(
        default=None,
        ge=1,
        description="Local worker processes for the surrogate's evaluations; none: in this "
        "process (serial).",
    )
    max_evaluations: int = Field(
        default=50_000,
        ge=1,
        description="The run's bound on surrogate evaluations (one per control vector and "
        "realization).",
    )
    simulate_every: int = Field(
        default=0,
        ge=0,
        description="Iterations evaluated wholly by the simulator: the "
        "first and every j-th after it (j = simulate_every: iterations 1, 1 + j, 1 + 2j, ...), "
        "their perturbation ensemble, trial points and the iterate's value; the surrogate "
        "answers in between. 0: the surrogate answers every iteration. Needs `outside = "
        '"simulator"`. Optimizers whose only objective value carried between iterations is '
        "the iterate's (EnOpt).",
    )
    refit: bool = Field(
        default=False,
        description="Refit the surrogate after each simulated iteration's accepted step, "
        "through its plugin, from the simulator's rows of the run's journal (never the "
        "surrogate's own), as `refit_rows` says; the refit has no held-out runs, the checks "
        "of the steps it takes gate it: needs 1 <= `check_every` <= `simulate_every` - 1.",
    )
    refit_rows: Literal["all", "latest"] = Field(
        default="all",
        description="What a refit is fitted on: `all`, the model's own build runs and every "
        "simulator row of the run so far; `latest`, only the simulator rows since the "
        "previous refit (a local model around the iterate).",
    )

    @model_validator(mode="after")
    def _refit_needs_a_schedule(self) -> Self:
        if self.refit and self.simulate_every < 1:
            raise ValueError(
                "refit = true refits from the simulated iterations' rows: give simulate_every >= 1"
            )
        if self.refit and not 1 <= self.check_every <= self.simulate_every - 1:
            # A refit serves iterations s + 1 ... s + j - 1 (j = simulate_every): only a
            # check among them gates it (the check at s tests the model it replaces).
            raise ValueError(
                "refit = true needs 1 <= check_every <= simulate_every - 1: a refit has no "
                "held-out runs, and a check of a step it took is its gate"
            )
        if self.simulate_every and self.outside != "simulator":
            # A simulated iteration's accepted step may leave the surrogate's domain: with
            # "stop" the run would stop there again on every resume.
            raise ValueError(
                'simulate_every needs outside = "simulator": the simulated iterations go '
                "where the surrogate's domain may not reach"
            )
        if self.refit_rows != "all" and not self.refit:
            raise ValueError("refit_rows applies with refit = true only")
        return self


class ObjectiveEntry(PluginSection):
    """One entry of ``[[objectives]]``: a scalar objective among several."""

    label: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="The objective's name in results and plots (default: its plugin's name, "
        "with _<statistic> unless the statistic is mean).",
    )
    statistic: str = Field(
        default="mean",
        pattern=r"^(mean|std|worst|p([1-9][0-9]?))$",
        description="What of its values over the realizations is minimised: mean, "
        "std (sample, at least 2 realizations), worst (the largest), or a percentile p1 ... "
        "p99 (e.g. p10, p50, p90) of the objective as minimised: for -NPV, p90 is NPV's "
        "10th percentile. The same objective may be listed with several statistics.",
    )

    @property
    def shown(self) -> str:
        """Return the name results show: the label, else the plugin's name (and statistic)."""
        if self.label:
            return self.label
        return self.name if self.statistic == "mean" else f"{self.name}_{self.statistic}"


class OptimizationConfig(_Strict):
    """A complete optimization run."""

    optimization: OptimizationSection
    controls: ControlsSection
    objective: PluginSection | None = Field(
        default=None,
        description="The objective: one plugin (a scalar one, or one that returns several, such "
        "as the test problem zdt1). Give either this or [[objectives]].",
    )
    objectives: list[ObjectiveEntry] = Field(
        default_factory=list,
        description="Several scalar objectives optimized together (e.g. by nsga2): each a "
        "plugin with options and an optional label. Response objectives share one simulation "
        "per member.",
    )
    optimizer: PluginSection
    simulator: PluginSection | None = Field(
        default=None, description="The simulator of a response objective such as npv."
    )
    execution: PluginSection = Field(
        default_factory=lambda: PluginSection(name="serial"),
        description="Where the simulations run (with [simulator]).",
    )
    model: ModelSection = Field(default_factory=ModelSection)
    parameterizations: list[PluginSection] = Field(
        default_factory=list,
        description="Maps from the controls to what the simulator reads, applied in order "
        "(control_schedule, control_scaling, ...); none: the controls "
        "are passed as is.",
    )
    output: OptimizationOutput = Field(default_factory=OptimizationOutput)
    surrogate: OptimizationSurrogateSection | None = Field(
        default=None,
        description="Evaluations by a fitted surrogate, checked by the simulator.",
    )

    @model_validator(mode="after")
    def _one_objective_form(self) -> Self:
        if (self.objective is None) == (not self.objectives):
            raise ValueError("give either [objective] or a list [[objectives]], not both")
        if self.objectives:
            if len(self.objectives) < 2:
                raise ValueError("[[objectives]] lists several objectives; for one use [objective]")
            shown = [entry.shown for entry in self.objectives]
            if len(set(shown)) != len(shown):
                raise ValueError(f"[[objectives]] names repeat ({shown}): give each a label")
        return self

    @property
    def objective_sections(self) -> list[PluginSection]:
        """Return the objective's section, or the list's."""
        if self.objective is not None:
            return [self.objective]
        return list(self.objectives)

    @property
    def objective_label(self) -> str:
        """Return the objective's name for summaries: its plugin's, or the list's labels."""
        if self.objective is not None:
            return self.objective.name
        return "+".join(entry.shown for entry in self.objectives)

    @model_validator(mode="after")
    def _model_needs_a_simulator(self) -> Self:
        if self.model.realizations_file is not None and self.simulator is None:
            raise ValueError("[model] realizations_file needs a [simulator]")
        if self.parameterizations and self.simulator is None:
            raise ValueError("[[parameterizations]] map controls for a [simulator]")
        if self.surrogate is not None and self.simulator is None:
            raise ValueError("[surrogate] stands in for a [simulator], which this has none of")
        return self
