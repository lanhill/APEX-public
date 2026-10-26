---
audience: public
example_plugins: [mae, geometric_esmda]
---

# Getting started: write and register your own plugins

## Goal

By the end of this page you have written plugins for APEX in packages of your own, registered them, and used them in runs, with tests. Two worked examples:

1. **A performance metric** (Steps 1 to 6): the simplest kind of plugin, a class with one method.
2. **A history-matching algorithm** ([Example 2](#example-2-a-new-history-matching-algorithm)): a new method that APEX's workflow runs, checkpoints and resumes like its own.

The same steps make any other kind of plugin: an update strategy, a localization, a prior, a simulator, an optimizer and more.

Almost everything in APEX is a plugin: methods, update strategies, localizations, priors, simulators, executors, metrics, optimizers, objectives and figures. Each kind has a *registry*. A configuration names a plugin, and APEX creates it from its registry:

```toml
[[metrics]]
name = "mae"                  # the plugin's registered name
options = { scale = 1.0 }     # passed to its constructor
```

So adding a capability means writing a class and registering it. APEX's own code stays unchanged, and your package can be shared and installed by others.

## Prerequisites

- APEX installed as the README describes, and Python 3.12.
- Some familiarity with the kind of plugin you write. `uv run apex plugins` lists every registry and its plugins, and the [reference](../reference/plugins/methods.md) gives each one's options.

## Step 1: choose the kind of plugin

Each registry has an entry-point group, `apex.<registry>`, where installed packages add plugins. Its interface is a Python protocol in `apex.core` or `apex.algorithms`; read an existing plugin of the same kind as a model.

| Registry (entry-point group `apex.<name>`) | What a plugin is | Interface |
|---|---|---|
| `metrics` | a performance metric, per member, at every forecast | `apex.core.metrics.PerformanceMetric` |
| `update_strategies` | the linear algebra of an update | `apex.algorithms.data_assimilation.protocols.UpdateStrategy` |
| `localizations` | a localization scheme | `apex.algorithms.data_assimilation.localization.Localization` |
| `methods` | an iterative method (ES-MDA, GIES, ...) | `apex.algorithms.data_assimilation.algorithm.EnsembleAlgorithm` |
| `priors` | a prior ensemble generator | `apex.core.simulation.PriorGenerator` |
| `simulators` | a forward model | `apex.core.simulation.Simulator` |
| `executors` | where members run | `apex.core.simulation.Executor` |
| `data_transforms` | a transform of the data | `apex.core.simulation.DataTransform` |
| `optimizers` | an optimizer | `apex.algorithms.optimization.base.OptimizerBase` |
| `objectives` | an analytic objective | `apex.algorithms.optimization.protocols.Objective` |

`uv run apex plugins --json` shows every registry and the options of its plugins.

## Example 1: a performance metric

### Step 2: write the plugin

Create a package of your own, outside the APEX folder:

```text
apex-mae/
├── pyproject.toml
├── src/apex_mae/__init__.py
└── tests/test_mae.py
```

`src/apex_mae/__init__.py`, a metric: the mean absolute data error of each member, in units of the observation errors.

```python
"""A mean absolute error, in units of the observation errors, as an APEX metric."""

import numpy as np
from numpy.typing import NDArray

from apex.core.metrics import MetricContext


class MeanAbsoluteError:
    """Mean absolute data error per member, in units of the observation errors.

    Parameters
    ----------
    scale
        A factor applied to the result (an example of an option).
    """

    #: The metric needs no true model, so it runs in real cases too.
    requires_truth = False

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def evaluate(self, context: MetricContext) -> dict[str, NDArray[np.float64]]:
        """Return ``{"all": (n_members,)}``: mean |d_obs - d| / sigma over the data."""
        observations = context.observations
        sigma = np.sqrt(observations.variances)[:, None]
        error = np.abs(observations.values[:, None] - context.predicted) / sigma
        return {"all": self.scale * error.mean(axis=0)}
```

What APEX takes from it:

- **The options** are the constructor's keyword arguments, with their type hints and defaults. APEX builds the options' schema from them, refuses unknown or mistyped options when a configuration is validated, and shows them in `apex plugins` and the web UI.
- **The summary** is the first line of the class docstring.
- **Arrays** keep the ensemble on the last axis: the forecast is `(n_data, n_members)`, a state variable `(size, n_members)`.
- **Errors:** raise `ValueError` with a clear message for invalid options or inputs.
- **Options that name files or programs:** annotate them with the types in `apex.core.options` (`InputFile`, `CodeFile`, `Program`, ...). A service with a `[policy]` then knows what they are and can check them.
- **Optional `status`:** a class attribute `status = "experimental"` is shown in `apex plugins`.
- **Randomness:** a plugin that needs random numbers takes a `numpy.random.Generator` from APEX, never the global `np.random`. Runs are then reproducible and resume bit for bit.

### Step 3: register it

In your `pyproject.toml`, add an entry point in the registry's group. Its name is the plugin's name in configurations:

```toml
[project]
name = "apex-mae"
version = "0.1.0"
description = "A mean-absolute-error performance metric for APEX"
requires-python = ">=3.12"
dependencies = ["numpy"]

[project.entry-points."apex.metrics"]
mae = "apex_mae:MeanAbsoluteError"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

APEX finds entry points when a registry is first used, so nothing else is needed. A name that is already taken is refused, so choose a distinctive one.

### Step 4: install it and check it is found

Install your package into the environment that runs APEX. In the APEX folder:

```bash
uv pip install -e /path/to/apex-mae
uv run apex plugins | grep mae
```

```text
metrics            mae                Mean absolute data error per member, in units of the observation errors.
```

A later plain `uv sync` makes the environment exactly APEX's lock again, and removes your package. Install it again, or use `uv sync --locked --inexact`, which keeps it.

To share the plugin, make it a project that depends on APEX instead, as the README's *Using APEX from another project* describes. Its users then install both together.

### Step 5: use it in a run

Copy `examples/linear_gaussian/config_esmda.toml` to `config_mae.toml` in the same folder, and give its run a name of its own (`[run] name = "linear_mae"`), so that it does not meet a run you made before. Then add the metric. Listing `[[metrics]]` replaces the default metrics, so name the ones you also want:

```toml
[[metrics]]
name = "normalized_data_mismatch"

[[metrics]]
name = "rmse"

[[metrics]]
name = "mae"
options = { scale = 1.0 }
```

```bash
cd examples/linear_gaussian
uv run apex validate config_mae.toml
uv run apex run config_mae.toml
uv run apex summary runs/linear_mae
```

The summary now has a column for it, from the prior to the result, e.g. `mae  4.082 → 0.9901`. The same values are in the run's checkpoints, in `apex report --plots`, on the web UI's run page and available to the agents.

### Step 6: test it

`tests/test_mae.py`, a test against a value you can work out by hand:

```python
import numpy as np

from apex.core.metrics import MetricContext
from apex.core.plugins import metrics
from apex.domain.observations import DataKey, ObservationSet


def test_mae_is_registered_and_computes_the_mean_error() -> None:
    keys = [DataKey("2020-01-01", "WOPR:P1"), DataKey("2020-02-01", "WOPR:P1")]
    observations = ObservationSet(keys, values=[1.0, 2.0], variances=[1.0, 4.0])
    predicted = np.array([[1.0, 3.0], [2.0, 6.0]])  # (n_data, n_members)
    context = MetricContext(state={}, predicted=predicted, observations=observations)

    metric = metrics.create("mae")  # found through the entry point

    np.testing.assert_allclose(metric.evaluate(context)["all"], [0.0, 2.0])
```

```bash
uv run pytest /path/to/apex-mae/tests
```

For a numerical method, test it against an independent reference as APEX's own tests do: an analytic solution (the linear-Gaussian case has an exact posterior), a published benchmark, or convergence on a toy problem.

## Example 2: a new history-matching algorithm

A history-matching method is an `EnsembleAlgorithm` (`apex.algorithms.data_assimilation.algorithm`): a small state machine that APEX's workflow drives. The method never runs a simulator itself. It asks for forecasts, and the workflow runs them on the configured simulator and executor, with failed-member handling, checkpoints, metrics, progress and cross validation. So a new method is one class, and it gets all of that without code of its own.

The example is **ES-MDA with a geometric inflation schedule**. ES-MDA (Emerick & Reynolds, 2013) assimilates the data several times with inflated errors, with `sum(1/alpha) = 1`. Here the factors decrease geometrically, `alpha_k = c * ratio**(n - 1 - k)`, so the early updates are damped and the last ones refine.

### The interface

| Method | When the workflow calls it | Returns |
|---|---|---|
| `start(setup)` | once, at the start; `setup` holds the prior `(n_state, N)`, the observations, the data-error covariance and the random streams | a `Step`: the forecasts to run (`ForecastRequest`s) |
| `advance(forecasts)` | after each step's forecasts; `forecasts[name].ok` marks the members that ran, `.predicted` their data `(n_data, n_ok)` | the next `Step`, with the `Record`s of the iterates to save (an empty request list ends the run) |
| `export_state()` | at each checkpoint | an `AlgorithmState`: every array and value the loop holds |
| `restore_state(setup, state)` | on resume | the pending `Step`, as if the run had never stopped |
| `estimate()` | for the posterior | the last kept ensemble and its forecast |

It also declares `status` and `iterative` (`False` for a fixed number of steps; `True` lets cross validation stop it). Two optional methods help: `max_forecasts()` gives the cost in `apex validate`'s plan, and `progress()` feeds the progress bars.

The rules that keep runs exact:

- Draw random numbers only from `setup.streams`, never from `np.random`. The streams are `prior`, `perturbations`, `analysis`, `localization` and `chop`; another name raises `KeyError`. This example uses `perturbations`. APEX checkpoints the streams, so a resumed run continues the same draws.
- Pass every ensemble array through `keep()`, which returns C-contiguous members. Mixed memory layouts round differently.
- Put everything the loop needs into `export_state()`, including a pending update whose forecast has not come back yet.
- Keep the ensemble on the last axis: states `(n_state, N)`, data `(n_data, N)`.

### The code

`apex-geomda/src/apex_geomda/__init__.py`:

```python
"""ES-MDA with a geometric inflation schedule, as an APEX history-matching method.

Emerick & Reynolds (2013), Computers & Geosciences 55, 3-15, with decreasing inflation
factors ``alpha_k = c * ratio**(n - 1 - k)``, scaled so that ``sum(1 / alpha_k) = 1``:
large steps first damp the early updates, small ones refine at the end.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_assimilation import (
    AlgorithmState,
    ApproxUpdate,
    Estimate,
    ForecastRequest,
    Forecasts,
    Record,
    RunSetup,
    Step,
    keep,
)
from apex.algorithms.data_assimilation.protocols import UpdateStrategy
from apex.core.progress import IterationProgress


def geometric_inflation(n_steps: int, ratio: float) -> NDArray[np.float64]:
    """Return ``n_steps`` decreasing inflation factors with ``sum(1 / alpha) = 1``."""
    if n_steps < 1 or not 1 <= ratio < np.inf:
        raise ValueError(f"need n_steps >= 1 and 1 <= ratio < inf, got {n_steps} and {ratio}")
    weights = ratio ** np.arange(n_steps - 1, -1, -1, dtype=np.float64)
    return weights * np.sum(1.0 / weights)


class GeometricESMDA:
    """ES-MDA with geometrically decreasing inflation factors.

    Parameters
    ----------
    n_steps
        Number of assimilation steps.
    ratio
        Ratio of one step's inflation to the next one's (1: ES-MDA's uniform schedule).
    strategy
        The update's linear algebra (``[method.strategy]``); ``approx`` by default.
    lower_bounds, upper_bounds
        State bounds ``(n_state,)`` from ``[state]``, applied after each update.
    """

    #: Shown by `apex plugins`; a plugin under development is "experimental".
    status = "experimental"
    #: A fixed number of steps: cross validation does not stop it.
    iterative = False

    def __init__(
        self,
        n_steps: int = 4,
        ratio: float = 2.0,
        strategy: UpdateStrategy | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> None:
        if int(n_steps) != n_steps:
            raise ValueError(f"n_steps must be a whole number, got {n_steps}")
        self.inflation = geometric_inflation(int(n_steps), float(ratio))
        self.strategy = strategy if strategy is not None else ApproxUpdate()
        # This loop calls the strategy's plain step only: refuse what it cannot drive
        # (a localization to prepare, a prior-increment term, a reference prediction,
        # fidelity levels, simulator gradients).
        needs = (
            "requires_prior_state",
            "requires_reference_prediction",
            "requires_levels",
            "uses_sensitivities",
        )
        if getattr(self.strategy, "localization", None) is not None or any(
            getattr(self.strategy, flag, False) for flag in needs
        ):
            raise ValueError(
                f"{type(self.strategy).__name__} needs more than this method gives it; use "
                "approx without localization, or a method of APEX's that supports it"
            )
        self.lower = None if lower_bounds is None else np.asarray(lower_bounds, float)
        self.upper = None if upper_bounds is None else np.asarray(upper_bounds, float)
        self._arrays: dict[str, NDArray[Any]] = {}
        self._values: dict[str, Any] = {}

    # ------------------------------------------------------------------ the loop

    def start(self, setup: RunSetup) -> Step:
        """Request the prior's forecast; draw the data the steps are scored against."""
        self._setup = setup
        prior = keep(setup.prior)
        n = prior.shape[-1]
        # One draw of perturbed data, never assimilated, scores every step alike.
        scoring = setup.observations.values[:, None] + setup.data_error.perturbations(
            setup.streams["perturbations"], n
        )
        self._arrays = {"state": prior, "member_ids": np.arange(n), "scoring": keep(scoring)}
        self._values = {"step": 0}
        return Step([ForecastRequest("forecast", 0, prior, np.arange(n))], iteration=0)

    def advance(self, forecasts: Forecasts) -> Step:
        """Keep the forecast members that ran, record the iterate, and update again."""
        a, step = self._arrays, self._values["step"]
        forecast = forecasts["forecast"]
        state = a.pop("pending", a["state"])
        a["member_ids"] = a["member_ids"][forecast.ok]
        a["state"] = keep(state, forecast.ok)
        a["scoring"] = keep(a["scoring"], forecast.ok)
        a["predicted"] = keep(forecast.predicted)
        alpha = None if step == 0 else float(self.inflation[step - 1])
        record = Record(
            iteration=step,
            kind="iterate",
            state=a["state"],
            predicted=a["predicted"],
            member_ids=a["member_ids"],
            scoring=a["scoring"],
            damping=alpha,
        )
        if step == len(self.inflation):
            self._values["done"] = True
            return Step([], iteration=step + 1, records=[record])
        return self._update(step, [record])

    def _update(self, step: int, records: list[Record]) -> Step:
        # ES-MDA step k (Emerick & Reynolds 2013), each member j:
        #   m_j <- m_j + C_md (C_dd + alpha_k C_d)^-1 (d_obs + sqrt(alpha_k) e_j - g(m_j))
        a, setup = self._arrays, self._setup
        alpha = float(self.inflation[step])
        n = a["state"].shape[-1]
        perturbed = setup.observations.values[:, None] + np.sqrt(alpha) * (
            setup.data_error.perturbations(setup.streams["perturbations"], n)
        )
        increment = self.strategy.step(
            a["state"], a["predicted"], perturbed, setup.data_error.inflated(alpha)
        )
        updated = a["state"] + increment
        if self.lower is not None or self.upper is not None:
            low = -np.inf if self.lower is None else self.lower[:, None]
            high = np.inf if self.upper is None else self.upper[:, None]
            updated = np.clip(updated, low, high)
        a["pending"] = keep(updated)
        self._values["step"] = step + 1
        request = ForecastRequest("forecast", step + 1, a["pending"], a["member_ids"])
        return Step([request], iteration=step + 1, records=records)

    # ------------------------------------------------------------------ plan, progress

    def max_forecasts(self) -> tuple[int, int]:
        """Return the forecasts a run takes, ``(ensemble, single-member)``, for its plan."""
        return len(self.inflation) + 1, 0

    def progress(self) -> IterationProgress:
        """Return the position in the schedule, for progress bars."""
        step, total = self._values.get("step", 0), len(self.inflation)
        if self._values.get("done"):
            return IterationProgress(total, total, "done")
        # Step k is pending until its forecast comes back.
        return IterationProgress(max(step - 1, 0), total, "prior" if step == 0 else f"step {step}")

    # ------------------------------------------------------------------ checkpoints

    def export_state(self) -> AlgorithmState:
        """Everything the loop holds, so that a resumed run continues bit for bit."""
        return AlgorithmState(dict(self._arrays), dict(self._values))

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Continue from a checkpoint: request the pending update's forecast again."""
        self._setup = setup
        self._arrays = {k: keep(v) for k, v in state.arrays.items()}
        self._values = dict(state.values)
        if "pending" not in self._arrays:  # checkpointed after the last step: finished
            return Step([], iteration=self._values["step"] + 1)
        step = self._values["step"]
        a = self._arrays
        return Step(
            [ForecastRequest("forecast", step, a["pending"], a["member_ids"])], iteration=step
        )

    def estimate(self) -> Estimate:
        """The last kept ensemble and its forecast."""
        a = self._arrays
        return Estimate(a["state"], a["predicted"], a["member_ids"])
```

The update itself comes from APEX: `ApproxUpdate` computes `C_md (C_dd + alpha C_d)^-1 (d - g(m))`. This loop calls only a strategy's plain `step`, so the constructor refuses strategies that need more: a localization (which must be prepared from the prior), the prior-increment term of `full` and `subspace`, a reference prediction, fidelity levels or simulator gradients. APEX's own `esmda` method shows how to drive a localization, and `gies` and `lm_enrml` the prior-increment term. A method that brings its own linear algebra would write it as an update strategy plugin (`apex.update_strategies`), so other methods can use it too.

### Register, install and run it

`pyproject.toml`, as for the metric, with the entry point in `apex.methods`:

```toml
[project.entry-points."apex.methods"]
geometric_esmda = "apex_geomda:GeometricESMDA"
```

```bash
uv pip install -e /path/to/apex-geomda
uv run apex plugins | grep geometric
```

```text
methods            geometric_esmda    [experimental] ES-MDA with geometrically decreasing inflation factors.
```

In `examples/linear_gaussian`, where Step 5 left you, copy the original `config_esmda.toml` (not the metric's copy) to `config_geo.toml`, give its run a name of its own (`[run] name = "linear_geo"`), and replace its `[method]` section with:

```toml
[method]
name = "geometric_esmda"
options = { n_steps = 4, ratio = 2.0 }
strategy = { name = "approx", options = { energy = 1.0 } }
```

```bash
uv run apex validate config_geo.toml      # at most 5 forecasts, 2,500 simulator runs
uv run apex run config_geo.toml
uv run apex report runs/linear_geo
```

The report shows the schedule (in the `damping` column) and the data mismatch falling:

```text
iter  data mismatch mean  data mismatch std*  members  accepted      damping
   0         253.594         225.7      500       yes
   1         48.9056         29.86      500       yes           15
   2         28.9766         14.22      500       yes          7.5
   3         22.2268         10.07      500       yes         3.75
   4         19.2559         8.226      500       yes         1.88
```

### Check that it resumes exactly

Stop a run early, resume it, and compare it with the uninterrupted run. Copy `config_geo.toml` to `config_geo_resumed.toml` with `[run] name = "linear_geo_resumed"`, then:

```bash
uv run apex run config_geo_resumed.toml --max-steps 2
uv run apex resume runs/linear_geo_resumed
```

Every checkpoint must be identical to the uninterrupted run's:

```python
import numpy as np

for k in range(5):
    a = np.load(f"runs/linear_geo/checkpoints/iter_{k}.npz")
    b = np.load(f"runs/linear_geo_resumed/checkpoints/iter_{k}.npz")
    assert all(np.array_equal(a[f], b[f]) for f in a.files)
```

### Test it

`tests/test_geomda.py`:

```python
import numpy as np
import pytest

from apex.algorithms.data_assimilation import methods
from apex_geomda import geometric_inflation


def test_the_schedule_decreases_and_its_reciprocals_sum_to_one() -> None:
    alpha = geometric_inflation(4, 2.0)
    np.testing.assert_allclose(alpha, [15.0, 7.5, 3.75, 1.875])
    np.testing.assert_allclose(np.sum(1.0 / alpha), 1.0)
    np.testing.assert_allclose(geometric_inflation(3, 1.0), [3.0, 3.0, 3.0])  # ES-MDA's


def test_a_strategy_this_loop_cannot_drive_is_refused() -> None:
    from apex.algorithms.data_assimilation import update_strategies

    with pytest.raises(ValueError, match="needs more than this method gives it"):
        methods.create("geometric_esmda", strategy=update_strategies.create("full"))


def test_the_method_is_found_through_its_entry_point() -> None:
    method = methods.create("geometric_esmda", n_steps=3, ratio=3.0)
    assert method.max_forecasts() == (4, 0)
```

Run them from the APEX folder, as the metric's:

```bash
uv run pytest /path/to/apex-geomda/tests
```

A method needs an independent reference. On the linear-Gaussian example the exact posterior is known: `make_case.py` writes the case, and the posterior mean and spread of a large ensemble should approach the Kalman posterior. Compare the run's RMSE (in `apex summary`) with ES-MDA's on the same case, and test the posterior against the exact one as APEX's own method tests do.

## Contributing it to APEX itself

A plugin can also become part of APEX. In APEX's own code, plugins register with their registry's decorator instead of an entry point, e.g. `@metrics.register("mae")` (`metrics` from `apex.core.plugins`), in a module that APEX's composition imports. A contribution should have:

- typed code with NumPy-style docstrings, and the ensemble on the last axis;
- tests, including an independent reference for a numerical method (an analytic solution, a published benchmark or convergence on a toy problem), and a test that reproduces any bug it fixes;
- a short description of the method and its reference paper;
- passing `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run lint-imports && uv run pytest`.

Open a pull request with it, or an issue to discuss the idea first.

## Pitfalls

- **`unknown metrics plugin 'mae'`:** the package is not installed in the environment that runs APEX, or the entry-point group is misspelt (`apex.metrics`, plural, as the registry). Check with `uv run python -c "from importlib.metadata import entry_points; print(entry_points(group='apex.metrics'))"`.
- **The plugin vanished after `uv sync`:** see Step 4.
- **`already registered`:** another plugin has the name; rename your entry point.
- **The metric is missing from the run:** `[[metrics]]` replaces the defaults, so list every metric you want.

## What to read next

- [The plugin reference](../reference/plugins/methods.md): every registry's plugins and their options.
- [The configuration reference](../reference/configuration/history_match.md): where each kind of plugin is named in a configuration.
- [Examples](../examples.md): add an example that uses your plugin, so others can try it from the web UI.
