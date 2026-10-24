"""ERT as a run engine: ERT's own smoothers as cells of APEX studies.

A run is three steps, in ``<run_dir>.ert/`` beside the run directory:

1. **export** the cell's configuration (A1, :func:`~.export.export_case`) to
   ``<run_dir>.ert/case/``: APEX's prior for the cell's seed as ``INIT_FILES``,
   ``RANDOM_SEED`` = the seed, ``MAX_RUNNING`` = the workers;
2. **run** ERT's run model (``ensemble_smoother`` or ``es_mda`` with ``weights``) as a
   subprocess, reporting each realization as it finishes (:class:`RunPathProgress`), and
   holding the machine-wide tokens its forward runs need (the simulator's ``resources`` x
   ``MAX_RUNNING``) for the whole run, so ERT's local queue never exceeds what APEX would
   run;
3. **import** (A2, :func:`~.importer.import_case`) into the run directory, which then holds
   an APEX run like any other.

ERT has no resume: a stopped or failed run starts again from the export. The run's
``provenance.json`` records the ERT version and the run model.

Registered through the entry point ``apex.engines`` (``ert``); ERT itself is imported only
when a run starts, so without the ``ert`` extra :meth:`ErtEngine.check` says so.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import tomli_w

from apex.core.engines import EngineContext, EngineMethod
from apex.core.errors import ConfigError, SimulationError
from apex.core.plugins import simulators
from apex.core.progress import ProgressReporter
from apex.domain.config import HistoryMatchConfig
from apex.domain.results import HistoryMatchResult

#: ERT's run models an ERT cell may name, and the options each takes.
RUN_MODELS: dict[str, frozenset[str]] = {
    "ensemble_smoother": frozenset(),
    "es_mda": frozenset({"weights"}),
}
#: ERT's default ES-MDA weights (``ert es_mda`` without ``--weights``).
DEFAULT_WEIGHTS = (4.0, 2.0, 1.0)


#: Seconds between looks at the run paths while ERT runs.
POLL_S = 2.0


class RunPathProgress:
    """Reports ERT's realizations as they finish, from the files ERT leaves in run paths.

    ERT's forward-model runner writes ``OK`` (or ``ERROR``) in a realization's run path when
    it ends. Each iteration is reported as one forecast; each realization, once, as a
    member. Progress never affects the run.

    Parameters
    ----------
    runs
        The case's ``runs/`` folder (``realization-<i>/iter-<k>``).
    reporter
        Where to report; ``None`` for nowhere.
    label
        The run model, for the forecasts' labels.
    """

    def __init__(self, runs: Path, reporter: ProgressReporter | None, label: str) -> None:
        self.runs = runs
        self.reporter = reporter
        self.label = label
        self.seen: dict[int, dict[int, bool]] = {}
        self.current: int | None = None
        self.started = 0.0

    def poll(self) -> None:
        """Report what finished since the last poll."""
        if self.reporter is None or not self.runs.exists():
            return
        for path in sorted(self.runs.glob("realization-*/iter-*")):
            iteration = int(path.name.split("-", 1)[1])
            realization = int(path.parent.name.split("-", 1)[1])
            if iteration not in self.seen:
                self._start(iteration)
            done = self.seen[iteration]
            if realization in done:
                continue
            if (path / "OK").exists() or (path / "ERROR").exists():
                ok = (path / "OK").exists()
                done[realization] = ok
                if iteration == self.current:
                    self.reporter.member_finished(member_id=realization, ok=ok)

    def close(self) -> None:
        """Finish the last forecast."""
        self._finish()

    def _start(self, iteration: int) -> None:
        assert self.reporter is not None
        self._finish()
        self.seen[iteration] = {}
        self.current = iteration
        self.started = time.monotonic()
        n = len(list(self.runs.glob(f"realization-*/iter-{iteration}")))
        self.reporter.forecast_started(
            iteration=iteration, label=f"ERT {self.label}, iteration {iteration}", n_members=n
        )

    def _finish(self) -> None:
        if self.reporter is None or self.current is None:
            return
        done = self.seen[self.current]
        self.reporter.forecast_finished(
            n_ok=sum(done.values()),
            n_failed=len(done) - sum(done.values()),
            seconds=time.monotonic() - self.started,
        )
        self.current = None


def _ert_version() -> str | None:
    try:
        return metadata.version("ert")
    except metadata.PackageNotFoundError:
        return None


def _weights(method: EngineMethod) -> tuple[float, ...]:
    weights = method.options.get("weights", DEFAULT_WEIGHTS)
    if isinstance(weights, str):
        weights = [w for w in weights.split(",") if w.strip()]
    return tuple(float(w) for w in weights)


class ErtEngine:
    """ERT's own ES and ES-MDA on an APEX case (see the module docstring)."""

    name = "ert"

    def check(self, config: HistoryMatchConfig, method: EngineMethod) -> list[str]:
        """Return why ERT cannot run ``method`` on the case (empty if it can)."""
        from apex.integrations.ert.export import _check_exportable

        problems = []
        if _ert_version() is None:
            problems.append(
                "the ert engine needs the `ert` extra (ERT 26.x): pip install apex[ert]"
            )
        allowed = RUN_MODELS.get(method.name)
        if allowed is None:
            problems.append(
                f"ERT run model {method.name!r} is not one of {', '.join(sorted(RUN_MODELS))}"
            )
        else:
            unknown = sorted(set(method.options) - allowed)
            if unknown:
                problems.append(f"ERT's {method.name} takes no options {unknown}")
            elif method.name == "es_mda":
                try:
                    weights = _weights(method)
                except (TypeError, ValueError):
                    weights = ()
                if not weights or any(w <= 0 for w in weights):
                    problems.append("es_mda weights must be positive numbers, e.g. [4, 4, 4, 4]")
        try:
            _check_exportable(config)
        except ConfigError as error:
            problems.append(str(error))
        return problems

    def simulator_runs(self, method: EngineMethod, members: int | None) -> int | None:
        """Return ``N x (updates + 1)``: the prior and every update's posterior are simulated."""
        updates = len(_weights(method)) if method.name == "es_mda" else 1
        return None if members is None else members * (updates + 1)

    def run(
        self, config: HistoryMatchConfig, method: EngineMethod, context: EngineContext
    ) -> HistoryMatchResult:
        """Export, run ERT and import into ``context.run_dir``.

        Raises
        ------
        ConfigError
            If ERT cannot run the case or the method.
        SimulationError
            If ERT fails.
        """
        from apex.execution.resources import FilePools
        from apex.integrations.ert.export import export_case
        from apex.integrations.ert.importer import import_case

        problems = self.check(config, method)
        if problems:
            raise ConfigError("; ".join(problems))
        run_dir = context.run_dir.resolve()
        work = run_dir.parent / f"{run_dir.name}.ert"
        # No resume in ERT: a run starts again from its export.
        shutil.rmtree(work, ignore_errors=True)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        work.mkdir(parents=True)
        config_file = work / "config.toml"
        document = json.loads(config.model_dump_json(exclude_none=True))
        document["run"]["output_dir"] = str(run_dir)
        config_file.write_text(tomli_w.dumps(document))
        case_dir = work / "case"
        export_case(config_file, case_dir)
        command = self._command(method, case_dir)
        workers = int(config.execution.options.get("workers", 1))
        simulator = simulators.create(config.simulator.name, **config.simulator.options)
        needs = {
            pool: n * workers for pool, n in (getattr(simulator, "resources", None) or {}).items()
        }
        environment = {
            **os.environ,
            # ERT's local queue runs fm_dispatch.py, and APEX's deck step this Python.
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]),
        }
        environment.pop("PYTHONPATH", None)
        log = work / "ert.log"
        progress = RunPathProgress(case_dir / "runs", context.progress, method.name)
        with FilePools().hold(needs), log.open("w") as handle:
            process = subprocess.Popen(
                command, cwd=case_dir, env=environment, stdout=handle,
                stderr=subprocess.STDOUT,
            )  # fmt: skip
            while process.poll() is None:
                progress.poll()
                time.sleep(POLL_S)
            progress.poll()
            progress.close()
        if process.returncode != 0:
            tail = log.read_text(errors="replace")[-2000:]
            raise SimulationError(
                f"ERT {method.name} failed ({process.returncode}); {log}:\n{tail}"
            )
        result = import_case(case_dir, run_dir)
        self._provenance(run_dir, method, command)
        return result

    @staticmethod
    def _command(method: EngineMethod, case_dir: Path) -> list[str]:
        path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
        executable = shutil.which("ert", path=path)
        if executable is None:
            raise ConfigError("`ert` is not on PATH: install apex[ert] (ERT 26.x)")
        command = [executable, method.name]
        if method.name == "es_mda":
            command += ["--weights", ",".join(f"{w:g}" for w in _weights(method))]
        return [*command, "case.ert"]

    @staticmethod
    def _provenance(run_dir: Path, method: EngineMethod, command: list[str]) -> None:
        path = run_dir / "provenance.json"
        record: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        record["engine"] = {
            **method.as_json(),
            "ert_version": _ert_version(),
            "command": command[1:],
        }
        path.write_text(json.dumps(record, indent=2))
