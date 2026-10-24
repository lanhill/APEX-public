"""Eclipse simulator adapter: render a Mako deck per member, run ``eclrun``, read the summary.

Requires the ``reservoir`` extra (``mako``, ``resdata``), an Eclipse installation with
``eclrun`` and licences. The deck template, the data keys and the summary reading are
those of the OPM Flow adapter (:mod:`apex.simulation.deck`), so a case moves between the
two simulators by changing ``[simulator] name``.

Ported from PET
---------------
PET module : SimulatorWrap ``subsurface/multphaseflow/eclipse.py`` (``ecl_100.call_sim``,
             ``ecl_300.call_sim``) and ``misc/system_tools/environ_var.py``
             (``EclipseRunEnvironment``), SimulatorWrap commit ``d29dc57`` (as installed
             with PET ``e65fb10``).
Behaviour  : ``eclrun [args] eclipse|e300 MODEL.DATA`` in the member's directory. A run
             has failed when ``eclrun`` exits non-zero, or when the ``.ECLEND`` file's
             "Error summary" counts errors or bugs (``eclrun`` can exit 0 after a failed
             run), or when there is no error summary at all (the run did not finish).
Differences: PET calls ``sys.exit(1)`` from inside its context manager on errors, and a
             bare ``except`` turns every other exception into a failed run. Here a failed
             run raises :class:`SimulationError` with the first errors of the ``.PRT``
             file, and the workflow's failure policy drops the member. On a timeout the
             whole process group is killed, so no orphaned Eclipse process keeps holding a
             licence. ``licenses`` caps how many runs execute at once
             (``max_parallel_runs``), whatever the executor's ``workers``.
"""

import os
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.options import (
    LEVELS,
    TEXT,
    CodeFile,
    InputFolder,
    Program,
    ProgramArguments,
    TextList,
)
from apex.core.plugins import simulators
from apex.core.simulation import MemberInput, MemberSeries, SeriesRequest
from apex.domain.observations import DataKey
from apex.simulation.deck import (
    DECK_NAME,
    FidelityLevels,
    failure_error,
    is_summary_file,
    prune,
    read_series,
    read_summary,
    render_deck,
    tail,
)

#: ``eclrun`` program of each Eclipse version.
PROGRAMS = {"e100": "eclipse", "e300": "e300"}

#: Seconds between SIGTERM and SIGKILL when a timed-out run is stopped.
KILL_GRACE_S = 10.0

LOG_NAME = "eclrun.log"


@simulators.register("eclipse")
class EclipseSimulator:
    """Runs Eclipse (E100 or E300) through ``eclrun`` for one member in its work directory.

    Parameters
    ----------
    deck_template_file
        Mako template of the ``.DATA`` deck (as for ``opm_flow``).
    version
        ``"e100"`` (black oil, ``eclrun eclipse``) or ``"e300"`` (compositional).
    executable
        The ``eclrun`` command (default ``eclrun``, looked up on ``PATH``).
    eclrun_args
        Extra ``eclrun`` arguments before the program, e.g. ``["--version=2023.1"]`` or
        ``["--nocleanup"]`` (PET's default).
    timeout_s
        Wall-clock limit per run; exceeding it kills the run and fails the member.
    keep_files
        Keep the member's deck and outputs. ``False`` deletes everything but the log, the
        deck, the ``.PRT``/``.ECLEND`` reports and the summary files after reading.
    licenses
        The number of Eclipse licences available. At most this many members run at once
        (the process executor honours ``max_parallel_runs``); ``None`` means no cap.
    template_values
        Extra fixed values passed to the template.
    case_dir
        The folder ``${case_dir}`` names in the template (its ``INCLUDE`` paths), for every
        level; default the template's folder (see :class:`OPMFlowSimulator`).
    transient_patterns
        Extra case-insensitive regular expressions: a failed run whose log matches one is
        transient, and the executor retries it (on top of the built-in licence and file-system
        patterns, ``apex.simulation.deck.TRANSIENT_PATTERNS``).

    Raises
    ------
    ValueError
        If the template does not exist, or an option is out of range.
    """

    #: The workflow gives each run its own work directory.
    needs_work_dir = True

    def __init__(
        self,
        deck_template_file: CodeFile,
        version: Literal["e100", "e300"] = "e100",
        executable: Program = "eclrun",
        eclrun_args: ProgramArguments = (),
        timeout_s: float = 3600.0,
        keep_files: bool = True,
        licenses: int | None = None,
        template_values: Annotated[Mapping[str, object] | None, TEXT] = None,
        transient_patterns: TextList = (),
        levels: Annotated[Sequence[Mapping[str, object]] | None, LEVELS] = None,
        case_dir: InputFolder | None = None,
    ) -> None:
        self.template_path = Path(deck_template_file).resolve()
        if not self.template_path.is_file():
            raise ValueError(f"deck template not found: {self.template_path}")
        self.case_dir = None if case_dir is None else Path(case_dir).resolve()
        if self.case_dir is not None and not self.case_dir.is_dir():
            raise ValueError(f"case_dir is not a folder: {self.case_dir}")
        # Multilevel ensembles: the composition passes the method's levels.
        self.levels = FidelityLevels(levels, self.template_path)
        if version not in PROGRAMS:
            raise ValueError(f"version must be one of {sorted(PROGRAMS)}, got {version!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        if licenses is not None and (
            isinstance(licenses, bool) or not isinstance(licenses, int) or licenses < 1
        ):
            raise ValueError(f"licenses must be a positive integer, got {licenses!r}")
        # The Eclipse variant, not a software version: `version` is the configuration option's
        # name, but the attribute must not be, since services read `simulator.version()` as the
        # optional software-version capability (identity, provenance, predictions).
        self.variant = version
        self.executable = executable
        self.eclrun_args = tuple(eclrun_args)
        self.timeout_s = timeout_s
        self.keep_files = keep_files
        self.max_parallel_runs = licenses
        self.template_values = dict(template_values or {})
        self.transient_patterns = tuple(transient_patterns)

    @property
    def resources(self) -> dict[str, int]:
        """Machine-wide tokens one run holds: one licence and one core."""
        return {"eclipse": 1, "cpu": 1}

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Render the deck, run Eclipse and return the summary values at ``keys``."""
        case = self._simulate(member)
        values = read_summary(case, keys)
        self._clean(case.parent)
        return values

    def run_series(self, member: MemberInput, request: SeriesRequest) -> MemberSeries:
        """Render the deck, run Eclipse and return the requested series."""
        case = self._simulate(member)
        series = read_series(case, request)
        self._clean(case.parent)
        return series

    def simulate(self, member: MemberInput) -> Path:
        """Render the deck and run Eclipse; return the case path (no extension).

        With :meth:`read` and :meth:`clean`, the file-based capability a model-error
        simulator wraps: the outputs stay until :meth:`clean`.
        """
        return self._simulate(member)

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the summary values of a simulated ``case`` at ``keys``."""
        return read_summary(case, keys)

    def clean(self, case: Path) -> None:
        """Prune the case's work directory as ``keep_files`` says."""
        self._clean(case.parent)

    def _simulate(self, member: MemberInput) -> Path:
        if member.work_dir is None:
            raise SimulationError("EclipseSimulator needs a work directory")
        work_dir = member.work_dir
        deck = work_dir / f"{DECK_NAME}.DATA"
        template, level_values = self.levels.for_member(member)
        values = {**self.template_values, **level_values}
        deck.write_text(render_deck(template, member, values, self.case_dir))
        self._run_eclipse(work_dir)
        try:
            check_run(work_dir / DECK_NAME)
        except SimulationError as error:
            # Eclipse may report a licence failure in its PRT and still exit with 0.
            prt = (work_dir / DECK_NAME).with_suffix(".PRT")
            raise failure_error(str(error), None, prt, self.transient_patterns) from error
        return work_dir / DECK_NAME

    def _clean(self, work_dir: Path) -> None:
        if not self.keep_files:
            prune(work_dir, _keep)

    def _run_eclipse(self, work_dir: Path) -> None:
        executable = shutil.which(self.executable)
        if executable is None:
            raise SimulationError(f"eclrun executable {self.executable!r} not found on PATH")
        command = [
            executable,
            *self.eclrun_args,
            PROGRAMS[self.variant],
            f"{DECK_NAME}.DATA",
        ]
        log = work_dir / LOG_NAME
        with log.open("w") as handle:
            # A new session makes eclrun and the simulator it starts one process group,
            # so a timeout can stop them all and release the licence.
            process = subprocess.Popen(
                command,
                cwd=work_dir,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as error:
                _stop(process)
                raise SimulationError(
                    f"eclipse exceeded {self.timeout_s} s in {work_dir}"
                ) from error
        if returncode != 0:
            raise failure_error(
                f"eclrun exited with {returncode} in {work_dir}; last output:\n" + tail(log),
                returncode,
                log,
                self.transient_patterns,
            )


def check_run(case: Path) -> None:
    """Check that the Eclipse run of ``case`` (path without extension) finished cleanly.

    Reads the "Error summary" of ``<case>.ECLEND``, as PET's ``EclipseRunEnvironment``.

    Raises
    ------
    SimulationError
        If the summary counts errors or bugs (with the first errors of ``<case>.PRT``),
        or there is no error summary (the run did not finish).
    """
    counts = error_summary(case.with_suffix(".ECLEND"))
    if counts is None:
        raise SimulationError(
            f"eclipse did not finish {case.name} (no error summary in {case.name}.ECLEND); "
            f"see {case.parent / LOG_NAME}"
        )
    if counts.get("Errors", 0) > 0 or counts.get("Bugs", 0) > 0:
        messages = prt_errors(case.with_suffix(".PRT"))
        raise SimulationError(
            f"eclipse run {case} failed with {counts.get('Errors', 0):g} error(s) and "
            f"{counts.get('Bugs', 0):g} bug(s)" + (":\n" + "\n".join(messages) if messages else "")
        )


def error_summary(path: Path) -> dict[str, float] | None:
    """Return the counts of an ``.ECLEND`` "Error summary" (``Errors``, ``Bugs``, ...).

    ``None`` means the file or its summary is missing (the run did not finish).
    """
    try:
        lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
    except OSError:
        return None
    try:
        start = lines.index("Error summary")
    except ValueError:
        return None
    counts = {}
    for line in lines[start + 1 : start + 6]:
        match = re.fullmatch(r"(\w+)\s+([0-9.]+)", line)
        if match:
            counts[match.group(1)] = float(match.group(2))
    return counts


def prt_errors(path: Path, limit: int = 3) -> list[str]:
    """Return the first ``limit`` error messages of an Eclipse ``.PRT`` file (``@--  ERROR``)."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    messages: list[str] = []
    i = 0
    while i < len(lines) and len(messages) < limit:
        if re.match(r"\s*@--\s*(ERROR|BUG)", lines[i]):
            block = [lines[i].strip()]
            i += 1
            while i < len(lines) and re.match(r"\s*@\s", lines[i]):
                block.append(lines[i].strip().lstrip("@").strip())
                i += 1
            messages.append(" ".join(block))
        else:
            i += 1
    return messages


def _stop(process: "subprocess.Popen[bytes]") -> None:
    """Terminate a run's process group, then kill it if it does not stop."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass  # already gone


def _keep(path: Path) -> bool:
    return (
        path.name == LOG_NAME
        or path.suffix.upper() in {".DATA", ".PRT", ".ECLEND"}
        or is_summary_file(path)
    )
