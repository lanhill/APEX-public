"""OPM Flow simulator adapter: render a Mako deck per member, run ``flow``, read the summary.

Requires the ``reservoir`` extra (``mako``, ``resdata``) and an OPM Flow executable.

The deck template, the data keys and the summary reading are shared with the Eclipse
adapter (:mod:`apex.simulation.deck`).
"""

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

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
from apex.core.simulation import CommandRunner, MemberInput, MemberSeries, SeriesRequest
from apex.domain.observations import DataKey
from apex.simulation.deck import (
    DECK_NAME,
    FidelityLevels,
    case_dir_files,
    failure_error,
    is_summary_file,
    prune,
    read_series,
    read_summary,
    render_deck,
    tail,
)

__all__ = ["DECK_NAME", "OPMFlowSimulator", "read_summary"]


@simulators.register("opm_flow")
class OPMFlowSimulator:
    """Runs OPM Flow for one member in its work directory.

    Parameters
    ----------
    deck_template_file
        Mako template of the ``.DATA`` deck.
    executable
        OPM Flow command (default ``flow``, looked up on ``PATH``).
    flow_args
        Extra command-line arguments, e.g. ``["--threads-per-process=1"]``.
    timeout_s
        Wall-clock limit per run; exceeding it fails the member.
    keep_files
        Keep the member's deck and outputs (``False`` deletes everything but the log and
        the summary files after reading, to save disk space).
    template_values
        Extra fixed values passed to the template.
    case_dir
        The folder ``${case_dir}`` names in the template (its ``INCLUDE`` paths), for every
        level; default the template's folder. A copied example sets it to its own folder,
        so that its data files are read while the template stays in the examples root.
    transient_patterns
        Extra case-insensitive regular expressions: a failed run whose log matches one is
        transient, and the executor retries it (on top of the built-in licence and file-system
        patterns, ``apex.simulation.deck.TRANSIENT_PATTERNS``).

    Raises
    ------
    ValueError
        If the template does not exist or ``timeout_s`` is not positive.
    """

    #: The workflow gives each run its own work directory.
    needs_work_dir = True
    #: Runs Flow through a member's ``runner`` when it has one (another host).
    accepts_runner = True

    def __init__(
        self,
        deck_template_file: CodeFile,
        executable: Program = "flow",
        flow_args: ProgramArguments = (),
        timeout_s: float = 3600.0,
        keep_files: bool = True,
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
        # Multilevel ensembles: OPM Flow has no COARSEN, so levels give their
        # own coarse decks (deck_template_file) instead of upscale.
        self.levels = FidelityLevels(levels, self.template_path)
        if self.levels.upscales:
            raise ValueError(
                "OPM Flow does not support COARSEN: give each coarse level its own "
                "deck_template_file instead of upscale (or use the Eclipse simulator)"
            )
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self.executable = executable
        self.flow_args = tuple(flow_args)
        self.timeout_s = timeout_s
        self.keep_files = keep_files
        self.template_values = dict(template_values or {})
        self.transient_patterns = tuple(transient_patterns)

    @property
    def resources(self) -> dict[str, int]:
        """Machine-wide tokens one run holds: its cores (``OMP_NUM_THREADS``, default 1)."""
        return {"cpu": max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))}

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Render the deck, run Flow and return the summary values at ``keys``."""
        case = self._simulate(member)
        values = read_summary(case, keys)
        self._clean(case.parent)
        return values

    def run_series(self, member: MemberInput, request: SeriesRequest) -> MemberSeries:
        """Render the deck, run Flow and return the requested series."""
        case = self._simulate(member)
        series = read_series(case, request)
        self._clean(case.parent)
        return series

    def simulate(self, member: MemberInput) -> Path:
        """Render the deck and run Flow; return the case path (no extension).

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
            raise SimulationError("OPMFlowSimulator needs a work directory")
        work_dir = member.work_dir
        deck = work_dir / f"{DECK_NAME}.DATA"
        deck.write_text(self._render(member))
        if member.runner is None:
            self._run_flow(deck, work_dir)
        else:
            self._run_elsewhere(member.runner, deck, work_dir)
        return work_dir / DECK_NAME

    def _clean(self, work_dir: Path) -> None:
        if not self.keep_files:
            prune(work_dir, _keep)

    def version(self) -> str | None:
        """Return ``flow --version`` (for provenance), or ``None`` if it cannot be run."""
        executable = shutil.which(self.executable)
        if executable is None:
            return None
        try:
            completed = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, timeout=60, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        lines = completed.stdout.strip().splitlines()
        return lines[0] if lines else None

    def _render(self, member: MemberInput) -> str:
        template, level_values = self.levels.for_member(member)
        case_dir = self.case_dir
        if member.runner is not None:
            # The deck's ${case_dir} includes, where Flow sees them on the runner's host.
            case_dir = member.runner.static_folder(
                case_dir or template.resolve().parent, case_dir_files(template)
            )
        return render_deck(template, member, {**self.template_values, **level_values}, case_dir)

    def _run_elsewhere(self, runner: CommandRunner, deck: Path, work_dir: Path) -> None:
        """Run Flow through ``runner``, with paths relative to ``work_dir``."""
        log = work_dir / "flow.log"
        command = [self.executable, deck.name, "--output-dir=.", *self.flow_args]
        env = {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
        code = runner.run(command, work_dir, env=env, timeout_s=self.timeout_s, log=log)
        if code is None:
            raise SimulationError(f"flow exceeded {self.timeout_s} s in {work_dir}")
        if code != 0:
            raise failure_error(
                f"flow exited with {code} in {work_dir}; last output:\n" + tail(log),
                code,
                log,
                self.transient_patterns,
            )

    def _run_flow(self, deck: Path, work_dir: Path) -> None:
        executable = shutil.which(self.executable)
        if executable is None:
            raise SimulationError(f"OPM Flow executable {self.executable!r} not found on PATH")
        log = work_dir / "flow.log"
        command = [executable, str(deck), f"--output-dir={work_dir}", *self.flow_args]
        # One OpenMP thread per run unless asked otherwise: the executor runs members in
        # parallel, and oversubscription slows everything down.
        env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
        try:
            with log.open("w") as handle:
                completed = subprocess.run(
                    command,
                    cwd=work_dir,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout_s,
                    env=env,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            raise SimulationError(f"flow exceeded {self.timeout_s} s in {work_dir}") from error
        if completed.returncode != 0:
            raise failure_error(
                f"flow exited with {completed.returncode} in {work_dir}; last output:\n"
                + tail(log),
                completed.returncode,
                log,
                self.transient_patterns,
            )


def _keep(path: Path) -> bool:
    # Keep the log, the deck and every summary file (unified .SMSPEC/.UNSMRY, or the
    # non-unified .Snnnn that Flow writes by default, and .ESMRY).
    return path.name == "flow.log" or path.suffix.upper() == ".DATA" or is_summary_file(path)
