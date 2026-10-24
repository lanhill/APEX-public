"""Draw figure plugins and presets on a run.

The plotter agent, the CLI and the REST route call :func:`draw`: each request's figure is
created by name from :data:`apex.core.plugins.figures`, checked against the run
(``applies``) and drawn into its subfolder. ``summary.json`` in the output folder lists what
was drawn and what was skipped, and why.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from apex.core.errors import ConfigError, PluginError
from apex.core.figures import FigureRequest, RunView
from apex.core.plugins import figure_presets, figures
from apex.services.figure_view import run_view

logger = logging.getLogger(__name__)

#: Where the plotter writes, under the run directory (the web UI shows ``report/``).
DEFAULT_SUBDIR = Path("report") / "plots"


@dataclass
class Drawn:
    """One request's outcome: the PNG files written, or why it was skipped."""

    figure: str
    subdir: str
    options: dict[str, Any]
    files: list[str] = field(default_factory=list)
    skipped: str | None = None


@dataclass
class DrawResult:
    """What a :func:`draw` call wrote, under ``output``."""

    output: Path
    drawn: list[Drawn]

    @property
    def files(self) -> list[str]:
        """Every PNG written, relative to the output folder."""
        return [f for d in self.drawn for f in d.files]

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary (also written as ``summary.json``)."""
        return {
            # The folder's name only: summary.json may be served to a web-UI user.
            "output": self.output.name,
            "figures": [asdict(d) for d in self.drawn],
            "written": len(self.files),
            "skipped": [
                {"figure": d.figure, "subdir": d.subdir, "reason": d.skipped}
                for d in self.drawn
                if d.skipped
            ],
        }


def check_request(request: FigureRequest, view: RunView | None = None) -> str | None:
    """Return why a request cannot be drawn (unknown figure, bad options, not for this run)."""
    if request.figure not in figures.names():
        return f"unknown figure {request.figure!r}; known: {sorted(figures.names())}"
    try:
        figure = figures.create(request.figure, **dict(request.options))
    except (PluginError, TypeError, ValueError) as error:
        return f"{request.figure}: {error}"
    if Path(request.subdir).is_absolute() or ".." in Path(request.subdir).parts:
        return f"{request.figure}: subdir {request.subdir!r} must be a relative folder"
    return None if view is None else figure.applies(view)


def preset_requests(
    name: str, view: RunView, options: Mapping[str, Any] | None = None
) -> list[FigureRequest]:
    """Return a preset's requests for a run.

    Raises
    ------
    ConfigError
        If the preset is unknown or its options are invalid.
    """
    if name not in figure_presets.names():
        raise ConfigError(f"unknown preset {name!r}; known: {sorted(figure_presets.names())}")
    try:
        preset = figure_presets.create(name, **dict(options or {}))
    except (PluginError, TypeError, ValueError) as error:
        raise ConfigError(f"preset {name}: {error}") from error
    return preset.requests(view)


#: The largest ``dpi_scale``: a 20 x 16 in box plot at 500 dpi is already 80 Mpixel.
MAX_DPI_SCALE = 2.0


def draw(
    run_path: Path,
    requests: Sequence[FigureRequest] = (),
    *,
    preset: str | None = None,
    output_dir: Path | None = None,
    dpi_scale: float = 1.0,
    view: RunView | None = None,
    locked: bool = False,
) -> DrawResult:
    """Draw ``requests`` (and a preset's) on the run in ``run_path``.

    The output folder's lock is taken while drawing (``locked``: the caller holds it, as
    the background process does), so a foreground and a background drawing never write
    into one folder at once.

    The output defaults to ``<run>/report/plots``. A request that cannot be drawn is
    skipped with its reason; none is drawn twice. Every file stays inside the output.

    Raises
    ------
    ConfigError
        If the run cannot be read, the preset is unknown, ``dpi_scale`` is out of range,
        a figure wrote outside the output folder, or the folder is being drawn into.
    """
    import os

    if not 0 < dpi_scale <= MAX_DPI_SCALE:
        raise ConfigError(f"dpi_scale must be in (0, {MAX_DPI_SCALE:g}], got {dpi_scale}")
    output = (output_dir or Path(run_path) / DEFAULT_SUBDIR).resolve()
    fd = None
    if not locked:
        output.mkdir(parents=True, exist_ok=True)
        fd = _try_lock(output)
        if fd is None:
            raise ConfigError("this run's figures are being drawn: try again when they are done")
    try:
        return _draw(run_path, requests, preset, output, dpi_scale, view)
    finally:
        if fd is not None:
            os.close(fd)


def _draw(
    run_path: Path,
    requests: Sequence[FigureRequest],
    preset: str | None,
    output: Path,
    dpi_scale: float,
    view: RunView | None,
) -> DrawResult:
    view = view or run_view(run_path)
    chosen = list(requests) + (preset_requests(preset, view) if preset else [])
    drawn: list[Drawn] = []
    seen: set[str] = set()
    # Output inside the run (the default): drawing stops if the run is deleted meanwhile.
    inside_run = output.is_relative_to(Path(run_path).resolve())
    for request in chosen:
        key = json.dumps([request.figure, dict(request.options), request.subdir], sort_keys=True,
                         default=str)  # fmt: skip
        if key in seen:
            continue
        seen.add(key)
        outcome = Drawn(request.figure, request.subdir, dict(request.options))
        drawn.append(outcome)
        reason = check_request(request, view)
        if reason is not None:
            outcome.skipped = _shown(reason, run_path)
            continue
        if inside_run and not Path(run_path).is_dir():
            outcome.skipped = "the run was deleted"
            break  # nothing is written for a deleted run (its folder is not recreated)
        figure = figures.create(request.figure, **dict(request.options))
        folder = (output / request.subdir).resolve()
        if not folder.is_relative_to(output):
            outcome.skipped = f"{request.figure}: its folder leads outside the output"
            continue
        if not output.is_dir():
            outcome.skipped = "the output folder was removed"
            break
        try:
            folder.mkdir(parents=True, exist_ok=True)
            pngs = figure.draw(view, folder, dpi_scale=dpi_scale)
        except (KeyError, ValueError, OSError) as error:
            outcome.skipped = _shown(f"{request.figure}: {error}", run_path)
            logger.warning(
                "figure skipped",
                extra={"event": "figure_skipped", "figure": request.figure, "error": str(error)},
            )
            continue
        outside = [p.name for p in pngs if not p.resolve().is_relative_to(output)]
        if outside:
            raise ConfigError(f"{request.figure} wrote outside its output folder: {outside}")
        outcome.files = [str(p.resolve().relative_to(output)) for p in pngs]
    result = DrawResult(output, drawn)
    if inside_run and not Path(run_path).is_dir():
        return result  # the run was deleted meanwhile: its folder is not recreated
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"state": "done", **result.summary()}, indent=1, default=str)
    )
    return result


def _shown(reason: str, run_path: Path) -> str:
    """Return a reason without the run's server path (it reaches the web UI and the chat)."""
    return reason.replace(str(Path(run_path).resolve()), "<run>").replace(str(run_path), "<run>")


#: Held by the drawing process (``start_drawing``) until it exits, crash or not.
LOCK_FILE = "plot.lock"


def _try_lock(output: Path, name: str = LOCK_FILE) -> int | None:
    """Return an open, exclusively flocked file descriptor on ``output/name``, or ``None``."""
    import fcntl
    import os

    fd = os.open(output / name, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        # Closed on every failure: a leaked descriptor would hold the lock for the
        # service's lifetime, and the run would stay "drawing".
        os.close(fd)
        if isinstance(error, BlockingIOError):
            return None
        raise
    return fd


def drawing_state(output: Path) -> dict[str, Any]:
    """Return a drawing's state: ``none``, ``drawing``, ``done`` or ``failed``.

    The folder's lock is tried first: held, a process is drawing (until it has exited,
    after its summary is written). Got, no process is, so the summary is final; a
    ``drawing`` summary then means the process was killed (out of memory) before its end.
    The probe holds the lock for an instant: a drawing started at that instant is refused
    and can be started again. The lock is ``flock``: on NFS it is a whole-file POSIX lock,
    reliable on one host.
    """
    import os

    try:
        fd = _try_lock(output) if output.is_dir() else None
    except OSError:  # the folder cannot be written: nothing was drawn by the service
        return {"state": "none"}
    if output.is_dir() and fd is None:
        return {"state": "drawing"}
    try:
        try:
            found: dict[str, Any] = json.loads((output / "summary.json").read_text())
        except (OSError, ValueError):
            return {"state": "none"}
    finally:
        if fd is not None:
            os.close(fd)
    if found.get("state") == "drawing":
        return {**found, "state": "failed",
                "error": "the drawing process ended before it finished"}  # fmt: skip
    return found


#: The machine-wide slots of background drawings, in the resource pools' folder.
SLOTS_DIR = "plot"


def _take_slot(max_drawings: int) -> int | None:
    """Return a flocked descriptor on a free drawing slot of the machine, or ``None``.

    The slots sit beside the resource pools (``execution/resources.py``), outside
    ``pools.toml``: their count is the service's ``[agents.plotter] max_drawings``.
    """
    from apex.execution.resources import default_root

    directory = default_root() / SLOTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for slot in range(max_drawings):
        fd = _try_lock(directory, f"{slot}.lock")
        if fd is not None:
            return fd
    return None


def start_drawing(
    run_path: Path,
    requests: Sequence[FigureRequest],
    *,
    output_dir: Path | None = None,
    dpi_scale: float = 1.0,
    max_drawings: int = 2,
) -> Path:
    """Draw ``requests`` in a detached process, and return the output folder at once.

    A classic set takes minutes, longer than a web request or an assistant's tool may
    wait. The figures appear as they are drawn; ``summary.json`` says ``drawing`` until
    the process ends (:func:`drawing_state` reads it). The process inherits the folder's
    lock and holds it until it exits, so a second drawing into the folder is refused
    meanwhile, and a crash releases it. It also holds one of the machine's
    ``max_drawings`` slots, so that drawings for many runs at once cannot exhaust the
    memory the simulations need.

    Raises
    ------
    ConfigError
        If the run's figures are already being drawn into that folder, the machine is
        drawing ``max_drawings`` runs' figures already, or ``dpi_scale`` is out of range.
    """
    import os
    import subprocess
    import sys
    import threading
    import uuid

    if not 0 < dpi_scale <= MAX_DPI_SCALE:
        raise ConfigError(f"dpi_scale must be in (0, {MAX_DPI_SCALE:g}], got {dpi_scale}")
    output = (output_dir or Path(run_path) / DEFAULT_SUBDIR).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fd = _try_lock(output)
    if fd is None:
        raise ConfigError("this run's figures are being drawn: try again when they are done")
    try:
        slot = _take_slot(max_drawings)
    except BaseException:
        os.close(fd)
        raise
    if slot is None:
        os.close(fd)
        raise ConfigError(f"the server is drawing {max_drawings} runs' figures already: "
                          "try again when one is done")  # fmt: skip
    plan_file = output / f".plot_plan_{uuid.uuid4().hex}.json"
    try:
        plan_file.write_text(json.dumps({
            "run": str(Path(run_path).resolve()),
            "requests": [{"figure": r.figure, "options": dict(r.options), "subdir": r.subdir}
                         for r in requests],
            "output": str(output),
            "dpi_scale": dpi_scale,
        }, default=str))  # fmt: skip
        # Written before the process starts: its own end ("done", "failed") comes after.
        (output / "summary.json").write_text(
            json.dumps({"state": "drawing", "requests": len(requests)})
        )
    except OSError as error:
        os.close(fd)
        os.close(slot)
        plan_file.unlink(missing_ok=True)
        raise ConfigError(f"the drawing could not start: {error.strerror}") from None
    try:
        try:
            with (output / "plot.log").open("ab") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "apex.services.figures", str(plan_file)],
                    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                    start_new_session=True, pass_fds=(fd, slot),
                )  # fmt: skip
        except OSError as error:
            plan_file.unlink(missing_ok=True)
            (output / "summary.json").write_text(
                json.dumps({"state": "failed", "error": f"could not start: {error.strerror}"})
            )
            raise ConfigError(f"the drawing could not start: {error.strerror}") from None
    finally:
        os.close(fd)  # the process keeps its inherited copies of the lock and the slot
        os.close(slot)
    # Reaped when it ends: a long-lived service would otherwise keep it as a zombie.
    threading.Thread(target=process.wait, name="plot-reaper", daemon=True).start()
    return output


def main(plan_file: str) -> int:
    """Draw a plan written by :func:`start_drawing` (the detached process); return its code."""
    import os
    import signal

    import apex.services.composition  # noqa: F401  (registers the figure plugins)

    path = Path(plan_file)
    output = path.parent
    run_path = Path("<run>")

    def failed(error: str) -> None:
        # Server paths masked: the run's, the output's and the plan's (it reaches the web UI).
        shown = _shown(error, run_path).replace(str(path), "<plan>").replace(str(output), "<plots>")
        (output / "summary.json").write_text(json.dumps({"state": "failed", "error": shown}))

    def terminated(signum: int, _frame: object) -> None:
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, terminated)
    try:
        os.nice(10)  # behind the simulations and the service on a shared machine
        plan = json.loads(path.read_text())
        run_path = Path(plan["run"])
        draw(run_path,
             [FigureRequest(r["figure"], r["options"], r["subdir"]) for r in plan["requests"]],
             output_dir=output, dpi_scale=float(plan["dpi_scale"]), locked=True)  # fmt: skip
    except (Exception, SystemExit) as error:  # noqa: BLE001 (recorded for the reader)
        failed(f"{type(error).__name__}: {error}")
        return 1
    finally:
        path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1]))
