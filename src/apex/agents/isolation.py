"""Run a tool's work in a worker process that can be killed.

A thread cannot be stopped: a tool the assistant abandons (the Stop button, or its time
limit) kept its CPU and memory until it returned. A tool run through :func:`run_in_worker`
instead runs in a process of its own, in its own process group, which is killed as soon as
the loop abandons the tool (:func:`apex.agents.loop.tool_cancelled`).

Workers start by ``forkserver`` where available, as the process executor's do: the service
is multi-threaded and must never be forked.
"""

import logging
import multiprocessing
import os
import signal
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from apex.agents.loop import tool_cancelled
from apex.core.errors import ApexError

logger = logging.getLogger(__name__)

_START_METHOD = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
_POLL_S = 0.1


class ToolCancelled(Exception):
    """Raised in the abandoned tool's thread once its worker is killed (its result is dropped)."""


class WorkerFailed(ApexError):
    """The worker process ended without a result (killed from outside, or out of memory)."""


def run_in_worker(function: Callable[..., Any], arguments: tuple[Any, ...]) -> Any:  # noqa: ANN401
    """Return ``function(*arguments)``, run in a worker process; re-raise what it raised.

    ``function`` must be a module-level function and ``arguments`` picklable. If the calling
    tool is abandoned meanwhile, the worker's process group is killed and
    :class:`ToolCancelled` raised.

    Raises
    ------
    ToolCancelled
        If the loop abandoned the tool.
    WorkerFailed
        If the worker ended without a result.
    """
    context = multiprocessing.get_context(_START_METHOD)
    receive, send = context.Pipe(duplex=False)
    process = context.Process(  # type: ignore[attr-defined]
        # Daemonic: the service's exit never waits for a tool (the tools start no processes).
        target=_main,
        args=(function, arguments, send),
        name="apex-tool",
        daemon=True,
    )
    process.start()
    send.close()  # the worker's end: the parent sees EOF once the worker has gone
    cancel = tool_cancelled()
    try:
        while True:
            if receive.poll(_POLL_S):
                try:
                    kind, value = receive.recv()
                except EOFError:
                    process.join(1.0)
                    raise WorkerFailed(
                        f"the tool's worker process ended (exit code {process.exitcode}) "
                        "without a result"
                    ) from None
                except Exception as error:  # noqa: BLE001 (a result that does not unpickle)
                    raise WorkerFailed(f"the tool's result could not be read: {error}") from None
                if kind == "ok":
                    return value
                raise value
            if cancel is not None and cancel.is_set():
                _kill(process)
                logger.info("tool worker killed", extra={"event": "tool_worker_killed",
                                                         "pid": process.pid})  # fmt: skip
                raise ToolCancelled("the tool was abandoned and its worker killed")
    finally:
        receive.close()
        process.join(5.0)
        if process.is_alive():
            _kill(process)


def _kill(process: multiprocessing.process.BaseProcess) -> None:
    pid = process.pid
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)  # the worker leads its own group (see _main)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.join(5.0)


def _main(function: Callable[..., Any], arguments: tuple[Any, ...], send: Connection) -> None:
    # Its own process group: a kill takes whatever the tool started too.
    os.setsid()
    try:
        result = function(*arguments)
    except Exception as error:  # noqa: BLE001 (handed back to the caller's process)
        try:
            send.send(("error", error))
        except Exception:  # noqa: BLE001 (an exception that does not pickle)
            send.send(("error", ValueError(f"{type(error).__name__}: {error}")))
    else:
        try:
            send.send(("ok", result))
        except Exception as error:  # noqa: BLE001 (a result that does not pickle)
            send.send(("error", ValueError(f"the tool's result cannot be returned: {error}")))
    finally:
        send.close()
