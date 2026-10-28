"""The service process with the REST API: ``python -m apex.api.server ROOT --http HOST:PORT``.

Started by ``apex service start --http HOST:PORT``. The scheduling loop of ADR
0014 (:func:`apex.services.daemon.serve_jobs`) runs in a thread, and uvicorn serves
:func:`apex.api.app.create_app` in the main thread. SIGTERM (``apex service stop``) stops
uvicorn, then the loop. A loop that ends unasked (a failure) stops uvicorn too, and the
process exits with status 1, so that its process manager (systemd's ``Restart=on-failure``)
restarts it: on 2026-10-25 the loop ended while uvicorn served on, and nothing was
scheduled for two days. The default host is ``127.0.0.1``: remote use goes through an SSH
tunnel, or an explicit host behind a TLS-terminating proxy.
"""

import argparse
import logging
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from apex.api.app import create_app
from apex.services.daemon import serve_jobs

logger = logging.getLogger("apex.service")


def parse_address(address: str) -> tuple[str, int]:
    """Split ``HOST:PORT`` (or ``PORT``, on ``127.0.0.1``)."""
    host, _, port = address.rpartition(":")
    try:
        return host or "127.0.0.1", int(port)
    except ValueError as error:
        raise ValueError(f"--http takes HOST:PORT or PORT, not {address!r}") from error


#: Seconds uvicorn gives open connections to finish once stopped. The web UI keeps event
#: streams open (jobs, chats); without a limit uvicorn waits for every browser to let go.
GRACEFUL_SHUTDOWN_S = 5


def watched(
    loop: Callable[[], None], stop: threading.Event, ended: Callable[[], None]
) -> Callable[[], None]:
    """Wrap the scheduling ``loop`` so that ``ended`` is called if it ends while ``stop`` is unset.

    The loop returns once ``stop`` is set; any other end (a failure, logged by the loop) is
    reported. Its exception goes on to the thread's hook.
    """

    def run() -> None:
        try:
            loop()
        finally:
            if not stop.is_set():
                logger.error(
                    "the scheduling loop ended unasked: the service stops, to be restarted",
                    extra={"event": "service_loop_ended"},
                )  # fmt: skip
                ended()

    return run


def main(argv: list[str] | None = None) -> None:
    """Entry point of the service process with the REST API."""
    import uvicorn
    from uvicorn.config import STARTUP_FAILURE

    parser = argparse.ArgumentParser(prog="apex.api.server")
    parser.add_argument("root", type=Path)
    parser.add_argument("--http", default="127.0.0.1:8765", help="HOST:PORT to serve on")
    args = parser.parse_args(argv)
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    host, port = parse_address(args.http)
    logging.basicConfig(
        filename=root / "service.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    from apex.agents.watch import service_hook

    hook = service_hook(root)  # the run monitor, if [agents.monitor] auto
    config = uvicorn.Config(
        create_app(root, host=host), host=host, port=port, log_level="warning",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )  # fmt: skip
    server = uvicorn.Server(config)
    stop, failed = threading.Event(), threading.Event()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        stop.set()
        server.should_exit = True  # a signal before uvicorn's own handlers stops it too

    # uvicorn handles SIGTERM itself, then raises it again with the handler it found: this
    # one, so that the process ends through the `finally` below (the loop stops and the
    # pid file goes), not through the default handler.
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def ended() -> None:
        failed.set()
        server.should_exit = True

    (root / "service.url").write_text(f"http://{host}:{port}\n")
    loop = threading.Thread(
        target=watched(lambda: serve_jobs(root, stop, hook), stop, ended), name="apex-scheduler"
    )
    loop.start()
    try:
        server.run()
    finally:
        stop.set()
        loop.join()
        (root / "service.url").unlink(missing_ok=True)
    if failed.is_set():
        raise SystemExit(1)
    if not server.started:  # e.g. the port is taken (as uvicorn.run exits)
        raise SystemExit(STARTUP_FAILURE)


if __name__ == "__main__":
    main()
