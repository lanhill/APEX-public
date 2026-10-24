"""The job service process: ``python -m apex.cli.service_daemon ROOT``.

Started by ``apex service start`` (without ``--http``; with it, ``apex.api.server`` also
serves the REST API). It runs :func:`apex.services.daemon.serve_jobs` until
SIGTERM or SIGINT (``apex service stop``).
"""

import logging
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

from apex.services.daemon import serve_jobs


def main() -> None:
    """Entry point of the service process."""
    root = Path(sys.argv[1])
    logging.basicConfig(
        filename=root / "service.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    stop = threading.Event()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    from apex.agents.watch import service_hook

    serve_jobs(root, stop, service_hook(root))  # the run monitor, if [agents.monitor] auto


if __name__ == "__main__":
    main()
