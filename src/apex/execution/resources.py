"""File-lock resource pools: machine-wide token counts.

Each pool is a directory of slot files, ``<root>/<pool>/<i>.lock`` for ``i < size``; a
token is an exclusive ``flock`` on one slot. The operating system releases the locks of
a process that dies, so a crashed run never keeps its tokens. The sizes are kept in
``<root>/pools.toml``, written with the defaults on first use and then authoritative for
every process, so that all of them agree on each pool's size:

    [pools]
    cpu = 18        # the core count minus 2
    eclipse = 3     # the site's Eclipse licences

The root is ``$APEX_RESOURCES``, else ``~/.apex/resources``. It must be on a **local**
disk of the machine: on a network file system ``flock`` may be emulated or shared across
machines, which would neither hold within this machine nor keep the machines apart.
Waiting runs poll without a queue, so a run needing several pools may wait longer than
one needing a single pool under contention (no fairness, but no deadlock).
"""

import fcntl
import os
import time
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO

#: The Eclipse licences assumed when none are configured: the ``eclipse`` pool's default
#: size, and the cap validation holds an Eclipse case's ``licenses`` to.
ECLIPSE_LICENCES = 3


def default_sizes() -> dict[str, int]:
    """Default pools: ``cpu`` = cores minus 2 (at least 1), ``eclipse`` = 3 licences."""
    return {"cpu": max(1, (os.cpu_count() or 1) - 2), "eclipse": ECLIPSE_LICENCES}


def default_root() -> Path:
    """Where the machine's pools live: ``$APEX_RESOURCES``, else ``~/.apex/resources``."""
    return Path(os.environ.get("APEX_RESOURCES", Path.home() / ".apex" / "resources"))


class FilePools:
    """Token pools of one machine, held with file locks.

    Parameters
    ----------
    root
        The pools' directory (default :func:`default_root`).
    poll_s
        Seconds between attempts while waiting for tokens.
    """

    def __init__(self, root: Path | None = None, poll_s: float = 0.2) -> None:
        self.root = Path(root) if root is not None else default_root()
        self.poll_s = poll_s

    def sizes(self) -> dict[str, int]:
        """Return the pools' sizes, writing ``pools.toml`` with the defaults if missing."""
        path = self.root / "pools.toml"
        if not path.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            text = "[pools]\n" + "".join(f"{k} = {v}\n" for k, v in default_sizes().items())
            temporary = path.with_name(f"pools.{os.getpid()}.tmp")
            temporary.write_text(text)
            # Only the first writer wins, so every process sees the same sizes.
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            finally:
                temporary.unlink()
        pools = tomllib.loads(path.read_text()).get("pools", {})
        return {str(name): int(size) for name, size in pools.items()}

    @contextmanager
    def hold(self, needs: Mapping[str, int]) -> Iterator[None]:
        """Wait for ``needs`` and hold them; all tokens are taken at once or none."""
        needs = {pool: n for pool, n in needs.items() if n > 0}
        if not needs:
            yield
            return
        sizes = self.sizes()
        _check(needs, sizes)
        while True:
            with ExitStack() as held:
                # All or nothing: holding some tokens while waiting for others could
                # deadlock two runs that each hold part of what the other needs.
                if all(self._take(held, pool, n, sizes[pool]) for pool, n in sorted(needs.items())):
                    yield
                    return
            time.sleep(self.poll_s)

    def try_hold(self, needs: Mapping[str, int]) -> ExitStack | None:
        """Take ``needs`` now if all are free; return the held tokens (close to release).

        ``None`` if any is taken: the caller may try another pool instead of waiting (the
        ``ssh`` executor's hosts). Unknown or too small pools raise as in
        :meth:`hold`.
        """
        needs = {pool: n for pool, n in needs.items() if n > 0}
        sizes = self.sizes()
        _check(needs, sizes)
        held = ExitStack()
        if all(self._take(held, pool, n, sizes[pool]) for pool, n in sorted(needs.items())):
            return held
        held.close()
        return None

    def _take(self, held: ExitStack, pool: str, n: int, size: int) -> bool:
        directory = self.root / pool
        directory.mkdir(parents=True, exist_ok=True)
        taken = 0
        for slot in range(size):
            handle = held.enter_context((directory / f"{slot}.lock").open("a"))
            if _try_lock(handle):
                taken += 1
                if taken == n:
                    return True
            else:
                handle.close()
        return False


def _check(needs: Mapping[str, int], sizes: Mapping[str, int]) -> None:
    for pool, n in needs.items():
        if pool not in sizes:
            raise ValueError(f"unknown resource pool {pool!r}; pools: {sorted(sizes)}")
        if n > sizes[pool]:
            raise ValueError(f"a run needs {n} {pool} tokens, but the pool holds {sizes[pool]}")


def _try_lock(handle: IO[str]) -> bool:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True
