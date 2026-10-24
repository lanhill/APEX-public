"""What produced a result: software versions, host and input hashes.

Written as ``provenance.json`` next to results, so that a result can be traced to the
APEX commit, the libraries and the inputs that made it. Bit-identical reproduction also
depends on the CPU (NumPy and OpenBLAS round differently on AVX-512), so the host is
recorded too.
"""

import datetime as dt
import hashlib
import importlib.metadata
import platform
import socket
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import apex

#: Libraries whose versions affect numerical results.
LIBRARIES = ("numpy", "scipy", "pywavelets", "resdata", "mako")


def file_hash(path: Path) -> str:
    """SHA-256 of a file's content."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str | None:
    """Run git in the APEX checkout; its output, or ``None`` for an installed package."""
    root = Path(apex.__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None  # an installed package, not a checkout
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _avx512() -> bool | None:
    # NumPy and OpenBLAS round differently with AVX-512, so record it.
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None  # not Linux
    return "avx512f" in text.split()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(
    inputs: Mapping[str, Path], *, simulator: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the provenance record.

    Parameters
    ----------
    inputs
        Named input files to hash, e.g. ``{"deck_template": path}``; missing files are
        recorded as such.
    simulator
        What is known of the simulator (name, executable, version).
    """
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "apex": {
            "version": apex.__version__,
            "commit": commit,
            "dirty": None if status is None else bool(status),
        },
        "python": platform.python_version(),
        "libraries": {name: _version(name) for name in LIBRARIES},
        "host": {
            "name": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "avx512": _avx512(),
        },
        "simulator": dict(simulator or {}),
        "inputs": {
            name: file_hash(path) if Path(path).is_file() else "missing"
            for name, path in inputs.items()
        },
    }
