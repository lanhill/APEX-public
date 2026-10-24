"""Read and write one keyword of a GRDECL text file, values in Eclipse cell order."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_grdecl(path: Path, keyword: str, values: NDArray[np.float64]) -> None:
    """Write ``values`` (Eclipse order, ``i`` fastest) as ``keyword`` in GRDECL text.

    Each value is its shortest exact representation, so reading it back gives the same
    double (what ``APEX_RENDER_DECK`` renders must equal APEX's own rendering).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(values, dtype=np.float64).ravel()
    lines = [f"{keyword}"]
    for start in range(0, flat.size, 5):
        lines.append(" ".join(repr(float(v)) for v in flat[start : start + 5]))
    lines.append("/")
    path.write_text("\n".join(lines) + "\n")


def read_grdecl(path: Path, keyword: str) -> NDArray[np.float64]:
    """Return ``keyword``'s values from a GRDECL text file, Eclipse order.

    Comments (``--``) are skipped and repeat counts (``3*0.25``) expanded.

    Raises
    ------
    ValueError
        If the file has no such keyword, or it is not terminated by ``/``.
    """
    tokens: list[str] = []
    for line in path.read_text().splitlines():
        tokens.extend(line.split("--", 1)[0].split())
    try:
        start = tokens.index(keyword) + 1
    except ValueError as error:
        raise ValueError(f"{path} has no keyword {keyword}") from error
    values: list[float] = []
    for token in tokens[start:]:
        if token == "/" or token.endswith("/"):
            body = token.rstrip("/")
            if body:
                values.extend(_expand(body))
            return np.array(values, dtype=np.float64)
        values.extend(_expand(token))
    raise ValueError(f"{path}: {keyword} is not terminated by '/'")


def _expand(token: str) -> list[float]:
    if "*" in token:
        count, value = token.split("*", 1)
        return [float(value)] * int(count)
    return [float(token)]
