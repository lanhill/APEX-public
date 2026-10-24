"""The audience a workspace package declares.

Installed metadata carries no ``[tool]`` tables, so a package declares its audience by an
entry point: ``[project.entry-points."apex.audience"] lab = "my_package"`` (the name is the
audience). Plugins and agents of a package that is not public are private.
"""

import functools
import re

#: The entry-point group by which a workspace package declares its audience:
#: ``[project.entry-points."apex.audience"] lab = "my_package"`` (the name is the audience).
AUDIENCE_GROUP = "apex.audience"


def audience(distribution: str) -> str:
    """Return the audience a distribution declares (``public`` when it declares none).

    Installed metadata carries no ``[tool]`` tables, so a package declares its audience by an
    entry point of :data:`AUDIENCE_GROUP` (``norce``, ``lab``, ``project:<name>``).
    """
    return _audiences().get(_normal(distribution), "public")


@functools.cache
def _audiences() -> dict[str, str]:
    import importlib.metadata

    found: dict[str, str] = {}
    for entry_point in importlib.metadata.entry_points(group=AUDIENCE_GROUP):
        if entry_point.dist is not None:
            found[_normal(entry_point.dist.name)] = entry_point.name
    return found


def _normal(distribution: str) -> str:
    # Distribution names compare case-insensitively, "-", "_" and "." alike (PEP 503).
    return re.sub(r"[-_.]+", "-", distribution).lower()
