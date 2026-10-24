"""Parsing an ERT case as ERT parses it."""

import contextlib
from pathlib import Path
from typing import Any

from apex.core.errors import ConfigError


def parse_case(case_file: Path) -> Any:  # noqa: ANN401 (ERT's ErtConfig)
    """Return the parsed ``ErtConfig`` of ``case_file``, with ERT's site plugins (APEX's).

    ERT resolves some paths (e.g. ``BASE_SURFACE``) against the working directory, and runs
    from the case's folder; so does this parse.
    """
    from ert.config import ErtConfig
    from ert.plugins import get_site_plugins

    require_supported_ert()
    case_file = Path(case_file).resolve()
    with contextlib.chdir(case_file.parent):
        return ErtConfig.with_plugins(get_site_plugins()).from_file(str(case_file))


#: The ERT major version the integration is written for (the ``ert`` extra).
ERT_MAJOR = 26


def require_supported_ert() -> None:
    """Refuse an installed ERT of another major version, whose private modules APEX uses.

    Raises
    ------
    ConfigError
        If ERT's major version is not :data:`ERT_MAJOR`.
    """
    from importlib import metadata

    version = metadata.version("ert")
    if version.split(".")[0] != str(ERT_MAJOR):
        raise ConfigError(
            f"APEX's ERT integration needs ERT {ERT_MAJOR}.x, but ERT {version} is installed: "
            f'install apex[ert] (pip install "ert>={ERT_MAJOR},<{ERT_MAJOR + 1}")'
        )
