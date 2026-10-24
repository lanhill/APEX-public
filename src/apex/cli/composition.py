"""Deprecated: moved to :mod:`apex.services.composition`; kept for one release."""

import warnings
from typing import Any

from apex.services import composition as _moved

warnings.warn(
    "apex.cli.composition moved to apex.services.composition; import it from there",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # noqa: ANN401 (forwards any attribute)
    return getattr(_moved, name)
