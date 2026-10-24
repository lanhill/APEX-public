"""Deprecated: moved to :mod:`apex.services.prediction`; kept for one release."""

import warnings
from typing import Any

from apex.services import prediction as _moved

warnings.warn(
    "apex.cli.prediction moved to apex.services.prediction; import it from there",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # noqa: ANN401 (forwards any attribute)
    return getattr(_moved, name)
