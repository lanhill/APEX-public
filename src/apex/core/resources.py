"""Machine-wide resource pools shared by every simulator run.

A pool is a named count of tokens, e.g. ``eclipse`` (the site's licences) or ``cpu``
(cores). A simulator declares what one run needs (an optional ``resources`` capability,
``{pool: tokens}``), and executors hold those tokens for the length of the run. The pools
span every process on the machine, so concurrent jobs, studies and command-line runs
together never use more than a pool holds.
"""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol


class ResourcePools(Protocol):
    """Named token pools, held across processes."""

    def sizes(self) -> Mapping[str, int]:
        """Return the size of every pool."""
        ...

    def hold(self, needs: Mapping[str, int]) -> AbstractContextManager[None]:
        """Wait until ``needs`` (``{pool: tokens}``) are free, and hold them while in use.

        Raises
        ------
        ValueError
            If a pool is unknown or smaller than what is needed (it could never be met).
        """
        ...
