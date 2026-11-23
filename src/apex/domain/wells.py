"""Model error's wells: their blocks and status by report step, and the features per block.

The layout of the networks: :func:`feature_names` per perforated block, and each
well's :class:`WellSchedule`, read from a deck's schedule (``apex.simulation.model_error``)
or stated by a simulator (``apex.core.case_state.WellLayout``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["PHASES", "STATIC", "WellSchedule", "feature_names"]

#: Static features (INIT), per perforated block.
STATIC = ("DX", "DY", "DZ", "PERMX", "PERMY", "PERMZ", "PORO")
#: The phases a deck may have, with their density and saturation keywords.
PHASES = {"oil": ("OIL_DEN", "SOIL"), "water": ("WAT_DEN", "SWAT"), "gas": ("GAS_DEN", "SGAS")}


def feature_names(phases: Sequence[str]) -> tuple[str, ...]:
    """Return the features per perforated block for a deck's phases, in PET-fork's order.

    The static ones, then each phase's density and saturation (oil, water, gas), with the
    oil saturation, which the simulators do not write, after the others, then PRESS_DIFF
    (the well's WBHP minus the block pressure). Oil-water: ``OIL_DEN WAT_DEN SWAT SOIL
    PRESS_DIFF``; with gas: ``OIL_DEN WAT_DEN SWAT GAS_DEN SGAS SOIL PRESS_DIFF``. The oil's
    ``OIL_DEN`` and ``SOIL`` are always in the layout: a deck without oil (a gas-water or
    CO2 storage deck) has them as 0 (:func:`well_features`; the user, 2026-11-07).

    Raises
    ------
    ValueError
        If ``phases`` is empty, repeats a phase or names an unknown one.
    """
    names = [p.lower() for p in phases]
    unknown = sorted(set(names) - set(PHASES))
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(
            f"phases must be one or more of {sorted(PHASES)}, each once, got {list(phases)}"
        )
    dynamic = ["OIL_DEN"]
    for phase in ("water", "gas"):
        if phase in names:
            dynamic += list(PHASES[phase])
    return (*STATIC, *dynamic, "SOIL", "PRESS_DIFF")


@dataclass(frozen=True)
class WellSchedule:
    """One well's perforations and status at each report time.

    Attributes
    ----------
    name, kind
        Well name and ``"producer"`` or ``"injector"``.
    outputs
        Its data types, e.g. ``("WOPR", "WBHP", "WWCT")``.
    blocks
        Every block it perforates at any report time, first seen first: a grid's 1-based
        ``(i, j, k)`` (a deck's schedule), or a mesh's 0-based cell in its case-state
        reader's order (a simulator's layout).
    perforated
        ``(n_steps, n_blocks)``: whether each block is perforated at each report time.
    open
        ``(n_steps,)``: whether the well is open at each report time.
    """

    name: str
    kind: str
    outputs: tuple[str, ...]
    blocks: tuple[tuple[int, int, int] | int, ...]
    perforated: NDArray[np.bool_]
    open: NDArray[np.bool_]

    def n_features(self, phases: Sequence[str] = ("oil", "water")) -> int:
        """Return the input features per report time: every feature at every block."""
        return len(feature_names(phases)) * len(self.blocks)
