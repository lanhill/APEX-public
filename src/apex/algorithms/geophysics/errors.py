"""Errors of the geophysics kernels that depend on a member's state.

The simulator (phase 2) turns exactly these into a member's non-transient
``SimulationError``, and nothing broader: an ordinary ``ValueError`` from a kernel is a
configuration or programming error and must stop the run, not pass as one member's
physics.
"""


class GeophysicsModelError(ValueError):
    """A member's model is outside a geophysical kernel's validity (its own fault)."""
