"""Figure plugins and presets; importing this package registers them.

``services`` imports it to register the plugins, as for every plugin family.
"""

from apex.reporting.figures import classic, geophysics

__all__ = ["classic", "geophysics"]
