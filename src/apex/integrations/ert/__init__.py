"""APEX and Equinor's ERT.

- :mod:`.export`: an APEX configuration as a self-contained ERT case (``apex ert export``).
- :mod:`.render`: the forward-model step ``APEX_RENDER_DECK``, which renders APEX's Mako
  deck in an ERT run path from the fields ERT writes there.
- :mod:`.plugin`: the ERT plugin (entry point ``ert``), registering the step with ERT.

Only :mod:`.plugin` imports ERT (the ``ert`` extra, ERT 26.x).
"""
