"""Surrogate models' core computations: the validated domain, the gate, the designs.

Pure functions over arrays, the ensemble (run) axis last; no plugin lives here. The
surrogate plugins are other packages' (``apex.surrogate_models`` entry points), and each is
held to the same domain (:mod:`.domain`) and gate (:mod:`.gate`), which core computes, not
the plugin. :mod:`.designs` draws the full runs a surrogate is trained and gated on.
"""
