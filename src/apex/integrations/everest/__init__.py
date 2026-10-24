"""APEX's optimizers for Everest: a ``ropt`` backend.

Everest picks it with ``optimization: algorithm: apex/enopt`` (or ``apex/genopt``,
``apex/smcopt``, ``apex/line_search``, ``apex/trust_region``); ``optimization: options``
are the APEX optimizer's options. Registered through the entry point
``ropt.plugins.backend`` (``apex``); see :mod:`apex.integrations.everest.backend`.
"""
