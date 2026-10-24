"""Every test runs isolated (:mod:`apex.testing`: its own resources and service root).

Here, at the top: pytest accepts ``pytest_plugins`` only in a top-level conftest, so every
folder of tests gets it, whichever folder pytest is given.
"""

pytest_plugins = ["apex.testing"]
