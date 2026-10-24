"""Examples and toy problems, grouped by workflow (the catalogue's manifests).

Each example is a folder with an ``example.toml``::

    [example]
    title = "ZDT1: a convex two-objective front"
    summary = "NSGA-II on Zitzler, Deb and Thiele's ZDT1 (10 variables)."
    source = "pymoo 0.6.2 (Blank & Deb 2020), problems/multi/zdt.py; Zitzler, Deb & Thiele 2000"
    tags = ["toy", "multi-objective", "analytic"]
    cost = "seconds"
    requires = []

    [[example.entries]]
    workflow = "optimization"
    config = "zdt1_nsga2.toml"

The grouping is by metadata: an example serves one or more workflows (the ids the Plugins
page uses: ``history_matching``, ``prediction``, ``optimization``, ...), one entry each, so a
new toy problem, or a new workflow, is a new folder with a manifest and no code.
"""

from typing import Literal

from pydantic import Field

from apex.domain.config import _Strict

#: How long an example takes on a laptop: what its card says.
type Cost = Literal["seconds", "minutes", "hours"]


class ExampleEntry(_Strict):
    """One configuration of an example, for one workflow."""

    workflow: str = Field(
        min_length=1,
        pattern=r"^[a-z_]+$",
        description="The workflow it serves (history_matching, prediction, optimization, ...).",
    )
    config: str = Field(
        min_length=1,
        description="The configuration, relative to the example's folder (a .toml file).",
    )
    title: str | None = Field(
        default=None, description="What this entry shows, when the example has several."
    )


class ExampleSection(_Strict):
    """``[example]``: what the catalogue shows of an example."""

    title: str = Field(min_length=3)
    summary: str = Field(min_length=3)
    source: str | None = Field(
        default=None,
        description="Where the problem or case comes from (a project, a paper), shown with it.",
    )
    tags: list[str] = Field(default_factory=list)
    cost: Cost = "seconds"
    requires: list[str] = Field(
        default_factory=list,
        description="Extras or programs it needs (e.g. reservoir, opm_flow); none: plain APEX.",
    )
    entries: list[ExampleEntry] = Field(min_length=1)


class ExampleManifest(_Strict):
    """An ``example.toml``."""

    example: ExampleSection
