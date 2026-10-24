"""QA/QC of a run (``apex qaqc``): PET's diagnostics on a finished linear case."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from typer.testing import CliRunner

from apex.algorithms import qaqc as kernels
from apex.cli.main import app
from apex.services import composition
from apex.services.qaqc import qaqc_report


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def test_the_report_covers_every_diagnostic(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=40)
    composition.start(case.config)
    run = case.directory / "runs" / "case"
    results = composition.load_results(run)

    prior = qaqc_report(run, iteration=0)
    posterior = qaqc_report(run)

    assert posterior.iteration == results.posterior_iteration
    # The linear case's 8 data are one type, y, at 8 times.
    assert [(c.datatype, len(c.times)) for c in prior.coverage] == [("y", 8)]
    expected = kernels.coverage(results.iterate(0).predicted, prior.coverage[0].observations)
    assert prior.coverage[0].outside == expected.outside
    assert len(posterior.data_scores) == 8
    assert posterior.pair_scores  # 28 pairs, the top 10 kept
    assert len(posterior.pair_scores) == 10
    assert list(posterior.movement) == ["x"]
    assert posterior.movement["x"].std < posterior.movement["x"].prior_std  # the update shrinks
    assert posterior.gains_max  # the 3-value field x is ranked
    # The data are fitted: the posterior's largest score is below the prior's.
    assert posterior.data_scores[0][1] < prior.data_scores[0][1]


def test_the_command_writes_the_report_and_figures(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run = case.directory / "runs" / "case"

    result = CliRunner().invoke(app, ["qaqc", str(run)])

    assert result.exit_code == 0, result.output
    text = (run / "qaqc" / "qaqc.md").read_text()
    assert "## Coverage" in text
    assert "## Movement from the prior" in text
    assert (run / "qaqc" / "coverage_y.png").is_file()
    assert np.isfinite(qaqc_report(run).movement["x"].std)
