"""Plurigaussian simulation: rules, conditioning, the prior and the truncation.

Independent references: GSTools' own ``PGS`` (the same lithotype indexing for the same
axes), the rule's analytic facies probabilities, and the analytic truncated normal.
"""

from pathlib import Path
from typing import Any

import gstools as gs
import numpy as np
import pytest
from scipy import special, stats

from apex.algorithms.facies import LithotypeRule, PGSPrior, PGSTruncation, condition_latent
from apex.core.plugins import parameterizations, priors
from apex.domain.state import StateVariable

CODES_2D = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [2, 2, 2, 1]])


def test_the_truncation_is_gstools_pgs_for_the_same_axes_2d() -> None:
    x, y = np.arange(40), np.arange(30)
    fields = [
        gs.SRF(gs.Gaussian(dim=2, len_scale=[8, 4], angles=0.4), seed=11).structured([x, y]),
        gs.SRF(gs.Exponential(dim=2, len_scale=6), seed=12).structured([x, y]),
    ]
    pgs = gs.PGS(2, fields)
    expected = pgs(CODES_2D)
    rule = LithotypeRule(CODES_2D, tuple(pgs.calc_lithotype_axes(CODES_2D.shape)))

    got = rule.facies(np.stack([f.ravel() for f in fields]))

    np.testing.assert_array_equal(got, np.asarray(expected).ravel())


def test_the_truncation_is_gstools_pgs_for_the_same_axes_3d() -> None:
    axes = [np.arange(12), np.arange(10), np.arange(4)]
    fields = [
        gs.SRF(gs.Gaussian(dim=3, len_scale=[5, 4, 2]), seed=s).structured(axes) for s in (1, 2, 3)
    ]
    codes = np.arange(27).reshape(3, 3, 3) % 4
    pgs = gs.PGS(3, fields)
    rule = LithotypeRule(codes, tuple(pgs.calc_lithotype_axes(codes.shape)))

    got = rule.facies(np.stack([f.ravel() for f in fields]))

    np.testing.assert_array_equal(got, np.asarray(pgs(codes)).ravel())


def test_facies_proportions_match_the_rules_normal_masses() -> None:
    rule = LithotypeRule.on_box(CODES_2D, extent=2.5)
    rng = np.random.default_rng(0)
    n = 400_000
    counts = np.bincount(rule.facies(rng.standard_normal((2, n))), minlength=3) / n

    expected = rule.proportions()

    assert expected.sum() == pytest.approx(1.0)
    # 5 standard errors of a binomial proportion.
    np.testing.assert_allclose(counts, expected, atol=5 * np.sqrt(0.25 / n))


def test_a_truncated_gaussian_has_its_target_proportions() -> None:
    rule = LithotypeRule.from_proportions([0.2, 0.5, 0.3])

    np.testing.assert_allclose(rule.proportions(), [0.2, 0.5, 0.3], atol=1e-14)
    edges = special.ndtri([0.2, 0.7])
    np.testing.assert_array_equal(
        rule.facies(np.array([[edges[0] - 1e-9, edges[0], 5.0]])), [0, 1, 2]
    )


def test_the_gibbs_value_at_one_cell_is_the_analytic_truncated_normal() -> None:
    """One field, one cell: the conditional is N(0, 1) truncated to facies 1's interval."""
    rule = LithotypeRule.from_proportions([0.3, 0.4, 0.3])
    lo, hi = special.ndtri([0.3, 0.7])
    draws = [
        condition_latent(rule, [np.eye(1)], np.array([1]), np.random.default_rng(s), sweeps=3)[0, 0]
        for s in range(2000)
    ]

    assert stats.kstest(draws, stats.truncnorm(lo, hi).cdf).pvalue > 0.01


def test_gibbs_values_honour_the_facies_of_correlated_cells() -> None:
    rule = LithotypeRule.on_box(CODES_2D, extent=2.0)
    distances = np.abs(np.subtract.outer(np.arange(5.0), np.arange(5.0)))
    covariance = np.exp(-((distances / 3.0) ** 2)) + 1e-9 * np.eye(5)
    observed = np.array([0, 0, 2, 1, 2])

    values = condition_latent(
        rule, [covariance, covariance], observed, np.random.default_rng(3), 50
    )

    np.testing.assert_array_equal(rule.facies(values), observed)


def _variables(grid: tuple[int, int, int], names: tuple[str, ...]) -> list[StateVariable]:
    size = grid[0] * grid[1] * grid[2]
    return [StateVariable(name, size, grid=grid) for name in names]


FIELDS: list[dict[str, Any]] = [
    {"model": "Gaussian", "len_scale": [6, 3]},
    {"model": "Exponential", "len_scale": 4},
]


def test_the_prior_is_reproducible_and_honours_its_conditioning(tmp_path: Path) -> None:
    np.save(tmp_path / "rule.npy", CODES_2D)
    cells, observed = [[2, 2, 1], [7, 3, 1], [4, 5, 1]], [2, 1, 0]
    prior = PGSPrior(
        FIELDS, rule_file=tmp_path / "rule.npy", conditioning={"cells": cells, "facies": observed},
        gibbs_sweeps=30,
    )  # fmt: skip
    grid = (10, 6, 1)
    variables = _variables(grid, ("y1", "y2"))

    first = prior.generate_joint(variables, 8, np.random.default_rng(5))
    again = prior.generate_joint(variables, 8, np.random.default_rng(5))
    other = prior.generate_joint(variables, 8, np.random.default_rng(6))

    np.testing.assert_array_equal(first["y1"], again["y1"])
    assert not np.array_equal(first["y1"], other["y1"])
    assert first["y1"].shape == (60, 8)
    assert first["y1"].flags.c_contiguous
    rule = LithotypeRule.on_box(CODES_2D, 3.0)
    index = [(i - 1) + grid[0] * (j - 1) for i, j, _ in cells]  # Eclipse order
    for member in range(8):
        latent = np.stack([first["y1"][index, member], first["y2"][index, member]])
        np.testing.assert_array_equal(rule.facies(latent), observed)


def test_unconditional_latent_fields_have_unit_variance() -> None:
    rule_prior = PGSPrior([FIELDS[0]], proportions=[0.5, 0.5])
    fields = rule_prior.generate_joint(
        _variables((30, 30, 1), ("y",)), 60, np.random.default_rng(1)
    )

    # Ensemble variance per cell, averaged: GSTools' randomization method is approximate.
    assert fields["y"].var(axis=-1, ddof=1).mean() == pytest.approx(1.0, abs=0.1)
    assert abs(fields["y"].mean()) < 0.1


def test_the_prior_validates_its_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"rule_file .* or proportions"):
        PGSPrior(FIELDS)
    np.save(tmp_path / "rule.npy", CODES_2D)
    with pytest.raises(ValueError, match="2 axes but 1 fields"):
        PGSPrior(FIELDS[:1], rule_file=tmp_path / "rule.npy")
    with pytest.raises(ValueError, match="model"):
        PGSPrior([{"model": "Wavy", "len_scale": 3}], proportions=[0.5, 0.5])
    prior = PGSPrior(
        [FIELDS[0]], proportions=[0.5, 0.5], conditioning={"cells": [[9, 1, 1]], "facies": [0]}
    )
    with pytest.raises(ValueError, match="outside the grid"):
        prior.generate_joint(_variables((4, 4, 1), ("y",)), 2, np.random.default_rng())


def test_the_truncation_maps_facies_to_constant_and_per_facies_properties() -> None:
    truncation = PGSTruncation(
        fields=["y"], facies=["shale", "sand"], proportions=[0.5, 0.5],
        values={"shale": {"poro": 0.05}, "sand": {"poro": 0.25}},
        property_fields={"permx": {"shale": "k_shale", "sand": "k_sand"}},
    )  # fmt: skip

    out = truncation.properties({
        "y": np.array([-1.0, 1.0, 0.5]), "k_shale": np.array([1.0, 2.0, 3.0]),
        "k_sand": np.array([10.0, 20.0, 30.0]),
    })  # fmt: skip

    assert truncation.variables == ("y", "k_sand", "k_shale")
    assert truncation.outputs == ("facies", "poro", "permx")
    np.testing.assert_array_equal(out["facies"], [0, 1, 1])
    np.testing.assert_array_equal(out["poro"], [0.05, 0.25, 0.25])
    np.testing.assert_array_equal(out["permx"], [1.0, 20.0, 30.0])


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"values": {"mud": {"poro": 0.1}}}, "unknown facies"),
        ({"values": {"shale": {"poro": 0.1}}}, "no value for facies sand"),
        ({"property_fields": {"permx": {"sand": "k"}}}, "every facies"),
        ({"facies": ["only"]}, "codes"),
    ],
)
def test_the_truncation_validates_its_options(options: dict[str, object], match: str) -> None:
    base: dict[str, object] = {
        "fields": ["y"],
        "facies": ["shale", "sand"],
        "proportions": [0.5, 0.5],
    }
    with pytest.raises(ValueError, match=match):
        PGSTruncation(**{**base, **options})  # type: ignore[arg-type]


def test_the_plugins_are_registered() -> None:
    assert priors.factory("gstools_pgs") is PGSPrior
    assert parameterizations.factory("pgs_truncation") is PGSTruncation


def _rejection(
    rule: LithotypeRule, covariances: list[np.ndarray], observed: np.ndarray, n: int, seed: int
) -> np.ndarray:
    """Exact draws of the conditioned latent values by rejection, ``(m, n_c, >= n)``."""
    rng = np.random.default_rng(seed)
    factors = [np.linalg.cholesky(c + 1e-6 * np.eye(len(c))) for c in covariances]
    kept: list[np.ndarray] = []
    while sum(k.shape[-1] for k in kept) < n:
        draws = np.stack([f @ rng.standard_normal((len(observed), 200_000)) for f in factors])
        facies = rule.facies(draws.reshape(len(factors), -1)).reshape(len(observed), -1)
        kept.append(draws[:, :, np.all(facies == observed[:, None], axis=0)])
    return np.concatenate(kept, axis=-1)


def _gibbs(
    rule: LithotypeRule, covariances: list[np.ndarray], observed: np.ndarray, chains: int
) -> np.ndarray:
    return np.stack(
        [
            condition_latent(rule, covariances, observed, np.random.default_rng(s), 15)
            for s in range(chains)
        ],
        axis=-1,
    )


def test_conditioning_a_smooth_vertical_well_matches_rejection_sampling() -> None:
    """The numerics review's case: 10 cells one apart under a Gaussian model (len_scale 5).

    Single-site Gibbs gave means of -47 against the exact -0.93; the HMC sampler must match
    rejection sampling within Monte Carlo error.
    """
    x = np.arange(10.0)
    covariance = gs.Gaussian(dim=1, len_scale=5).covariance(np.abs(np.subtract.outer(x, x)))
    rule = LithotypeRule.from_proportions([0.5, 0.5])
    observed = np.array([0] * 5 + [1] * 5)
    exact = _rejection(rule, [covariance], observed, 20_000, 0)[0]
    chains = 300

    sampled = _gibbs(rule, [covariance], observed, chains)[0]

    error = exact.std(axis=-1) / np.sqrt(chains)
    assert np.all(np.abs(sampled.mean(axis=-1) - exact.mean(axis=-1)) < 4.5 * error)
    np.testing.assert_allclose(sampled.std(axis=-1), exact.std(axis=-1), rtol=0.2)
    assert np.abs(sampled).max() < 6


def test_conditioning_two_fields_matches_rejection_sampling() -> None:
    rule = LithotypeRule.on_box(CODES_2D, extent=2.0)
    x = np.array([0.0, 1.0, 3.0])
    covariance = np.exp(-np.abs(np.subtract.outer(x, x)) / 2.0)  # exponential, rho = 0.61
    observed = np.array([0, 2, 1])
    exact = _rejection(rule, [covariance, covariance], observed, 20_000, 1)
    chains = 300

    sampled = _gibbs(rule, [covariance, covariance], observed, chains)

    error = exact.std(axis=-1) / np.sqrt(chains)
    assert np.all(np.abs(sampled.mean(axis=-1) - exact.mean(axis=-1)) < 4.5 * error)
    np.testing.assert_allclose(sampled.std(axis=-1), exact.std(axis=-1), rtol=0.2)


def test_split_facies_cannot_be_conditioned() -> None:
    """A checkerboard's facies 0 is in two parts no one-field move links."""
    rule = LithotypeRule(np.array([[0, 1], [1, 0]]), (np.array([0.0, 1.0]), np.array([0.0, 1.0])))
    with pytest.raises(ValueError, match="not connected"):
        condition_latent(rule, [np.eye(1), np.eye(1)], np.array([0]), np.random.default_rng())
    # Truncation alone is fine with split regions.
    np.testing.assert_array_equal(rule.facies(np.array([[-1.0, 1.0], [-1.0, 1.0]])), [0, 0])


def test_the_box_is_cut_into_equal_bins() -> None:
    rule = LithotypeRule.on_box(np.arange(6), extent=3.0)

    # Bins of width 1: (-inf, -2), [-2, -1), ..., [2, inf).
    np.testing.assert_array_equal(
        rule.facies(np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 9.0]])), np.arange(6)
    )
    expected = np.diff(special.ndtr(np.array([-np.inf, -2, -1, 0, 1, 2, np.inf])))
    np.testing.assert_allclose(rule.proportions(), expected, atol=1e-14)


def test_conditioning_cells_listed_twice_are_merged_or_refused() -> None:
    prior = PGSPrior([FIELDS[0]], proportions=[0.5, 0.5],
                     conditioning={"cells": [[1, 1, 1], [1, 1, 1]], "facies": [1, 1]})  # fmt: skip
    assert prior.cells is not None
    assert prior.cells.shape == (1, 3)
    with pytest.raises(ValueError, match="given facies 1 and 0"):
        PGSPrior([FIELDS[0]], proportions=[0.5, 0.5],
                 conditioning={"cells": [[1, 1, 1], [1, 1, 1]], "facies": [1, 0]})  # fmt: skip


def test_no_conditioning_data_is_the_unconditional_prior() -> None:
    variables = _variables((8, 6, 1), ("y",))
    plain = PGSPrior([FIELDS[0]], proportions=[0.5, 0.5])
    empty = PGSPrior([FIELDS[0]], proportions=[0.5, 0.5], conditioning={"cells": [], "facies": []})

    np.testing.assert_array_equal(
        plain.generate_joint(variables, 4, np.random.default_rng(2))["y"],
        empty.generate_joint(variables, 4, np.random.default_rng(2))["y"],
    )


def test_3d_anisotropic_conditioning_is_honoured() -> None:
    """A vertical well of three cells, anisotropic cells, a smooth model."""
    cells = [[3, 2, 1], [3, 2, 2], [3, 2, 3], [6, 4, 2]]
    observed = [0, 0, 1, 1]
    prior = PGSPrior(
        [{"model": "Gaussian", "len_scale": [6, 4, 1]}], proportions=[0.5, 0.5],
        conditioning={"cells": cells, "facies": observed}, cell_size=(2.0, 1.0, 0.5),
        gibbs_sweeps=10,
    )  # fmt: skip
    grid = (7, 5, 3)
    fields = prior.generate_joint(_variables(grid, ("y",)), 6, np.random.default_rng(4))["y"]

    index = [(i - 1) + grid[0] * (j - 1) + grid[0] * grid[1] * (k - 1) for i, j, k in cells]
    rule = LithotypeRule.from_proportions([0.5, 0.5])
    for member in range(6):
        np.testing.assert_array_equal(rule.facies(fields[index, member][None, :]), observed)
    assert np.abs(fields).max() < 8


def test_the_priors_facies_proportions_and_variogram_match_the_model(tmp_path: Path) -> None:
    np.save(tmp_path / "rule.npy", CODES_2D)
    prior = PGSPrior(
        [{"model": "Exponential", "len_scale": 3}, {"model": "Exponential", "len_scale": 3}],
        rule_file=tmp_path / "rule.npy",
    )
    grid = (40, 40, 1)
    fields = prior.generate_joint(_variables(grid, ("y1", "y2")), 40, np.random.default_rng(8))
    rule = LithotypeRule.on_box(CODES_2D, 3.0)

    facies = rule.facies(np.stack([fields["y1"].ravel(), fields["y2"].ravel()]))
    np.testing.assert_allclose(
        np.bincount(facies, minlength=3) / facies.size, rule.proportions(), atol=0.03
    )
    # Correlation at a lag of 3 cells along i (Eclipse order: i fastest): exp(-1).
    y = fields["y1"].reshape(grid[1], grid[0], -1)  # (j, i, member)
    lagged = np.mean(y[:, :-3] * y[:, 3:]) / np.mean(y * y)
    assert lagged == pytest.approx(np.exp(-1.0), abs=0.06)


def test_the_sampler_converges_from_its_rough_fallback_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real cases (many well cells) rarely find a feasible unconditional start."""
    from apex.algorithms.facies import rules

    monkeypatch.setattr(rules, "START_TRIES", 0)
    x = np.arange(10.0)
    covariance = gs.Gaussian(dim=1, len_scale=5).covariance(np.abs(np.subtract.outer(x, x)))
    rule = LithotypeRule.from_proportions([0.5, 0.5])
    observed = np.array([0] * 5 + [1] * 5)
    exact = _rejection(rule, [covariance], observed, 20_000, 2)[0]
    chains = 300

    sampled = _gibbs(rule, [covariance], observed, chains)[0]

    error = exact.std(axis=-1) / np.sqrt(chains)
    assert np.all(np.abs(sampled.mean(axis=-1) - exact.mean(axis=-1)) < 4.5 * error)
    np.testing.assert_allclose(sampled.std(axis=-1), exact.std(axis=-1), rtol=0.2)


def test_a_facies_on_two_disjoint_intervals_is_sampled_on_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Facies 0 is low or high values (codes 0, 1, 0): only the coordinate move jumps."""
    from apex.algorithms.facies import rules

    monkeypatch.setattr(rules, "START_TRIES", 0)
    rule = LithotypeRule.on_box(
        np.array([0, 1, 0]), extent=1.5
    )  # bins (-inf,-.5) [-.5,.5) [.5,inf)
    x = np.array([0.0, 1.0, 2.0])
    covariance = np.exp(-np.abs(np.subtract.outer(x, x)) / 1.5)
    observed = np.array([0, 1, 0])
    exact = _rejection(rule, [covariance], observed, 20_000, 3)[0]
    chains = 400

    sampled = _gibbs(rule, [covariance], observed, chains)[0]

    # Both parts of facies 0 are visited, in the exact proportion (within Monte Carlo error).
    high_exact, high = (exact[0] > 0).mean(), (sampled[0] > 0).mean()
    assert abs(high - high_exact) < 4.5 * np.sqrt(high_exact * (1 - high_exact) / chains)
    error = exact.std(axis=-1) / np.sqrt(chains)
    assert np.all(np.abs(sampled.mean(axis=-1) - exact.mean(axis=-1)) < 4.5 * error)
