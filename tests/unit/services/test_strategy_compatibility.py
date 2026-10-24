"""The method-strategy table is the composition's answer (the release tester, 2026-10-19).

The documentation derives its table from :func:`strategy_compatibility`, which builds each
pair as ``apex run`` does; these are pairs whose answer is known from the methods.
"""

import pytest

from apex.services.compatibility import strategy_compatibility


@pytest.fixture(scope="module")
def table() -> dict[str, dict[str, str | None]]:
    return strategy_compatibility()


def test_every_strategy_and_method_has_an_answer(table: dict[str, dict[str, str | None]]) -> None:
    from apex.services.plugins import REGISTRIES

    assert set(table) == set(REGISTRIES["update_strategies"].names())
    assert all(set(row) == set(REGISTRIES["methods"].names()) for row in table.values())


@pytest.mark.parametrize(
    ("strategy", "accepted", "refused"),
    [
        ("approx", {"es", "esmda", "gies", "lm_enrml", "gn_enrml"}, {"multilevel_esmda"}),
        ("rlmmac", {"gies", "lm_enrml", "chop"}, {"es", "esmda", "gn_enrml"}),
        ("gmac", {"gies", "lm_enrml"}, {"es", "esmda", "gn_enrml"}),
        ("full", {"gies", "lm_enrml", "gn_enrml"}, {"es", "esmda"}),
        ("subspace", {"gies", "lm_enrml", "gn_enrml"}, {"es", "esmda"}),
        ("hybrid", {"multilevel_esmda"}, {"es", "esmda", "gies"}),
        ("local_rlmmac", {"gies", "lm_enrml"}, {"esmda", "gn_enrml"}),
        ("local_analysis", {"es", "esmda", "lm_enrml", "gn_enrml"}, {"gies"}),
    ],
)
def test_known_pairs(
    table: dict[str, dict[str, str | None]], strategy: str, accepted: set[str], refused: set[str]
) -> None:
    row = table[strategy]
    assert {m for m in accepted if row[m] is not None} == set()
    assert {m for m in refused if row[m] is None} == set()


def test_es_mda_refuses_the_prior_increment_strategies_in_its_own_words(
    table: dict[str, dict[str, str | None]],
) -> None:
    for strategy in ("full", "subspace", "subspace2", "margis"):
        reason = table[strategy]["esmda"]
        assert reason is not None
        assert "prior-increment term" in reason


def test_the_guides_table_is_the_compositions(table: dict[str, dict[str, str | None]]) -> None:
    """docs/guide/plugins/update_strategies.md's method-strategy table, when it has one."""
    from pathlib import Path

    page = Path(__file__).resolve().parents[3] / "docs/guide/plugins/update_strategies.md"
    lines = page.read_text(encoding="utf-8").splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith("| Strategy |")]
    if not heads:
        pytest.skip("the guide page has no method-strategy table")
    methods = [c.strip().strip("`") for c in lines[heads[0]].strip("|").split("|")[1:]]
    rows = {}
    for line in lines[heads[0] + 2 :]:
        if not line.startswith("| `"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = dict(zip(methods, cells[1:], strict=True))
    for strategy, row in rows.items():
        for method, cell in row.items():
            expected = "yes" if table[strategy][method] is None else "no"
            assert cell == expected, (strategy, method, table[strategy][method])
    assert set(rows) == set(table)


def test_private_and_experimental_plugins_are_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """The table is for public pages (review): no private or experimental plugin in it."""
    from apex.services import compatibility

    monkeypatch.setattr(compatibility, "is_private", lambda registry, name: name == "subspace")
    from apex.services.plugins import REGISTRIES

    factory = REGISTRIES["methods"].factory("lm_enrml")
    monkeypatch.setattr(factory, "status", "experimental", raising=False)

    table = compatibility.strategy_compatibility()

    assert "subspace" not in table
    assert all("lm_enrml" not in row for row in table.values())
    assert "subspace" in compatibility.strategy_compatibility(public_only=False)


def test_a_pair_that_cannot_be_built_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """One plugin's fault is logged and answered unknown; the rest of the table stands."""
    from apex.services import compatibility

    real = compatibility._refusal

    def broken(method: str, strategy: str, layout: object) -> str | None:
        if (method, strategy) == ("gies", "approx"):
            raise RuntimeError("boom")
        return real(method, strategy, layout)  # type: ignore[arg-type]

    monkeypatch.setattr(compatibility, "_refusal", broken)

    table = compatibility.strategy_compatibility()

    assert table["approx"]["gies"] == f"{compatibility.UNKNOWN}RuntimeError: boom"
    assert table["approx"]["lm_enrml"] is None


def test_every_refusal_offers_only_methods_that_accept_the_strategy(
    table: dict[str, dict[str, str | None]],
) -> None:
    """Release tester N9 (2026-10-20): ES-MDA's refusal of gmac offered gn_enrml, which
    refuses gmac too. A method a refusal names (by its plugin name) must accept the
    strategy."""
    import re

    methods = set(next(iter(table.values())))
    wrong = []
    for strategy, row in table.items():
        for method, reason in row.items():
            if reason is None:
                continue
            for other in methods - {method}:
                named = re.search(rf"(?<![\w-]){re.escape(other)}(?![\w-])", reason)
                if named and row[other] is not None:
                    wrong.append((strategy, method, other, reason))
    assert not wrong, wrong
