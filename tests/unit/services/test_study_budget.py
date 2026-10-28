"""A study over its budget says what each arm costs and what would fit (a reported issue, #248)."""

from apex.services.study import over_budget


def test_the_refusal_names_each_arms_cost_and_the_replicates_that_fit() -> None:
    costs = [(f"toy/{arm}/base", r, 32) for r in range(3) for arm in ("enopt", "genopt")]
    message = over_budget(192, 100, costs)
    assert message.startswith("the study may take 192 simulator runs, over its budget of 100.")
    assert (
        "Runs per replicate: toy/enopt/base 32, toy/genopt/base 32 (64 in all, 3 replicates)"
        in message
    )
    assert "To fit: run 1 replicate (64 runs), or lower the cost of the costliest arm" in message


def test_one_replicate_over_the_budget_and_unbounded_arms_are_said() -> None:
    message = over_budget(160, 50, [("c/a/base", 0, 100), ("c/b/base", 0, 60)])
    assert (
        "even one replicate is over it; lower the cost of the costliest arm, c/a/base (100 runs)"
        in message
    )
    assert "or drop an arm" in message
    unbounded = over_budget(None, 50, [("c/a/base", 0, None), ("c/b/base", 0, 60)])
    assert unbounded.endswith("the runs of c/a/base cannot be bounded")
