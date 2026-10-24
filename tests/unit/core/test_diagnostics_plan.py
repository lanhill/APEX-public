"""Resolution of the diagnostics plan: ``[output]`` names and plugin switches."""

import numpy as np
import pytest

from apex.core.diagnostics import DiagnosticsPlan, RecordingDiagnostics

AVAILABLE = {
    "strategy": {"gain": "K", "singular_values": "s"},
    "localization": {"taper": "rho", "gain": "localized K"},
}


def _on(plan: DiagnosticsPlan) -> set[str]:
    return {f"{role}.{name}" for role, name in plan.items}


def test_empty_by_default() -> None:
    assert _on(DiagnosticsPlan.build(AVAILABLE)) == set()


def test_bare_names_apply_to_every_role_qualified_to_one() -> None:
    assert _on(DiagnosticsPlan.build(AVAILABLE, ["gain"])) == {"strategy.gain", "localization.gain"}
    assert _on(DiagnosticsPlan.build(AVAILABLE, ["strategy.gain"])) == {"strategy.gain"}


def test_plugin_switches_win_both_ways_and_warn_when_contradicting() -> None:
    plan = DiagnosticsPlan.build(
        AVAILABLE,
        ["gain"],
        {"strategy": {"gain": False, "singular_values": True}},
    )
    assert _on(plan) == {"localization.gain", "strategy.singular_values"}
    assert plan.items[("strategy", "singular_values")].source == "plugin"
    assert plan.items[("localization", "gain")].source == "[output]"
    assert len(plan.warnings) == 1
    assert "strategy.gain is not saved" in plan.warnings[0]


def test_plugin_settings_override_the_run_wide_ones() -> None:
    plan = DiagnosticsPlan.build(
        AVAILABLE, ["gain"], {"strategy": {"max_mb": 5.0, "every": 3}}, max_mb=100, every=2
    )
    assert (plan.items[("strategy", "gain")].max_mb, plan.items[("strategy", "gain")].every) == (
        5,
        3,
    )
    assert (
        plan.items[("localization", "gain")].max_mb,
        plan.items[("localization", "gain")].every,
    ) == (100, 2)


@pytest.mark.parametrize(
    ("requested", "overrides", "message"),
    [
        (["nothing"], {}, "no configured plugin produces"),
        (["method.step"], {}, "'method' plugin produces no"),
        ([], {"strategy": {"gian": True}}, "produces no 'gian'"),
        ([], {"strategy": {"gain": 1.0}}, "true or false"),
        ([], {"method": {"step": True}}, "which produce none"),
        ([], {"strategy": {"every": 0}}, "every >= 1"),
    ],
)
def test_invalid_requests_are_rejected(
    requested: list[str], overrides: dict[str, dict[str, bool | float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DiagnosticsPlan.build(AVAILABLE, requested, overrides)


def test_the_recorder_respects_every_and_the_size_limit() -> None:
    plan = DiagnosticsPlan.build(AVAILABLE, ["gain"], {"localization": {"max_mb": 1e-5}}, every=2)
    sink = RecordingDiagnostics(plan)
    formed: list[str] = []

    def gain() -> np.ndarray:
        formed.append("gain")
        return np.zeros((10, 10))

    sink.iteration = 1
    assert not sink.wants("strategy", "gain")
    sink.add("strategy", "gain", gain)
    sink.iteration = 2
    sink.add("strategy", "gain", gain)
    sink.add("localization", "gain", gain, nbytes=800)
    items, index = sink.take()
    assert set(items) == {"strategy.gain"}
    assert formed == ["gain"]  # the oversized item, with its size known, is never formed
    assert index["localization.gain"].startswith("skipped: ")
    assert sink.take() == ({}, {})
