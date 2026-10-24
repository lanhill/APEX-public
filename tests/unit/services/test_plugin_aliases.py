"""An alias's description names the plugin it is (the release tester, 2026-10-19)."""

from apex.services.plugins import plugin_info


def test_an_alias_says_whose_it_is() -> None:
    original = plugin_info("localizations", "adaptive")
    alias = plugin_info("localizations", "adaptive_prior")
    assert not original.summary.startswith("An alias")
    assert alias.summary == f"An alias of 'adaptive'. {original.summary}"


def test_no_plugin_summary_points_at_its_module_docstring() -> None:
    """Users of ``apex plugins`` cannot read module docstrings (the release tester, 2026-10-19).

    A summary must say what the plugin is, not "(see the module docstring)".
    """
    from apex.services.plugins import all_plugins

    offenders = [
        f"{info.registry}:{info.name}: {info.summary}"
        for info in all_plugins()
        if "module docstring" in info.summary or not info.summary.strip()
    ]
    assert not offenders, "\n".join(offenders)
