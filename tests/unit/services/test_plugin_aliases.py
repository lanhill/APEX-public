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


def test_no_schema_description_shows_docstring_markup() -> None:
    """The release tester's H8: ``apex plugins --json`` gave nested types' docstrings whole,
    with reStructuredText literals and a raw NumPy "Parameters" section."""
    import json

    from apex.services.plugins import all_plugins

    def descriptions(node: object) -> list[str]:
        if isinstance(node, dict):
            own = [node["description"]] if isinstance(node.get("description"), str) else []
            return own + [d for v in node.values() for d in descriptions(v)]
        if isinstance(node, list):
            return [d for v in node for d in descriptions(v)]
        return []

    offenders = [
        f"{info.registry}:{info.name}: {text[:80]}"
        for info in all_plugins()
        for text in descriptions(json.loads(json.dumps(info.options)))
        if "``" in text or "\n" in text or "----" in text
    ]
    assert not offenders, "\n".join(offenders)


def test_a_types_description_keeps_its_prose_and_notes() -> None:
    """Its review: the opening text and Notes stay, lists and code go, roles become code."""
    from apex.services.plugins import _prose

    doc = (
        "One entry of :class:`~a.b.Entry`.\n\nMore ``text``.\n\nParameters\n----------\n"
        "data\n    The data.\n\nNotes\n-----\nUnlike PET, it is 1-based.\n\n"
        "Examples\n--------\n>>> Entry()\n"
    )
    assert _prose(doc) == "One entry of `Entry`. More `text`. Unlike PET, it is 1-based."
    # A docstring that is all lists has no prose: its schema then has no description.
    assert _prose("Parameters\n----------\nx\n    An x.") == ""
