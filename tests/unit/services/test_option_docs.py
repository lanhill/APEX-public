"""Plugin options are described from their NumPy docstrings (a release tester's H5).

`apex plugins`, the web UI's Plugins page and the generated reference show an option's
description; it comes from the plugin's "Parameters" section, so every public option must
have one, and none may send the user to the code.
"""

import re

from apex.services.plugins import all_plugins, is_private, option_docs, plugin_info

#: Descriptions that send the reader to the code instead of saying what the option does.
CODE_POINTERS = re.compile(
    r"the module|\bsee `|\(see |:class:|:func:|:data:|``|EPF_DEFAULTS|tsvdCut", re.IGNORECASE
)


class _Base:
    """A base.

    Parameters
    ----------
    alpha, beta : float
        Two options at once, with a ``literal`` and a :class:`~a.b.Thing`.
    gamma
        Its first line
        and its second.
    """


class _Child(_Base):
    """A child.

    Parameters
    ----------
    gamma
        Said again (see `Entry`).
    delta
        See `_Base`.
    epsilon
        See `_Nowhere`.

    Notes
    -----
    zeta
        Not a parameter.
    """


def test_entries_are_parsed_and_inherited() -> None:
    docs = option_docs(_Child)

    assert docs["alpha"] == docs["beta"] == "Two options at once, with a `literal` and a `Thing`."
    assert docs["gamma"] == "Said again."  # the child's wins, its pointer aside dropped
    assert "delta" not in docs  # a pointer to a class without that entry: none
    assert "epsilon" not in docs
    assert "zeta" not in docs


def test_every_public_option_says_what_it_does() -> None:
    missing, pointing = [], []
    for info in all_plugins():
        if is_private(info.registry, info.name):
            continue
        for option, schema in info.options.get("properties", {}).items():
            text = schema.get("description") or ""
            where = f"{info.registry}:{info.name}.{option}"
            if not text:
                missing.append(where)
            elif CODE_POINTERS.search(text):
                pointing.append(f"{where}: {text[:80]}")

    assert missing == [], f"options without a description: {missing}"
    assert pointing == [], f"descriptions pointing at the code: {pointing}"


def test_summaries_do_not_point_at_the_code() -> None:
    """Nor keep reStructuredText's literals (the tester's H7: ``.DATA`` in `apex plugins`)."""
    public = [i for i in all_plugins() if not is_private(i.registry, i.name)]
    pointing = [f"{i.registry}:{i.name}: {i.summary}" for i in public
                if CODE_POINTERS.search(i.summary)]  # fmt: skip

    assert pointing == []


def test_wls_says_what_it_does() -> None:
    """The tester's H4: wls's summary and its two options."""
    info = plugin_info("variogram_estimators", "wls")
    properties = info.options["properties"]

    assert "see the module" not in info.summary
    assert properties["n_lags"]["description"]
    assert "half the data's horizontal extent" in properties["max_lag"]["description"]


def test_nested_types_are_described_too() -> None:
    """A localization's entries are a nested type: its fields have descriptions as well."""
    options = plugin_info("localizations", "distance_entries").options
    entry = options["$defs"]["Entry"]["properties"]

    assert all(field.get("description") for field in entry.values()), entry


class _Wrapped:
    """Options named over two lines, and a prose paragraph in the section.

    Parameters
    ----------
    first, second,
    third
        Three at once (m; see `Burden`).
    Labels: a paragraph, not an entry
        Its text belongs to nothing.
    """


def test_wrapped_names_prose_and_asides() -> None:
    docs = option_docs(_Wrapped)

    assert docs == dict.fromkeys(("first", "second", "third"), "Three at once (m).")
