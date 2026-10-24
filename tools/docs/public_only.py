"""mkdocs hook of the public documentation build: public pages only.

``mkdocs.public.yml`` builds from ``docs/`` like the private build, but keeps only the pages
whose front matter says ``audience: public`` (a notebook: its ``apex`` metadata). A page is
private unless marked, so a forgotten mark cannot release it. Other files (images) are kept
only when a public page references them. With ``--strict``, a public page linking to a
dropped page, or a navigation entry naming one, fails the build.
"""

import json
import posixpath
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote


def _public(path: str) -> bool:
    """Whether a page is public.

    ``audience: public`` in its front matter, or in a notebook's ``apex`` metadata; anything
    else is private (a key given twice too), as the documentation's own checks read it. A
    test holds the two readings equal on every page.
    """
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if file.suffix == ".ipynb":
        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            return False
        meta = document.get("metadata", {}) if isinstance(document, dict) else {}
        apex = meta.get("apex", {}) if isinstance(meta, dict) else {}
        return isinstance(apex, dict) and apex.get("audience") == "public"
    return _front_matter_audience(text) == "public"


def _front_matter_audience(text: str) -> str | None:
    """Return the front matter's top-level ``audience`` (``None`` if absent or given twice)."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    found: list[str] = []
    for line in text[4:end].split("\n"):
        if line[:1].isspace():
            continue  # a nested key is not the page's
        name, colon, value = line.strip().partition(":")
        if colon and name.strip() == "audience":
            found.append(value.strip().strip("'\""))
    return found[0] if len(found) == 1 else None


#: What a page points at: Markdown links and images (``](target "title")``, ``](<a b>)``),
#: reference definitions (``[id]: target``) and HTML ``src``/``href`` attributes. In a
#: notebook's JSON the quotes are escaped (``\\"``).
_TARGETS = (
    re.compile(r"\]\(\s*<([^>]+)>"),
    re.compile(r"\]\(\s*([^)\s<>]+)"),
    re.compile(r"^\s*\[[^\]]+\]:\s*<?([^\s>]+)", re.MULTILINE),
    re.compile(r"(?:src|href)\s*=\s*\\?[\"']([^\"'\\]+)"),
)


def _referenced(page: Any) -> set[str]:  # noqa: ANN401 (a mkdocs File)
    """Return the files a page links to or shows, as paths relative to the docs folder."""
    text = Path(page.abs_src_path).read_text(encoding="utf-8")
    base = posixpath.dirname(page.src_uri)
    found = set()
    for pattern in _TARGETS:
        for target in pattern.findall(text):
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not target or "://" in target or target.startswith(("mailto:", "data:")):
                continue
            # A root-absolute link is relative to the docs folder.
            joined = target.lstrip("/") if target.startswith("/") else posixpath.join(base, target)
            found.add(posixpath.normpath(joined))
    return found


def on_files(files: Any, config: Any) -> Any:  # noqa: ANN401 (mkdocs' hook API)
    """Return the public pages, and only the other files (images) public pages reference."""
    from mkdocs.structure.files import Files

    docs = Path(config["docs_dir"]).resolve()
    pages = [f for f in files
             if f.is_documentation_page() or f.src_uri.endswith(".ipynb")]  # fmt: skip
    public = [f for f in pages if _public(f.abs_src_path)]
    wanted = {target for f in public for target in _referenced(f)}
    # The theme's own files (stylesheets, scripts, fonts) are not the documentation's: they
    # come from outside docs/ and are always kept, or the public site is unstyled.
    theme = [f for f in files if f not in pages and (
        f.abs_src_path is None
        or not Path(f.abs_src_path).resolve().is_relative_to(docs))]  # fmt: skip
    kept = [*public, *(f for f in files if f not in pages and f not in theme
                       and f.src_uri in wanted), *theme]  # fmt: skip
    return Files(kept)
