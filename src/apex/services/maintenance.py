"""Housekeeping of a services root: reindex and clean up."""

import json
from pathlib import Path

from apex.core.errors import ApexError
from apex.core.storage import CatalogueEntry
from apex.services import composition
from apex.services.identity import run_id
from apex.services.validation import case_key
from apex.storage.catalogue import now
from apex.storage.run_directory import RunDirectory
from apex.storage.sqlite_store import SQLiteStore


def run_directories(root: Path, store: SQLiteStore) -> list[Path]:
    """Return the run directories to index: known ones, under ``runs/`` and ``studies/``."""
    found = {e.path for e in store.entries()}
    for pattern in ("runs/*/config.json", "studies/**/config.json"):
        found |= {p.parent for p in root.glob(pattern) if "predictions" not in p.parts}
    return sorted(found)


def reindex(root: Path, store: SQLiteStore) -> list[CatalogueEntry]:
    """Rebuild the catalogue from the run directories; return the entries written.

    A run's id comes from its ``provenance.json`` (the id it was started under), else it
    is computed from its configuration snapshot. Annotations already known are kept.
    """
    known = {e.path: e for e in store.entries()}
    written = []
    for path in run_directories(root, store):
        provenance = path / "provenance.json"
        try:
            config = composition.snapshot_config(RunDirectory(path))
        except ApexError:
            continue  # not a run directory
        previous = known.get(path)
        # A known run keeps its catalogue id: a guest's is their own (owned_run_id), not the
        # content id its provenance records.
        identity = previous.run_id if previous else None
        if identity is None and provenance.exists():
            identity = json.loads(provenance.read_text()).get("run_id")
        identity = identity or run_id(config)
        annotations = dict(previous.annotations) if previous else {}
        annotations.setdefault("case_key", case_key(config))
        entry = CatalogueEntry(
            identity, path, config.run.name, previous.created if previous else now(), annotations
        )
        store.register(entry)
        written.append(entry)
    return written
