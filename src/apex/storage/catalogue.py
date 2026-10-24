"""A run catalogue in a JSON-lines file (``<root>/catalogue.jsonl``).

Entries are appended under an exclusive file lock, so concurrent jobs can register runs;
the last entry for an id wins. Entries whose directory no longer exists are ignored.
"""

import datetime as dt
import fcntl
import json
from pathlib import Path

from apex.core.storage import CatalogueEntry


def now() -> str:
    """Return the current time, ISO 8601 in UTC, to the second."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class FileCatalogue:
    """The catalogue of one services root."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "catalogue.jsonl"

    def register(self, entry: CatalogueEntry) -> None:
        """Append ``entry``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "run_id": entry.run_id,
                "path": str(entry.path),
                "name": entry.name,
                "created": entry.created,
                "annotations": entry.annotations,
            }
        )
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line + "\n")

    def _all(self) -> dict[str, CatalogueEntry]:
        if not self.path.exists():
            return {}
        latest: dict[str, CatalogueEntry] = {}
        with self.path.open(encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                latest[raw["run_id"]] = CatalogueEntry(
                    raw["run_id"],
                    Path(raw["path"]),
                    raw["name"],
                    raw["created"],
                    raw["annotations"],
                )
        return latest

    def find(self, run_id: str) -> CatalogueEntry | None:
        """Return the live run with ``run_id``, or ``None``."""
        entry = self._all().get(run_id)
        return entry if entry is not None and entry.path.exists() else None

    def entries(self) -> list[CatalogueEntry]:
        """Return every live run, oldest first."""
        return sorted((e for e in self._all().values() if e.path.exists()), key=lambda e: e.created)
