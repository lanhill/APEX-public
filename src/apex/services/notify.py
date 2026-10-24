"""Notifications when a job ends: a command, a webhook, or nothing.

Failures to notify are logged and never fatal.
"""

import json
import logging
import shlex
import string
import subprocess
import urllib.request
from collections.abc import Mapping
from typing import Any

from apex.services.settings import NotificationSection

logger = logging.getLogger(__name__)


def notify(
    settings: NotificationSection, record: dict[str, Any], extra: Mapping[str, str] | None = None
) -> None:
    """Tell the configured command or webhook that a job ended, or what ``extra`` says.

    ``extra`` adds fields the command can use (``{status}``, ``{findings}``, ... of a run
    monitor report) and the webhook receives.
    """
    fields = {
        "job": record["id"],
        "kind": record["kind"],
        "state": record["state"],
        **(extra or {}),
    }
    if settings.command:
        # A field this notification lacks (a job's end has no monitor {status}) is empty,
        # so one command serves both kinds; each part stays one argument, never shell text.
        blank = _Blank(fields)
        command = [
            string.Formatter().vformat(part, (), blank) for part in shlex.split(settings.command)
        ]
        try:
            done = subprocess.run(command, timeout=30, check=False, capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError) as error:
            logger.warning(
                "notification command failed: %s", error, extra={"event": "notify_failed"}
            )
        else:
            if done.returncode != 0:
                # e.g. a webhook script whose URL expired: say so rather than stay silent.
                detail = (done.stderr or done.stdout).strip()[-500:]
                logger.warning(
                    "notification command exited with %d: %s",
                    done.returncode,
                    detail,
                    extra={"event": "notify_failed", "returncode": done.returncode},
                )
    if settings.webhook:
        body = json.dumps(
            {**fields, "error": record.get("error"), "run_dirs": record.get("run_dirs", [])}
        )
        request = urllib.request.Request(
            settings.webhook, data=body.encode(), headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except OSError as error:
            logger.warning(
                "notification webhook failed: %s", error, extra={"event": "notify_failed"}
            )


class _Blank(dict[str, Any]):
    """Fields for ``str.format``; a missing one is an empty string."""

    def __missing__(self, key: str) -> str:
        return ""
