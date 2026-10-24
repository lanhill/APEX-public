"""Telling a chat that the job it launched has ended.

A study the web UI's assistant proposed, and a person launched from its card, is a job of
the job service. When it ends, the chat gets a message saying so and what it can do next
(a report, a comparison, a diagnosis), and the job's notification (a Teams card) names the
study and the chat. Failures here are logged and never fatal: the job has ended either way.

The web UI submits the study, then records the launch (two requests): a quick study can end,
and be notified, in between. :func:`after_launch` then tells the chat when the launch is
recorded; a chat already told of a job is never told again.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("apex.service")


def chat_followup(root: Path, record: dict[str, Any]) -> dict[str, str] | None:
    """Tell the chats that launched ``record``'s job that it ended; return notice fields.

    Returns the extra fields of the job's notification (``event``, ``status``,
    ``summary``), or ``None`` if no chat launched the job.
    """
    if not (Path(root) / "chats.db").exists():
        return None
    from apex.storage.chat_store import ChatStore

    try:
        store = ChatStore(Path(root))
        launches = store.launched(str(record["id"]))
        if not launches:
            return None
        name = _study_name(record) or record["id"]
        state = str(record["state"])
        reason = f" ({record['error']})" if record.get("error") else ""
        titles, told = [], 0
        for chat_id, proposal in launches:
            chat = store.get(chat_id)
            titles.append(chat.title)
            text = (
                f"The study **{name}** you launched (job `{record['id']}`) has ended: "
                f"**{state}**{reason}. "
                + (
                    "Ask me for its report, its comparison of the variants, or a diagnosis "
                    "of a run."
                    if state == "complete"
                    else "Ask me to diagnose its runs, or revise the draft and launch it again."
                )
            )
            # Once, however many tell it (the notification, a late launch's follow-up).
            job = str(record["id"])
            told += store.tell_job_ended(chat_id, job, text,
                                         {"event": "job_ended", "job": job, "state": state,
                                          "draft": proposal.draft}, state)  # fmt: skip
    except Exception as failure:  # noqa: BLE001 (a follow-up never fails the job's end)
        logger.warning(
            "could not tell the chat that job %s ended: %s", record.get("id"), failure,
            extra={"event": "chat_followup_failed", "job": record.get("id")},
        )  # fmt: skip
        return None
    if not told:
        return None  # told already, by the other teller, whose notice names the chat
    return {
        "event": "chat_job_ended",
        "status": state,
        "summary": (
            f"Study {name}, launched from the chat “{titles[0]}”, ended: {state}{reason}. "
            "Open the chat in the web UI for its report or comparison."
        ),
    }


#: Job states after which nothing more happens (as apex.services.jobs.FINAL).
_FINAL = frozenset({"complete", "stopped", "failed", "cancelled"})


def after_launch(root: Path, record: dict[str, Any]) -> None:
    """Tell the chat now if ``record``'s job ended before its launch was recorded.

    The job's own notification came first and found no chat to tell; one still to come
    tells it itself.
    """
    if record.get("parent") or record.get("state") not in _FINAL or not record.get("notified"):
        return
    chat_followup(root, record)


def _study_name(record: dict[str, Any]) -> str | None:
    spec = record.get("spec") or {}
    study = ((spec.get("study") or {}).get("study")) or {}
    name = study.get("name")
    return str(name) if name else None
