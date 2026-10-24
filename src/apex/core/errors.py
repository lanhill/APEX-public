"""Exception hierarchy shared across APEX layers."""

from pydantic import ValidationError


class ApexError(Exception):
    """Base class for all APEX errors."""


class PluginError(ApexError):
    """Raised when a plugin cannot be registered or created."""


class DuplicatePluginError(PluginError):
    """Raised when a plugin name is registered twice in the same registry."""


class UnknownPluginError(PluginError):
    """Raised when a requested plugin name is not registered."""


class ConfigError(ApexError):
    """Raised when a run configuration is invalid or refers to missing inputs."""


class SimulationError(ApexError):
    """Raised by a simulator when one member's forward run fails."""


class TransientSimulationError(SimulationError):
    """A member's run failed for a reason that may not recur, so it is worth retrying.

    For example a licence checkout, a file-system hiccup (a stale handle, an I/O error) or
    a process killed by the operating system. A simulator raises it only when it can tell;
    a failure of the simulation itself (a convergence failure, a deck error) repeats
    identically on a retry and stays a plain :class:`SimulationError`.
    """


class WorkflowError(ApexError):
    """Raised when a workflow cannot continue (e.g. too many failed members)."""


class RunStopped(WorkflowError):
    """A run was asked to stop before it reached its next checkpoint; it resumes from its last.

    Raised where a run waits outside its steps (the ssh executor waiting for a verified copy
    of APEX): the job ends ``stopped``, not ``failed``.
    """


def validation_problems(error: ValidationError, *, limit: int = 20) -> str:
    """Return a pydantic error as ``location: message`` lines, without the input values.

    ``str(error)`` repeats each wrong value (``input_value=...``): a file that is not a
    configuration, such as a ``.env``, would come back with its secrets in the error, to
    whoever asked for the validation (a tool client, a REST client; release tester N3,
    2026-10-20). The locations name the keys, the messages say what is wrong; neither
    holds a value.

    Parameters
    ----------
    error
        The validation error.
    limit
        At most this many problems are listed; the rest are counted.
    """
    problems = error.errors()
    lines = [
        f"  {'.'.join(str(p) for p in e['loc']) or '(document)'}: {e['msg']}"
        for e in problems[:limit]
    ]
    if len(problems) > limit:
        lines.append(f"  ... and {len(problems) - limit} more")
    return "\n".join(lines)
