"""APEX's diagnosis and report agents for ERT users.

The ERT workflow jobs ``APEX_DIAGNOSE`` and ``APEX_REPORT`` (:mod:`.plugin`) call these. A
target is an APEX run directory (an ``APEX_HISTORY_MATCH`` run, or an import), or an
exported case's folder, whose ERT run is imported first (``<case>/apex_run``). The agents
run on their rules (no model) unless a provider is given; they read records and never spend
simulator runs.
"""

from pathlib import Path

from apex.core.errors import ConfigError
from apex.core.llm import LLMProvider

#: The file a run directory has and an exported case's folder does not.
RUN_MARKER = "state.json"


def run_directory(target: Path) -> Path:
    """Return the run directory of ``target``, importing an exported case's ERT run.

    Raises
    ------
    ConfigError
        If ``target`` is neither a run directory nor an exported case.
    """
    target = target.resolve()
    if (target / RUN_MARKER).exists():
        return target
    if (target / "apex_export.json").exists():
        imported = target / "apex_run"
        if not (imported / RUN_MARKER).exists():
            from apex.integrations.ert.importer import import_case

            import_case(target, imported)
        return imported
    raise ConfigError(f"{target} is neither an APEX run directory nor an exported ERT case")


def diagnose(target: Path, root: Path | None = None, provider: LLMProvider | None = None) -> Path:
    """Diagnose the run; write ``<run>/agents/diagnosis.md``; return its path."""
    from apex.agents.diagnosis import Diagnostician
    from apex.services.api import ApexService

    run_dir = run_directory(target)
    result = Diagnostician(ApexService(root), provider).diagnose(str(run_dir))
    d = result.diagnosis
    lines = [f"# Diagnosis of {run_dir.name}", "", f"**{d.verdict}**: {d.summary}", ""]
    lines += [f"- [{c.severity}] {c.code}: {c.evidence}" for c in d.causes]
    if d.recommendations:
        lines += ["", "## Recommendations", ""]
        lines += [
            f"{i}. {r.change} (why: {r.rationale})" for i, r in enumerate(d.recommendations, 1)
        ]
    lines += ["", f"_Decided by {result.decided_by} (APEX_DIAGNOSE)._"]
    path = run_dir / "agents" / "diagnosis.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def report(target: Path, root: Path | None = None, provider: LLMProvider | None = None) -> Path:
    """Write the run's narrative report to ``<run>/report/narrative.md``; return its path."""
    from apex.agents.reporter import Reporter
    from apex.services.api import ApexService

    run_dir = run_directory(target)
    result = Reporter(ApexService(root), provider).run_report(str(run_dir))
    path = run_dir / "report" / "narrative.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.text)
    return path
