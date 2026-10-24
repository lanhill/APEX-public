"""The case-converter's ``ert`` skill: ERT cases to and from APEX cases.

Both ways are code, with no model: ``apex ert convert`` (an ERT case whose forward model
is FLOW or ECLIPSE on a deck, as a native APEX case) and ``apex ert export``
(an APEX case as a self-contained ERT case). Registered by entry point
(``apex.converter_skills``), so that only a use of it imports ERT's code (the ``ert``
extra).
"""

import importlib.util
from pathlib import Path
from typing import ClassVar

from apex.core.conversion import (
    Capability,
    ConversionResult,
    ConversionUsageError,
    ConverterSkill,
)


class ErtSkill(ConverterSkill):
    """ERT cases (``.ert``): to APEX cases (``apex ert convert``) and back (``apex ert export``)."""

    name = "ert"
    formats: ClassVar[tuple[str, ...]] = ("ert_case", "apex_config")
    capabilities: ClassVar[frozenset[Capability]] = frozenset({"convert_to", "convert_from"})
    guidance = (
        "ERT (Equinor's ensemble reservoir tool) runs forward models on a case file (.ert); "
        "cases whose forward model is FLOW or ECLIPSE on a DATA_FILE become native APEX "
        "cases, and APEX cases become self-contained ERT cases."
    )
    unsupported: ClassVar[tuple[str, ...]] = (
        "ERT forward models other than FLOW or ECLIPSE (they stay on APEX_HISTORY_MATCH)",
        "SURFACE parameters, fields with transforms, FORWARD_INIT, GEN_KW distributions "
        "beyond normal, lognormal, uniform, logunif, truncated_normal, const and raw",
        "to ERT: only cases on opm_flow or eclipse decks that `apex ert export` takes (it "
        "names what it refuses)",
    )

    def accepts(self, source: Path) -> bool:
        """Whether ``source`` is an ERT case (``.ert``)."""
        return source.suffix.lower() == ".ert"

    def convert_from(self, source: Path, output: Path) -> ConversionResult:
        """Write the ERT case at ``source`` as a native APEX case in ``output``."""
        _need_ert()
        from apex.integrations.ert.convert import convert_case

        converted = convert_case(source, output)
        return ConversionResult(converted.config, list(converted.notes))

    def convert_to(self, config: Path, output: Path) -> ConversionResult:
        """Write the APEX case at ``config`` as an ERT case in ``output``."""
        _need_ert()
        from apex.integrations.ert.export import export_case

        exported = export_case(config, output)
        notes = [f"{exported.realizations} realizations, {exported.assimilated} data "
                 f"assimilated, {exported.held_out} held out; run it with `apex ert run "
                 f"{output} {' '.join(exported.run_arguments)}`"]  # fmt: skip
        return ConversionResult(exported.case, notes)


def _need_ert() -> None:
    """Refuse plainly when ERT is not installed (the ``ert`` extra)."""
    if importlib.util.find_spec("ert") is None:
        raise ConversionUsageError("ERT is not installed: install APEX with the ert extra")
