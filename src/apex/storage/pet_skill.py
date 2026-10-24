"""The case-converter's ``pet`` skill: PET configurations to and from APEX.

Both ways are code, with no model: ``apex pet-import``'s importer
(:mod:`apex.storage.pet_config`) and its inverse (:mod:`apex.storage.pet_export`), which
writes what the importer reads back to the same APEX case.
"""

from pathlib import Path
from typing import ClassVar

from apex.core.conversion import Capability, ConversionResult, ConverterSkill
from apex.core.plugins import converter_skills
from apex.storage.pet_config import import_pet_case
from apex.storage.pet_export import export_pet_case


@converter_skills.register("pet")
class PetSkill(ConverterSkill):
    """PET configurations (``.toml``): to APEX cases (``apex pet-import``) and back."""

    name = "pet"
    formats: ClassVar[tuple[str, ...]] = ("pet_config", "apex_config")
    capabilities: ClassVar[frozenset[Capability]] = frozenset({"convert_to", "convert_from"})
    guidance = (
        "PET (the Python Ensemble Toolbox) runs ES, ES-MDA, LM-EnRML, GN-EnRML and "
        "PET-fork's GIES with rlmmac, on decks rendered from a Mako runfile, with a "
        "stored or variogram prior and data in wide CSV tables."
    )
    unsupported: ClassVar[tuple[str, ...]] = (
        "APEX's own methods and strategies (GMAC, CHOP, adaptive_rows, the ERT strategies)",
        "parameterizations, model error, cross validation, geophysics",
        "data transforms and localization (not exported yet)",
        "template values, joint priors, priors other than a stored ensemble or a variogram",
    )

    def accepts(self, source: Path) -> bool:
        """Whether ``source`` is a PET configuration (a TOML with ``[dataassim]``)."""
        if source.suffix.lower() != ".toml" or not source.is_file():
            return False
        return "[dataassim]" in source.read_text(errors="replace")

    def convert_to(self, config: Path, output: Path) -> ConversionResult:
        """Write the APEX case at ``config`` as a PET case in ``output``."""
        result = export_pet_case(config, output)
        return ConversionResult(result.path, result.notes)

    def convert_from(self, source: Path, output: Path) -> ConversionResult:
        """Write the PET case at ``source`` as an APEX case in ``output``."""
        result = import_pet_case(source, output)
        notes = [*result.notes, *(f"ignored: {key}" for key in result.ignored)]
        return ConversionResult(result.config, notes)
