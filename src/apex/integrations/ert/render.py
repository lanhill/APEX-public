r"""The forward-model step ``APEX_RENDER_DECK``: APEX's Mako deck in an ERT run path.

ERT writes each ``FIELD`` of an exported case into the run path (``<field>.grdecl``); this
step reads them back as APEX's state variables (named in ``apex_export.json``) and renders
the case's deck template with APEX's own :func:`~apex.simulation.deck.render_deck`, so APEX
and ERT run the same deck for the same parameters.

    python -m apex.integrations.ert.render --export apex_export.json \
        --output CASE.DATA --realization 3
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.simulation import MemberInput
from apex.simulation.deck import render_deck
from apex.simulation.ert import MEMBER_FILE
from apex.simulation.grdecl import read_grdecl


def render(export: Path, output: Path, realization: int, run_path: Path = Path()) -> None:
    """Render the exported case's deck for one realization into ``output``.

    Raises
    ------
    ValueError
        If a field's file is missing from the run path or lacks its keyword.
    """
    manifest = json.loads(export.read_text())
    state = {}
    for name, field in manifest["fields"].items():
        path = run_path / field["file"]
        if not path.exists():
            raise ValueError(f"{path} is missing: ERT writes field {name} there before this step")
        state[field["variable"]] = read_grdecl(path, name)
    state.update(read_scalar_state(manifest, run_path))
    if manifest.get("parameterized"):
        # The state is the latent fields: the deck reads the properties, as in
        # APEX's own runs.
        state = properties(Path(manifest["config"]), state)
    # The values APEX's adapters give a template without fidelity levels.
    values = {"level": -1, "coarse": [], **manifest.get("template_values", {})}
    # In APEX's batches (phase B2) the run path names the member: a deck sees APEX's id.
    named = run_path / MEMBER_FILE
    member_id = int(named.read_text()) if named.exists() else realization
    member = MemberInput(member_id, state, run_path)
    case_dir = Path(manifest["case_dir"]) if manifest.get("case_dir") else None
    deck = render_deck(Path(manifest["deck_template_file"]), member, values, case_dir)
    (run_path / output).write_text(deck)


def read_scalar_state(
    manifest: Mapping[str, Any], run_path: Path
) -> dict[str, NDArray[np.float64]]:
    """Return the scalar state variables from the run path's surface.

    Raises
    ------
    ValueError
        If the surface is missing or has another number of values.
    """
    scalars = manifest.get("scalars")
    if not scalars:
        return {}
    from apex.integrations.ert.export import read_scalars

    path = run_path / scalars["file"]
    if not path.exists():
        raise ValueError(f"{path} is missing: ERT writes the scalar surface there")
    values = read_scalars(path)
    sizes = [int(size) for _, size in scalars["variables"]]
    if values.size != sum(sizes):
        raise ValueError(f"{path} has {values.size} values; the scalars have {sum(sizes)}")
    state, start = {}, 0
    for (name, _), size in zip(scalars["variables"], sizes, strict=True):
        state[str(name)] = values[start : start + size]
        start += size
    return state


def properties(
    config_path: Path, state: Mapping[str, NDArray[np.float64]]
) -> dict[str, NDArray[np.float64]]:
    """Return a member's properties from its state, by the configuration's parameterizations."""
    from apex.services.composition import build_layout, build_parameterizations, load_config
    from apex.workflows.parameterization import PropertyMap

    config = load_config(config_path)
    return PropertyMap(build_parameterizations(config), build_layout(config)).member(state)


def main(argv: list[str] | None = None) -> int:
    """Run the step (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Render APEX's deck in an ERT run path.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realization", type=int, default=0)
    args = parser.parse_args(argv)
    render(args.export, args.output, args.realization)
    return 0


if __name__ == "__main__":
    sys.exit(main())
