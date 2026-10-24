"""PET itself reads what APEX exports: its reader and validator, in .venv-pet.

The round trip through APEX's importer cannot see keys PET needs that APEX ignores
(``obsname``, ``datatype``, ``reportpoint``); PET's own ``input_output.config.validate``
can. It runs in ``.venv-pet`` (PET at the reference commit), never in APEX's environment.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from apex.storage.pet_config import import_pet_case
from apex.storage.pet_export import RUNFILE, export_pet_case
from apex.testing import PET_CASES, PET_STORED, write_pet_case

REPOSITORY = Path(__file__).resolve().parents[2]
#: PET's interpreter: the repository's .venv-pet, or $APEX_PET_PYTHON (a worktree has none).
PET_PYTHON = Path(os.environ.get("APEX_PET_PYTHON", REPOSITORY / ".venv-pet" / "bin" / "python"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not PET_PYTHON.is_file(),
        reason="needs .venv-pet, PET at its reference commit (docs/migration/pet_parity.md)",
    ),
]

CHECK = """
import json, sys
from input_output.read_config import read_toml
from input_output.config import validate, unknown_keys
prb, sim, ens = read_toml(sys.argv[1])
print(json.dumps({"problems": [str(p) for p in validate(prb, sim, ens)],
                  "unknown": unknown_keys(prb, ens),
                  "dates": all(hasattr(t, "year") for t in sim["reportpoint"]),
                  "runfile": sim["runfile"]}))
"""


def _pet_reads(path: Path) -> dict[str, object]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # PET's own, never a checkout
    done = subprocess.run([str(PET_PYTHON), "-c", CHECK, str(path)], capture_output=True,
                          text=True, env=env, check=True, timeout=120)  # fmt: skip
    found: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return found


@pytest.mark.parametrize("scheme", ["esmda", "es", "lmenrml", "gnenrml"])
def test_pet_reads_the_exported_case_without_a_problem(tmp_path: Path, scheme: str) -> None:
    apex = import_pet_case(
        write_pet_case(tmp_path / "pet", PET_STORED, PET_CASES[scheme]), tmp_path / "apex"
    )
    exported = export_pet_case(apex.config, tmp_path / "out")

    found = _pet_reads(exported.path)

    assert found == {"problems": [], "unknown": [], "dates": True, "runfile": RUNFILE}
    assert (tmp_path / "out" / f"{RUNFILE}.mako").is_file()
