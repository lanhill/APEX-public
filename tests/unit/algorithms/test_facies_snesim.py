"""The ``mpslib_snesim`` prior with a stub program.

The stub stands in for MPSlib's ``mps_snesim_tree``: it reads the parameter file APEX wrote
and writes ``n_real`` GSLIB realisations (stripes shifted by the seed), copying the hard
data unless told to break them. MPSlib itself runs in the integration test.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.facies.snesim import MpslibSnesim, read_gslib_grid, write_gslib_grid
from apex.core.plugins import priors
from apex.domain.state import StateVariable

STUB = """#!{python}
import sys, re
from pathlib import Path
text = Path(sys.argv[1]).read_text()
value = lambda key: re.search(re.escape(key) + r"[^#]*# *(.*)", text).group(1).split()
n_real, seed = int(value("Number of realizations")[0]), int(value("Random Seed")[0])
nx, ny, nz = (int(value(f"Simulation grid size {{a}}")[0]) for a in "XYZ")
dx, dy, dz = (float(value(f"Simulation grid grid cell size {{a}}")[0]) for a in "XYZ")
hard = []
name = value("HardData filename")[0]
if Path(name).exists():
    rows = Path(name).read_text().splitlines()[6:]
    hard = [[float(v) for v in r.split()] for r in rows if r.strip()]
Path("seen.txt").write_text(text)
if Path("soft.dat").exists():
    soft = Path("soft.dat").read_text()
    if int(value("Debug mode")[0]) >= 3:  # APEX's soft-data check: print the pdf used
        k = int(soft.splitlines()[1]) - 3
        pdf = [float(v) for v in soft.splitlines()[5 + k].split()[3:]]
        pdf = [1.0] if {one_category} else pdf[:-1] + [1 - sum(pdf[:-1])]
        print("SOFT: " + " ".join(f" P[{{c}}]={{p:g}}" for c, p in enumerate(pdf)))
    else:  # kept next to the stub, for the soft-data test
        (Path(__file__).parent / "soft_seen.dat").write_text(soft)
for r in range(n_real):
    codes = [((i + seed + r) // 3) % 2 for k in range(nz) for j in range(ny) for i in range(nx)]
    for x, y, z, c in hard:
        cell = int(x / dx) + nx * (int(y / dy) + ny * int(z / dz))  # MPSlib truncates
        codes[cell] = ({broken} if {broken} < 2**40 else -997799) if {broken} >= 0 else int(c)
    body = "\\n".join(str(c) for c in codes)
    Path(f"ti.dat_sg_{{r}}.gslib").write_text(f"{{nx}} {{ny}} {{nz}}\\n1\\nv\\n{{body}}\\n")
"""


def _stub(tmp_path: Path, broken: float = -1, one_category: bool = False) -> Path:
    """The stub; ``one_category`` reads one soft category, as scikit-mps 0.6.0's binary."""
    path = tmp_path / f"stub_{broken}_{one_category}.py"
    path.write_text(STUB.format(python=sys.executable, broken=broken, one_category=one_category))
    path.chmod(0o755)
    return path


def _ti(tmp_path: Path) -> Path:
    i = np.arange(20)
    image = np.stack([(i // 4) % 2] * 20, axis=1)[:, :, None]  # (20, 20, 1) stripes
    path = tmp_path / "ti.npy"
    np.save(path, image)
    return path


def _variables(n: int = 2) -> list[StateVariable]:
    return [StateVariable(name=f"d{k}", size=60, grid=(10, 6, 1)) for k in range(n)]


def test_the_prior_runs_the_program_and_returns_signed_distances(tmp_path: Path) -> None:
    prior = priors.create(
        "mpslib_snesim",
        training_image_file=str(_ti(tmp_path)),
        conditioning={"cells": [[2, 3, 1]], "facies": [1]},
        program=str(_stub(tmp_path)),
        template_size=[5, 4, 1],
        cell_size=[10.0, 10.0, 1.0],
    )
    assert isinstance(prior, MpslibSnesim)
    out = prior.generate_joint(_variables(), 3, np.random.default_rng(0))
    assert set(out) == {"d0", "d1"}
    assert out["d0"].shape == (60, 3)
    facies = np.argmax(np.stack([out["d0"], out["d1"]]), axis=0)  # (60, 3)
    cell = (2 - 1) + 10 * (3 - 1)
    assert np.all(facies[cell] == 1)  # the hard datum
    # Signed distances: positive in the cell's own facies, negative in the other.
    assert np.all((out["d1"] > 0) == (facies == 1))
    again = prior.generate_joint(_variables(), 3, np.random.default_rng(0))
    np.testing.assert_array_equal(out["d0"], again["d0"])


def test_the_parameter_file_carries_the_options(tmp_path: Path) -> None:
    stub = _stub(tmp_path)
    prior = MpslibSnesim(
        str(_ti(tmp_path)), program=str(stub), template_size=[5, 4, 2], n_multiple_grids=2,
        n_cond=9, random_path="preferential", entropy_factor=3.0,
    )  # fmt: skip
    text = prior._parameters((10, 6, 1), n_real=4, seed=77, hard="")
    assert "Number of realizations # 4" in text
    assert "Random Seed (0 `random` seed) # 77" in text
    assert "Search template size Z # 2 2" in text
    assert "Number of mulitple grids (start from 0) # 2" in text
    assert "Maximum number condtitional data (0: all) # 9" in text
    assert "# 2 3" in text  # preferential path, entropy factor 3


def test_hard_data_the_program_does_not_honour_fail_loudly(tmp_path: Path) -> None:
    prior = MpslibSnesim(
        str(_ti(tmp_path)),
        conditioning={"cells": [[1, 1, 1]], "facies": [1]},
        program=str(_stub(tmp_path, broken=0)),
    )
    with pytest.raises(RuntimeError, match="does not honour"):
        prior.generate_joint(_variables(), 2, np.random.default_rng(1))


def test_what_does_not_fit_is_refused(tmp_path: Path) -> None:
    ti, stub = str(_ti(tmp_path)), str(_stub(tmp_path))
    prior = MpslibSnesim(ti, program=stub)
    with pytest.raises(ValueError, match="2 facies"):
        prior.generate_joint(_variables(3), 2, np.random.default_rng(1))
    with pytest.raises(ValueError, match="ensemble_size"):
        prior.generate_joint(_variables(), None, np.random.default_rng(1))
    with pytest.raises(ValueError, match="outside the grid"):
        MpslibSnesim(ti, program=stub, conditioning={"cells": [[11, 1, 1]], "facies": [0]}
                     ).generate_joint(_variables(), 2, np.random.default_rng(1))  # fmt: skip
    bad = tmp_path / "bad.npy"
    np.save(bad, np.array([[0, 2], [2, 0]]))
    with pytest.raises(ValueError, match=r"0\.\.K-1"):
        MpslibSnesim(str(bad), program=stub)


def test_gslib_grids_round_trip(tmp_path: Path) -> None:
    codes = np.random.default_rng(2).integers(0, 3, size=(4, 5, 2))
    write_gslib_grid(codes, tmp_path / "g.dat")
    np.testing.assert_array_equal(read_gslib_grid(tmp_path / "g.dat"), codes)


def test_cells_mpslib_leaves_unsimulated_fail_clearly(tmp_path: Path) -> None:
    """MPSlib's -997799 (an unsimulated node) is MPSlib's problem, not the configuration's."""
    prior = MpslibSnesim(
        str(_ti(tmp_path)),
        conditioning={"cells": [[1, 1, 1]], "facies": [1]},
        program=str(_stub(tmp_path, broken=-997799 + 2**40)),
    )
    with pytest.raises(RuntimeError, match="unsimulated"):
        prior.generate_joint(_variables(), 2, np.random.default_rng(1))


def _soft(tmp_path: Path, soft: np.ndarray, name: str = "soft.npy") -> str:
    path = tmp_path / name
    np.save(path, soft)
    return str(path)


def test_soft_data_are_written_as_mpslibs_point_set(tmp_path: Path) -> None:
    """Soft probabilities (nx, ny, nz, K): one row ``x y z p_0 .. p_K-1`` per given cell."""
    soft = np.full((10, 6, 1, 2), np.nan)
    soft[1, 2, 0] = [0.2, 0.8]  # Eclipse cell (2, 3, 1)
    soft[9, 5, 0] = [0.6, 0.4]
    stub = _stub(tmp_path)
    prior = MpslibSnesim(
        str(_ti(tmp_path)), program=str(stub), cell_size=[10.0, 20.0, 1.0],
        soft_probability_file=_soft(tmp_path, soft),
    )  # fmt: skip
    prior.generate_joint(_variables(), 2, np.random.default_rng(0))
    lines = (tmp_path / "soft_seen.dat").read_text().splitlines()
    assert lines[:7] == ["soft data", "5", "x", "y", "z", "p0", "p1"]
    rows = [[float(v) for v in line.split()] for line in lines[7:]]
    # Cell centres (MPSlib truncates x / dx), full precision.
    assert rows == [[15.0, 50.0, 0.5, 0.2, 0.8], [95.0, 110.0, 0.5, 0.6, 0.4]]


def test_soft_data_that_do_not_fit_are_refused(tmp_path: Path) -> None:
    ti, stub = str(_ti(tmp_path)), str(_stub(tmp_path))
    good = np.full((10, 6, 1, 2), np.nan)
    good[0, 0, 0] = [0.5, 0.5]
    cases = {
        "must be \\(nx, ny, nz, 2\\)": good[..., :1],
        "all given or all NaN": np.where(np.arange(2) == 0, 0.5, good),
        "in \\[0, 1\\]": np.where(np.isnan(good), np.nan, [1.5, -0.5]),
        "sum to 1": np.where(np.isnan(good), np.nan, [0.5, 0.6]),
        "no cell": np.full((10, 6, 1, 2), np.nan),
    }
    three = np.full((20, 20, 1, 3), np.nan)
    three[0, 0, 0] = [0.3, 0.0, 0.7]  # uncertain with a zero: MPSlib's conflict branch
    i = np.arange(20)
    np.save(tmp_path / "ti3.npy", np.stack([(i // 4) % 3] * 20, axis=1)[:, :, None])
    with pytest.raises(ValueError, match="zero probability in an uncertain pdf"):
        MpslibSnesim(str(tmp_path / "ti3.npy"), program=stub,
                     soft_probability_file=_soft(tmp_path, three, "three.npy"))  # fmt: skip
    three[0, 0, 0] = [0.0, 0.0, 1.0]  # certain: allowed
    MpslibSnesim(str(tmp_path / "ti3.npy"), program=stub,
                 soft_probability_file=_soft(tmp_path, three, "three.npy"))  # fmt: skip
    for k, (match, soft) in enumerate(cases.items()):
        with pytest.raises(ValueError, match=match):
            MpslibSnesim(ti, program=stub, soft_probability_file=_soft(tmp_path, soft, f"{k}.npy"))
    other_grid = np.full((5, 6, 1, 2), np.nan)
    other_grid[0, 0, 0] = [0.5, 0.5]
    prior = MpslibSnesim(ti, program=stub, soft_probability_file=_soft(tmp_path, other_grid))
    with pytest.raises(ValueError, match="grid"):
        prior.generate_joint(_variables(), 2, np.random.default_rng(1))


def test_a_program_that_ignores_soft_probabilities_is_refused(tmp_path: Path) -> None:
    """scikit-mps 0.6.0's mps_snesim_tree reads one soft category (P(facies 0) = 1)."""
    soft = np.full((10, 6, 1, 2), np.nan)
    soft[1, 2, 0] = [0.2, 0.8]
    prior = MpslibSnesim(
        str(_ti(tmp_path)), program=str(_stub(tmp_path, one_category=True)),
        soft_probability_file=_soft(tmp_path, soft),
    )  # fmt: skip
    with pytest.raises(RuntimeError, match="build MPSlib from source"):
        prior.generate_joint(_variables(), 2, np.random.default_rng(0))


def test_coordinates_find_their_cell_with_fractional_cell_sizes(tmp_path: Path) -> None:
    """MPSlib locates a datum by (int)((x - 0) / dx) in float32; corners written with %g
    moved data to the previous cell for dx = 0.3 (review of 2026-10-04). Centres do not.
    """
    for size in (0.3, 0.1, 3.048, 30.48, 12.3456789):
        prior = MpslibSnesim(
            str(_ti(tmp_path)), program=str(_stub(tmp_path)), cell_size=[size, 1.0, 1.0]
        )
        for i in range(2000):
            x = np.float32(float(prior._coordinates([i, 0, 0]).split()[0]))
            assert int(x / np.float32(size)) == i


def test_soft_contradictions_and_fractional_codes_are_refused(tmp_path: Path) -> None:
    """MPSlib's conflict branch returns a probability as the facies code (three or more
    facies): a fraction, or a facies of soft probability 0, is refused.
    """
    ti, stub = str(_ti(tmp_path)), str(_stub(tmp_path))
    soft = np.full((10, 6, 1, 2), np.nan)
    soft[0, 0, 0] = [1.0, 0.0]
    prior = MpslibSnesim(ti, program=stub, soft_probability_file=_soft(tmp_path, soft))
    codes = np.zeros(60, dtype=np.int64)
    prior._check_soft_data([codes])  # facies 0, soft p = 1
    codes[0] = 1
    with pytest.raises(RuntimeError, match="soft probability 0"):
        prior._check_soft_data([codes])
    fractional = MpslibSnesim(
        ti, program=str(_stub(tmp_path, broken=0.5)),
        conditioning={"cells": [[1, 1, 1]], "facies": [1]},
    )  # fmt: skip
    with pytest.raises(RuntimeError, match="non-integer"):
        fractional.generate_joint(_variables(), 2, np.random.default_rng(1))
