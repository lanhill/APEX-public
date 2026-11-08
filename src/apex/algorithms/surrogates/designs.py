"""Designs: the inputs of the full runs a surrogate is trained and gated on.

Pure functions, drawing only from the generator they are given (the build's stream
``design``). Every design has training runs and held-out runs, and **the held-out runs are
drawn first**: for a given seed they stay the same whatever the number of training runs, so
gates of designs of different sizes compare like with like.

- ``lhs``: a Latin hypercube over a box of inputs (McKay, Beckman & Conover 1979,
  *Technometrics* 21(2), 239-245): each coordinate's range is cut into ``n`` equal strata and
  every stratum holds exactly one point, at a uniform place within it, the strata matched
  across coordinates by independent random permutations;
- ``prior``: draws from the case's prior generator (the caller's ``draw``), so fields keep
  their geostatistics;
- ``file`` and ``run``: columns of a given ensemble (a stored ensemble, or the forecasts an
  existing run holds), the held-out columns chosen at random first.

A design may also have **coarse-only** runs (``n_coarse_only``: members a build runs on its
coarse stack only), drawn last, after the training runs, so that a design without them is
drawn exactly as before.

Arrays are ``{variable: (size, n_runs)}``, C-contiguous, the run axis last.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.core.plugins import surrogate_designs
from apex.core.surrogates import BuiltinDesign

#: Core's plans; packages add others through ``surrogate_designs``.
BUILTIN_PLANS = ("prior", "lhs", "file", "run")
DesignPlan = str


# One class per plan, each with its own summary: registered as one class, ``lhs``, ``file``
# and ``run`` were listed as aliases of ``prior`` (release tester H36).
class PriorDesign(BuiltinDesign):
    """Draws from the case's prior generators, so fields keep their geostatistics."""


class LatinHypercubeDesign(BuiltinDesign):
    """A Latin hypercube over the state or an optimization's controls (``[design.box]``)."""


class FileDesign(BuiltinDesign):
    """Columns of a stored ensemble, of the state or of controls (``ensemble_file``)."""


class RunDesign(BuiltinDesign):
    """The forecasts a history-matching run already holds (``run_dir``): no simulator runs."""


_BUILTIN_CLASSES: dict[str, type[BuiltinDesign]] = {
    "prior": PriorDesign,
    "lhs": LatinHypercubeDesign,
    "file": FileDesign,
    "run": RunDesign,
}


def _register_builtins() -> None:
    for name in BUILTIN_PLANS:
        surrogate_designs.register(name)(_BUILTIN_CLASSES[name])


_register_builtins()

#: Draws ``n`` runs' inputs, ``{variable: (size, n)}``, from the generator given.
Draw = Callable[[int, np.random.Generator], Mapping[str, NDArray[np.float64]]]


@dataclass(frozen=True, eq=False)
class Design:
    """The inputs of a design's runs.

    Attributes
    ----------
    plan
        How they were chosen.
    training
        ``{variable: (size, n_training)}``.
    held_out
        ``{variable: (size, n_held_out)}``: never used in a fit.
    training_columns, held_out_columns
        For a design taken from a given ensemble, the columns of it each set holds;
        ``None`` for drawn designs.
    coarse_only
        ``{variable: (size, n_coarse_only)}``, the coarse-only runs; ``None`` without them.
    coarse_only_columns
        For a design taken from a given ensemble, their columns; else ``None``.
    """

    plan: DesignPlan
    training: Mapping[str, NDArray[np.float64]]
    held_out: Mapping[str, NDArray[np.float64]]
    training_columns: tuple[int, ...] | None = None
    held_out_columns: tuple[int, ...] | None = None
    coarse_only: Mapping[str, NDArray[np.float64]] | None = None
    coarse_only_columns: tuple[int, ...] | None = None


def _counts(n_training: int, n_held_out: int) -> None:
    if n_training < 1 or n_held_out < 1:
        raise ValueError(
            f"a design needs at least one training and one held-out run, got {n_training} "
            f"and {n_held_out}"
        )


def latin_hypercube(
    lower: ArrayLike, upper: ArrayLike, n_points: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return ``n_points`` points of a Latin hypercube in the box, ``(n_coordinates, n_points)``.

    Coordinate ``i`` of point ``j`` is ``lower_i + (upper_i - lower_i) (p_ij + u_ij) / n``,
    with ``p_i`` a random permutation of ``0..n-1`` and ``u_ij`` uniform on ``[0, 1)``: exactly
    one point per stratum ``[k/n, (k+1)/n)`` of each coordinate. The draws: first the
    ``(n_coordinates, n_points)`` uniforms, then one permutation per coordinate, in order.

    Raises
    ------
    ValueError
        If the bounds are not finite, ``lower > upper``, or ``n_points < 1``.
    """
    low = np.asarray(lower, dtype=np.float64).reshape(-1)
    high = np.asarray(upper, dtype=np.float64).reshape(-1)
    if low.shape != high.shape or not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("a Latin hypercube needs finite bounds of equal sizes")
    if np.any(low > high):
        raise ValueError("a Latin hypercube needs lower <= upper")
    if n_points < 1:
        raise ValueError("a Latin hypercube needs at least one point")
    uniforms = rng.random((low.size, n_points))
    strata = (
        np.stack([rng.permutation(n_points) for _ in range(low.size)])
        if low.size
        else (np.zeros((0, n_points), dtype=np.int64))
    )
    unit = (strata + uniforms) / n_points
    return np.ascontiguousarray(low[:, None] + (high - low)[:, None] * unit)


def _split(
    stacked: NDArray[np.float64], sizes: Mapping[str, int]
) -> dict[str, NDArray[np.float64]]:
    blocks: dict[str, NDArray[np.float64]] = {}
    start = 0
    for name, size in sizes.items():
        blocks[name] = np.ascontiguousarray(stacked[start : start + size])
        start += size
    return blocks


def lhs_design(
    box: Mapping[str, tuple[ArrayLike, ArrayLike]],
    n_training: int,
    n_held_out: int,
    rng: np.random.Generator,
    n_coarse_only: int = 0,
) -> Design:
    """Return a Latin-hypercube design over a box of inputs.

    ``box`` is ``{variable: (lower (size,), upper (size,))}``; every entry of every variable is
    a coordinate. The held-out runs are one hypercube, drawn first; the training runs another,
    and the ``n_coarse_only`` coarse-only runs a third (none drawn when 0).
    """
    _counts(n_training, n_held_out)
    lows, highs, sizes = [], [], {}
    for name, (lower, upper) in box.items():
        low = np.asarray(lower, dtype=np.float64).reshape(-1)
        high = np.asarray(upper, dtype=np.float64).reshape(-1)
        if low.shape != high.shape:
            raise ValueError(f"the box of {name!r} needs bounds of equal sizes")
        lows.append(low)
        highs.append(high)
        sizes[name] = low.size
    if not sizes:
        raise ValueError("a Latin-hypercube design needs at least one variable")
    lower, upper = np.concatenate(lows), np.concatenate(highs)
    held_out = latin_hypercube(lower, upper, n_held_out, rng)
    training = latin_hypercube(lower, upper, n_training, rng)
    coarse_only = (
        _split(latin_hypercube(lower, upper, n_coarse_only, rng), sizes)
        if n_coarse_only > 0
        else None
    )
    return Design(
        plan="lhs",
        training=_split(training, sizes),
        held_out=_split(held_out, sizes),
        coarse_only=coarse_only,
    )


def _checked(
    blocks: Mapping[str, NDArray[np.float64]], n: int, what: str
) -> dict[str, NDArray[np.float64]]:
    result = {}
    for name, value in blocks.items():
        array = np.ascontiguousarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != n:
            raise ValueError(f"{what} {name!r} must be (size, {n}), got {array.shape}")
        result[name] = array
    return result


def repeated_columns(
    first: Mapping[str, NDArray[np.float64]], second: Mapping[str, NDArray[np.float64]]
) -> list[tuple[int, int]]:
    """Return the pairs ``(i, j)`` where column i of ``first`` equals column j of ``second``.

    Columns are compared exactly, every variable stacked in ``first``'s order: a held-out
    run equal to a training run would be gated on what the model was fitted on.
    """
    names = list(first)
    if not names:
        return []
    a = np.concatenate([np.asarray(first[n], dtype=np.float64) for n in names])
    b = np.concatenate([np.asarray(second[n], dtype=np.float64) for n in names])
    seen: dict[bytes, list[int]] = {}
    for j in range(b.shape[1]):
        seen.setdefault(np.ascontiguousarray(b[:, j]).tobytes(), []).append(j)
    return [
        (i, j)
        for i in range(a.shape[1])
        for j in seen.get(np.ascontiguousarray(a[:, i]).tobytes(), [])
    ]


def _refuse_repeats(held_out: Mapping[str, NDArray[np.float64]],
                    training: Mapping[str, NDArray[np.float64]]) -> None:  # fmt: skip
    pairs = repeated_columns(held_out, training)
    if pairs:
        i, j = pairs[0]
        raise ValueError(
            f"{len(pairs)} held-out run(s) repeat training runs (held-out {i} equals training "
            f"{j}): the gate would judge the model on what it was fitted on (a generator that "
            "ignores its random stream, such as a stored ensemble, or repeated columns)"
        )


def sampled_design(
    draw: Draw,
    n_training: int,
    n_held_out: int,
    rng: np.random.Generator,
    n_coarse_only: int = 0,
) -> Design:
    """Return a design drawn from a generator, e.g. the case's prior (plan ``prior``).

    ``draw(n_held_out, rng)`` is called first, then ``draw(n_training, rng)``, then (if
    ``n_coarse_only > 0``) ``draw(n_coarse_only, rng)``.

    Raises
    ------
    ValueError
        If a held-out run repeats a training run (e.g. a generator that ignores ``rng``).
    """
    _counts(n_training, n_held_out)
    held_out = _checked(draw(n_held_out, rng), n_held_out, "the drawn held-out input")
    training = _checked(draw(n_training, rng), n_training, "the drawn training input")
    if set(held_out) != set(training):
        raise ValueError("the draws gave different variables")
    _refuse_repeats(held_out, training)
    coarse_only = None
    if n_coarse_only > 0:
        coarse_only = _checked(
            draw(n_coarse_only, rng), n_coarse_only, "the drawn coarse-only input"
        )
        if set(coarse_only) != set(training):
            raise ValueError("the draws gave different variables")
    return Design(plan="prior", training=training, held_out=held_out, coarse_only=coarse_only)


def ensemble_design(
    plan: Literal["file", "run"],
    ensemble: Mapping[str, NDArray[np.float64]],
    n_held_out: int,
    rng: np.random.Generator,
    n_training: int | None = None,
    n_coarse_only: int = 0,
) -> Design:
    """Return a design from the columns of a given ensemble ``{variable: (size, n_columns)}``.

    The columns are permuted once: the first ``n_held_out`` of the permutation are held out,
    the next ``n_training`` (all the others if ``None``) train, and the next
    ``n_coarse_only`` are the coarse-only runs.

    Raises
    ------
    ValueError
        If the ensemble has fewer columns than the design needs, or a held-out column
        equals a training column.
    """
    if not ensemble:
        raise ValueError("an ensemble design needs at least one variable")
    n_columns = next(iter(ensemble.values())).shape[-1]
    blocks = _checked(ensemble, n_columns, "the ensemble's")
    wanted = n_columns - n_held_out if n_training is None else n_training
    _counts(wanted, n_held_out)
    if n_held_out + wanted + n_coarse_only > n_columns:
        asked = (
            f"{n_held_out} held-out, {wanted} training and {n_coarse_only} coarse-only"
            if n_coarse_only
            else f"{n_held_out} held-out and {wanted} training"
        )
        raise ValueError(
            f"the ensemble has {n_columns} columns, fewer than the {asked} runs asked for"
        )
    order = rng.permutation(n_columns)
    held = order[:n_held_out]
    train = order[n_held_out : n_held_out + wanted]
    training = {k: np.ascontiguousarray(v[:, train]) for k, v in blocks.items()}
    held_out = {k: np.ascontiguousarray(v[:, held]) for k, v in blocks.items()}
    _refuse_repeats(held_out, training)
    start = n_held_out + wanted
    only = order[start : start + n_coarse_only]
    return Design(
        plan=plan,
        training=training,
        held_out=held_out,
        training_columns=tuple(int(i) for i in train),
        held_out_columns=tuple(int(i) for i in held),
        coarse_only={k: np.ascontiguousarray(v[:, only]) for k, v in blocks.items()}
        if n_coarse_only > 0
        else None,
        coarse_only_columns=tuple(int(i) for i in only) if n_coarse_only > 0 else None,
    )
