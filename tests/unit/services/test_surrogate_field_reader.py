"""Field snapshots ask their reader for a mesh and states only.

A surrogate takes fields on an unstructured mesh only when it declares
``unstructured_fields``; any other needs a reader that promises a structured grid
(``structured_mesh``), refused before any run otherwise, with the reason.
"""

from collections.abc import Iterator

import pytest

from apex.core.errors import ConfigError
from apex.core.plugins import case_state_readers, surrogate_models
from apex.services.surrogates import field_reader
from apex.simulation.eclipse_files import EclipseFiles

READER, GRID_PLUGIN, MESH_PLUGIN = "test_mesh_reader", "test_grid_fields", "test_mesh_fields"


class _MeshReader:
    """A reader class of the state alone: no structured grid, no corner geometry."""


class _GridFields:
    """A field surrogate that needs a structured grid (declares nothing)."""


class _MeshFields:
    """A field surrogate that takes an unstructured mesh."""

    unstructured_fields = True


@pytest.fixture(autouse=True)
def _plugins() -> Iterator[None]:
    case_state_readers.add(READER, _MeshReader)  # type: ignore[arg-type]
    surrogate_models.add(GRID_PLUGIN, _GridFields)  # type: ignore[arg-type]
    surrogate_models.add(MESH_PLUGIN, _MeshFields)  # type: ignore[arg-type]
    try:
        yield
    finally:
        case_state_readers.remove(READER)
        surrogate_models.remove(GRID_PLUGIN)
        surrogate_models.remove(MESH_PLUGIN)


def test_the_deck_readers_promise_a_structured_grid() -> None:
    assert field_reader("opm_flow", GRID_PLUGIN) == (EclipseFiles, True)
    assert field_reader("eclipse", MESH_PLUGIN) == (EclipseFiles, False)


def test_a_mesh_reader_feeds_a_surrogate_that_takes_a_mesh() -> None:
    """No corner geometry is asked for (phase 0 asked for it)."""
    reader, structured = field_reader(READER, MESH_PLUGIN)
    assert reader is _MeshReader  # type: ignore[comparison-overlap]
    assert not structured


def test_a_grid_based_surrogate_refuses_a_mesh_reader() -> None:
    with pytest.raises(ConfigError, match=r"'test_grid_fields' needs its fields on a structured"):
        field_reader(READER, GRID_PLUGIN)


def test_a_simulator_without_a_reader_is_refused() -> None:
    with pytest.raises(ConfigError, match="the 'linear' simulator has no case-state reader"):
        field_reader("linear", GRID_PLUGIN)
