"""The case folder's files a deck template names (``deck.case_dir_files``)."""

from pathlib import Path

import pytest

from apex.simulation.deck import case_dir_files


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("INCLUDE\n '${case_dir}/grid/Grid.grdecl' /\n", ["grid/Grid.grdecl"]),
        # GEOS's file lists: the comma and the braces are no part of the names.
        ('tableFiles="{ ${case_dir}/pvdo.txt, ${case_dir}/pvtw.txt }"', ["pvdo.txt", "pvtw.txt"]),
        ('tableFiles="{${case_dir}/a.txt,${case_dir}/b.txt}"', ["a.txt", "b.txt"]),
        ("INCLUDE\n '${case_dir}/PERM(1).GRDECL' /\n", ["PERM(1).GRDECL"]),
        # The trade-off: a call's closing parenthesis stays (the copy warns of it).
        ("include(${case_dir}/x.inc)", ["x.inc)"]),
        ("INCLUDE\n '${case_dir}/include/' /\n", ["include"]),
    ],
)
def test_the_named_files_end_at_quotes_spaces_and_list_punctuation(
    tmp_path: Path, text: str, expected: list[str]
) -> None:
    template = tmp_path / "deck.mako"
    template.write_text(text)
    assert case_dir_files(template) == expected


@pytest.mark.parametrize("path", ["grid_${level}.inc", "f(${i}).inc", "${name}.inc"])
def test_a_computed_path_needs_the_whole_folder(tmp_path: Path, path: str) -> None:
    template = tmp_path / "deck.mako"
    template.write_text(f"INCLUDE\n '${{case_dir}}/{path}' /\n")
    assert case_dir_files(template) is None
