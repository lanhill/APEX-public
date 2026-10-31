"""What a plugin's string options name: files, code, programs or plain text.

A plugin's options come from its factory's typed signature (``services/plugins.py``). An
option that can hold a string declares what it names with ``typing.Annotated`` and a
:class:`Role`, so that the service's ``[policy]`` knows which values to confine,
and refuses those it cannot classify instead of guessing from the option's name::

    def __init__(
        self,
        deck_template_file: CodeFile,              # a template APEX renders (code)
        executable: Program = "flow",              # a program APEX runs
        flow_args: ProgramArguments = (),          # that program's arguments
        forward_file: InputFile | None = None,     # a file APEX reads
        variables: Annotated[Sequence[str], TEXT] = (),   # names: data, not paths
    ) -> None: ...

A ``Literal`` of strings is text by its type and needs no role. A mapping (or a sequence
of mappings) whose keys name different things declares each key with :func:`mapping`;
its other keys must hold no string.
"""

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

#: What an option's strings are: a file or folder APEX reads (``input``), a template it
#: renders (``code``; Mako templates are Python), a folder it writes (``output``), a program
#: it runs (``executable``), that program's arguments (``arguments``), another machine its
#: runs go to (``host``), a program on such a machine (``remote_program``: its path
#: there, which nothing here can check), or data that names nothing on disk (``text``:
#: variable and well names, labels, patterns, values).
RoleKind = Literal[
    "input", "code", "output", "executable", "arguments", "host", "remote_program", "text",
    "mapping",
]  # fmt: skip


@dataclass(frozen=True)
class Role:
    """The declaration an option's annotation carries (see the module docstring)."""

    kind: RoleKind
    keys: Mapping[str, "Role"] = field(default_factory=dict)


INPUT = Role("input")
CODE = Role("code")
OUTPUT = Role("output")
EXECUTABLE = Role("executable")
ARGUMENTS = Role("arguments")
HOST = Role("host")
REMOTE_PROGRAM = Role("remote_program")
TEXT = Role("text")


def mapping(**keys: Role) -> Role:
    """Return the role of a mapping (or a sequence of mappings) with these keys' roles."""
    return Role("mapping", keys)


#: A multilevel ensemble's levels: each a mapping of the method's and the
#: simulator's keys. A level's own deck template is code; its forward file is read.
LEVELS = mapping(
    members=TEXT, weight=TEXT, first=TEXT, size=TEXT, upscale=TEXT, matrix=TEXT,
    deck_template_file=CODE, forward_file=INPUT,
)  # fmt: skip

InputFile = Annotated[str | Path, INPUT]
InputFolder = Annotated[str | Path, INPUT]
CodeFile = Annotated[str | Path, CODE]
Program = Annotated[str, EXECUTABLE]
ProgramArguments = Annotated[Sequence[str], ARGUMENTS]
Text = Annotated[str, TEXT]
TextList = Annotated[Sequence[str], TEXT]


def role_of(annotation: Any) -> Role | None:  # noqa: ANN401 (any annotation)
    """Return the role an annotation declares, ``TEXT`` for string ``Literal``s, else ``None``.

    ``Annotated`` is looked for through ``X | None`` and other unions, and through
    ``type X = ...`` aliases (``TypeAliasType``, whose origin ``typing`` does not report: an
    option typed by such an alias was taken as undeclared).
    """
    if isinstance(annotation, typing.TypeAliasType):
        return role_of(annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        for item in annotation.__metadata__:
            if isinstance(item, Role):
                return item
        return role_of(typing.get_args(annotation)[0])
    if origin is Literal:
        return TEXT
    for argument in typing.get_args(annotation) if origin is not None else ():
        found = role_of(argument)
        if found is not None:
            return found
    return None


def holds_text(value: object) -> bool:
    """Return whether a value has a string (or path) anywhere in it."""
    if isinstance(value, str | Path):
        return True
    if isinstance(value, Mapping):
        return any(holds_text(k) or holds_text(v) for k, v in value.items())
    if isinstance(value, list | tuple | set | frozenset):
        return any(holds_text(v) for v in value)
    return False
