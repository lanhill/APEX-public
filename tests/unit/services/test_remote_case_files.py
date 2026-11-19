"""Remote members' case files: ``prepare_remote`` takes the
files a member needs from the simulator's ``remote_case_files()`` when the stack has one,
through wrappers that do not forward attributes, and refuses it with a ``case_dir`` option.
OpenFOAM's case marker names no file, so without this the hosts would mirror nothing."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from apex.core.errors import ConfigError
from apex.core.simulation import RemoteSpec
from apex.domain.config import HistoryMatchConfig
from apex.services.composition import prepare_remote


class Executor:
    """Runs members remotely; keeps the spec it is given."""

    remote = True
    remote_code_provider = "os.path:join"  # any importable callable: never called here

    def __init__(self) -> None:
        self.spec: RemoteSpec | None = None

    def attach_remote(self, spec: RemoteSpec) -> None:
        self.spec = spec


class Lists:
    """A simulator listing its case's files."""

    def remote_case_files(self) -> Sequence[str]:
        return ["case.foam", "constant/polyMesh/points", "system/controlDict"]


class Wraps:
    """A wrapper that does not forward attributes (as geophysics' and model error's)."""

    def __init__(self, inner: object) -> None:
        self.inner = inner


class Plain:
    """A simulator without the hook."""


def _config(tmp_path: Path, **options: object) -> HistoryMatchConfig:
    template = tmp_path / "case" / "case.foam"
    template.parent.mkdir(exist_ok=True)
    template.write_text("${case_dir}/include.inc\n")
    return HistoryMatchConfig.model_validate({
        "run": {"name": "r", "seed": 1, "output_dir": str(tmp_path / "runs" / "r")},
        "state": {"variables": [{"name": "x", "size": 1}]},
        "prior": {"ensemble_size": 4, "generator": {"name": "gaussian"}},
        "observations": {"file": str(tmp_path / "obs.csv")},
        "case": {"kind": "real"},
        "simulator": {"name": "linear",
                      "options": {"deck_template_file": str(template), **options}},
        "method": {"name": "esmda", "options": {"inflation": [1.0]}},
    })  # fmt: skip


@pytest.mark.parametrize("simulator", [Lists(), Wraps(Lists()), Wraps(Wraps(Lists()))])
def test_the_simulators_list_is_the_spec_includes(tmp_path: Path, simulator: object) -> None:
    executor = Executor()
    prepare_remote(executor, simulator, _config(tmp_path))  # type: ignore[arg-type]
    assert executor.spec is not None
    assert executor.spec.includes == ("case.foam", "constant/polyMesh/points", "system/controlDict")


def test_without_the_hook_the_template_names_the_includes(tmp_path: Path) -> None:
    executor = Executor()
    prepare_remote(executor, Wraps(Plain()), _config(tmp_path))  # type: ignore[arg-type]
    assert executor.spec is not None
    assert executor.spec.includes == ("include.inc",)


def test_the_hook_with_a_case_dir_is_refused(tmp_path: Path) -> None:
    executor = Executor()
    config = _config(tmp_path, case_dir=str(tmp_path / "case"))
    with pytest.raises(ConfigError, match="cannot be combined with a case_dir option"):
        prepare_remote(executor, Wraps(Lists()), config)  # type: ignore[arg-type]
    assert executor.spec is None


def test_a_cycle_of_wrappers_ends_without_the_hook(tmp_path: Path) -> None:
    wrapper = Wraps(object())
    wrapper.inner = wrapper  # a cycle: the lookup ends, and the template names the includes
    executor = Executor()
    prepare_remote(executor, wrapper, _config(tmp_path))  # type: ignore[arg-type]
    assert executor.spec is not None
    assert executor.spec.includes == ("include.inc",)
