"""The catalogue of examples and toy problems (services/examples.py, the Examples page).

Every example's manifest is valid and every entry's configuration validates; the cheap toy
optimizations run end to end; a web-UI user copies an example into their own folder, where
its deck templates stay in the examples root, which the file checks trust.
"""

import importlib.util
import tomllib
from pathlib import Path

import pytest

from apex.services import optimization
from apex.services.api import ApexService
from apex.services.examples import CODE_SUFFIXES, copy_example, find_examples, get_example

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.services.auth import ui_token

ROOT = Path(__file__).resolve().parents[2] / "examples"
EXAMPLES = find_examples(ROOT, strict=True)
ENTRIES = [(e.id, entry.config) for e in EXAMPLES for entry in e.manifest.example.entries]


def test_the_catalogue_has_every_workflow_and_the_toy_problems() -> None:
    workflows = {entry.workflow for e in EXAMPLES for entry in e.manifest.example.entries}
    assert {"history_matching", "optimization"} <= workflows
    ids = {e.id for e in EXAMPLES}
    problems = ("sch", "fon", "kur", "zdt1", "zdt2", "zdt3", "dtlz2", "bnh", "tnk", "srn")
    analytic = {f"toy/multi_objective/{n}" for n in problems}
    assert analytic <= ids
    assert "toy/toy_reservoir_npv_water" in ids
    assert "toy/toy_reservoir_npv_co2" in ids
    # many objectives, and statistics over realizations.
    assert {"toy/multi_objective/dtlz1", "toy/multi_objective/dtlz2_5",
            "toy/toy_reservoir_robust_npv"} <= ids  # fmt: skip
    optimizers_used = {
        tomllib.loads((ROOT / e.id / entry.config).read_text("utf-8"))["optimizer"]["name"]
        for e in EXAMPLES for entry in e.manifest.example.entries
        if entry.workflow == "optimization"
    }  # fmt: skip
    assert {"nsga2", "nsga3", "moead"} <= optimizers_used
    # Each analytic problem credits where it comes from (the user's request, 2026-10-06).
    for example in EXAMPLES:
        if example.id in analytic:
            source = example.manifest.example.source or ""
            assert "pymoo" in source or "not in pymoo" in source, example.id


@pytest.mark.parametrize(("example_id", "config"), ENTRIES)
def test_every_entry_validates(tmp_path: Path, example_id: str, config: str) -> None:
    service = ApexService(tmp_path / "service")
    path = ROOT / example_id / config
    document = tomllib.loads(path.read_text("utf-8"))
    if "optimizer" in document:
        validation = service.validate_optimization(path)
    else:
        validation = service.validate(path)
    assert validation.valid, (example_id, config, validation.errors)


#: Extras a cheap example may need and still run here, when they are installed.
AVAILABLE = {"ecalc"} if importlib.util.find_spec("libecalc") else set()
CHEAP = [
    (e.id, entry.config) for e in EXAMPLES for entry in e.manifest.example.entries
    if e.manifest.example.cost == "seconds" and set(e.manifest.example.requires) <= AVAILABLE
    and entry.workflow == "optimization" and e.id.startswith("toy/")
]  # fmt: skip


@pytest.mark.parametrize(("example_id", "config"), CHEAP)
def test_the_cheap_toy_examples_run(tmp_path: Path, example_id: str, config: str) -> None:
    example = get_example(ROOT, example_id)
    copied = copy_example(example, tmp_path)
    [path] = [p for _, _, p in copied.configs if p.name == config]
    run = optimization.optimize(path, max_iterations=5)
    assert run.result.nit >= 1


# ---------------------------------------------------------------------------- the web UI


ACCESS = """
[workspace]
folders = ["shared"]

[plugins]
optimizers = ["*"]
objectives = ["*"]
response_objectives = ["*"]
simulators = ["*"]
executors = ["*"]
methods = ["*"]
update_strategies = ["*"]
localizations = ["*"]
priors = ["*"]
metrics = ["*"]

[actions]
submit = true
examples = {examples}
"""


def _service(
    tmp_path: Path, examples: bool = True, service: str = ""
) -> tuple[TestClient, dict[str, str], Path]:
    root = tmp_path / "service"
    root.mkdir()
    (tmp_path / "shared").mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n{service}')
    (root / "ui_access.toml").write_text(ACCESS.format(examples=str(examples).lower()))
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}
    return TestClient(create_app(root)), guest, root


def test_a_guest_copies_an_example_into_their_folder_and_validates_it(tmp_path: Path) -> None:
    http, alice, _ = _service(tmp_path)
    listed = http.get("/v1/examples", headers=alice).json()["examples"]
    zdt1 = next(e for e in listed if e["id"] == "toy/multi_objective/zdt1")
    assert "pymoo" in zdt1["source"]
    assert zdt1["entries"][0]["workflow"] == "optimization"
    copied = http.post("/v1/examples:copy", headers=alice,
                       json={"id": "toy/multi_objective/zdt1"}).json()  # fmt: skip
    assert copied["folder"] == "UI/guests/alice/examples/toy/multi_objective/zdt1"
    config = next(c for c in copied["configs"] if c["path"].endswith("zdt1_nsga2.toml"))
    path = str(tmp_path / config["path"])
    checked = http.post("/v1/configs:validate", headers=alice, json={"config_file": path})
    assert checked.json()["valid"], checked.json()
    again = http.post("/v1/examples:copy", headers=alice,
                      json={"id": "toy/multi_objective/zdt1"}).json()  # fmt: skip
    assert again["folder"].endswith("zdt1-2")  # a copy never overwrites
    assert http.post("/v1/examples:copy", headers=alice,
                     json={"id": "../../etc"}).status_code == 404  # fmt: skip


def test_copying_examples_is_off_unless_the_list_allows_it(tmp_path: Path) -> None:
    """All actions are off by default; browsing the catalogue stays allowed."""
    http, alice, _ = _service(tmp_path, examples=False)
    assert http.get("/v1/examples", headers=alice).json()["examples"]
    refused = http.post("/v1/examples:copy", headers=alice, json={"id": "toy/rastrigin"})
    assert refused.status_code == 403
    assert "examples" not in http.get("/v1/whoami", headers=alice).json()["actions"]


@pytest.mark.parametrize("detour", [
    "{root}/linear_gaussian", "toy/../linear_gaussian", "./linear_gaussian", "linear_gaussian/",
    "toy//rastrigin", "/linear_gaussian",
])  # fmt: skip
def test_an_id_that_takes_a_detour_is_refused(tmp_path: Path, detour: str) -> None:
    """Review finding B1 (2026-10-06): an id climbing out and back into the examples root
    (``../../<root>/linear_gaussian``) resolved inside it, so it was accepted, and the copy
    went to the examples root itself, which every user's catalogue then listed."""
    http, alice, _ = _service(tmp_path)
    climb = "/".join([".."] * 30) + str(ROOT.resolve())
    example_id = detour.format(root=climb)
    response = http.post("/v1/examples:copy", headers=alice, json={"id": example_id})
    assert response.status_code == 404, response.json()
    assert not list(ROOT.glob("*-2")), "a copy landed in the examples root"


def test_a_second_copy_runs_under_names_of_its_own(tmp_path: Path) -> None:
    """Release tester U10: copying an example twice kept its [run] name, so the second
    copy's run always collided with the first's in the user's runs/ folder."""
    example = get_example(ROOT, "linear_gaussian")
    first = copy_example(example, tmp_path / "dest")
    second = copy_example(example, tmp_path / "dest")
    assert (first.folder.name, second.folder.name) == ("linear_gaussian", "linear_gaussian-2")
    for name in ("config_esmda.toml", "config_gies.toml"):
        original = (example.folder / name).read_text("utf-8")
        kept = (first.folder / name).read_text("utf-8")
        renamed = (second.folder / name).read_text("utf-8")
        own = tomllib.loads(original)["run"]["name"]
        assert tomllib.loads(kept)["run"]["name"] == own  # the first copy keeps its names
        assert tomllib.loads(renamed)["run"]["name"] == f"{own}-2"
        # Only the name changed: comments and every other line stay as they were.
        assert renamed.replace(f'"{own}-2"', f'"{own}"') == kept
    # An optimization's name too.
    toy = get_example(ROOT, "toy/rastrigin")
    copy_example(toy, tmp_path / "toy")
    again = copy_example(toy, tmp_path / "toy")
    ga = tomllib.loads((again.folder / "rastrigin_ga.toml").read_text("utf-8"))
    assert ga["optimization"]["name"] == "rastrigin_ga-2"
    assert ga["optimizer"]["name"] == "ga"  # other sections' names are not touched


def test_a_copys_names_skip_runs_already_in_the_users_runs_folder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run of the example's name may already be in the user's runs/ (another copy, since
    deleted, or their own file): the copy's names skip every taken one, and its plain
    ``"runs/<name>"`` references follow; others are left with a warning."""
    from apex.services.examples import _renamed_references

    example = get_example(ROOT, "linear_gaussian")
    runs = tmp_path / "runs"
    for taken in ("linear_esmda", "linear_esmda-2"):
        (runs / taken).mkdir(parents=True)
    first = copy_example(example, tmp_path / "dest", runs_dir=runs)
    assert first.folder.name == "linear_gaussian"  # the folder was free
    config = tomllib.loads((first.folder / "config_esmda.toml").read_text("utf-8"))
    assert config["run"]["name"] == "linear_esmda-3"
    # Nothing taken: the first copy keeps the example's names.
    fresh = copy_example(example, tmp_path / "fresh", runs_dir=tmp_path / "empty")
    kept = tomllib.loads((fresh.folder / "config_esmda.toml").read_text("utf-8"))
    assert kept["run"]["name"] == "linear_esmda"

    text = (
        'run_dir = "runs/lg"\nalso = "./runs/lg/predictions"\nkeep = "runs/lgx"\n'
        '# runs/lg in a comment\nabsolute = "/w/runs/lg"\n'
    )
    with caplog.at_level("WARNING"):
        renamed = _renamed_references(text, {"lg"}, "-2", example, "p.toml")
    assert tomllib.loads(renamed) == {
        "run_dir": "runs/lg-2", "also": "./runs/lg-2/predictions", "keep": "runs/lgx",
        "absolute": "/w/runs/lg",
    }  # fmt: skip
    assert "not renamed" in caplog.text


def test_copies_made_before_either_ran_get_names_of_their_own(tmp_path: Path) -> None:
    """The review of U10: with runs/linear_esmda taken, copy 1 took -2 and copy 2 (folder
    -2) took -2 too, since copy 1 had not run yet; the second submission was refused."""
    example = get_example(ROOT, "linear_gaussian")
    runs = tmp_path / "runs"
    (runs / "linear_esmda").mkdir(parents=True)
    names = []
    for _ in range(3):
        copy = copy_example(example, tmp_path / "dest", runs_dir=runs)
        names.append(tomllib.loads((copy.folder / "config_esmda.toml").read_text("utf-8"))
                     ["run"]["name"])  # fmt: skip
    assert len(set(names)) == 3
    assert "linear_esmda" not in names


def test_a_copy_never_leaves_its_destination(tmp_path: Path) -> None:
    from apex.services.examples import Example

    real = get_example(ROOT, "toy/rastrigin")
    forged = Example("../../outside", real.folder, real.manifest)
    with pytest.raises(ValueError, match="leaves the destination"):
        copy_example(forged, tmp_path / "dest")
    assert not (tmp_path / "outside").exists()


def test_a_guests_copies_count_against_their_storage(tmp_path: Path) -> None:
    http, alice, _ = _service(tmp_path, service="[guests]\nmax_storage_mb = 0.02\n")
    first = http.post("/v1/examples:copy", headers=alice, json={"id": "toy/rastrigin"})
    assert first.status_code == 200, first.json()
    refused = None
    for _ in range(40):  # each copy is a few kB: the limit is reached soon
        refused = http.post("/v1/examples:copy", headers=alice, json={"id": "toy/rastrigin"})
        if refused.status_code != 200:
            break
    assert refused is not None
    assert refused.status_code == 403
    assert "storage" in refused.json()["detail"]


def _deck_examples(tmp_path: Path) -> Path:
    """An examples root with a deck case laid out as larger cases are: an include in a
    subfolder, and a nested configuration, no entry, whose template is ``../deck.mako``."""
    import re
    import shutil

    root = tmp_path / "examples"
    case = root / "deckcase"
    shutil.copytree(ROOT / "3dbox_production_avo", case)
    (case / "grid").mkdir()
    (case / "Schdl.sch").rename(case / "grid" / "Schdl.sch")
    deck = case / "deck.mako"
    deck.write_text(deck.read_text().replace("${case_dir}/Schdl.sch", "${case_dir}/grid/Schdl.sch"))
    nested = (case / "config_esmda.toml").read_text()
    nested = re.sub(r'(_?file = ")([^"/]+")', r"\1../\2", nested)
    (case / "synthetic").mkdir()
    (case / "synthetic" / "config_esmda.toml").write_text(nested)
    assert '"../deck.mako"' in nested
    assert '"../observations.csv"' in nested
    return root


def test_a_copied_deck_template_stays_in_the_examples_root(tmp_path: Path) -> None:
    """A deck template is code: not copied, its configuration names it in the
    examples root, and a guest's file checks accept it there (the server's file)."""
    from apex.domain.config import HistoryMatchConfig
    from apex.services.access import Access, PathRules, UIAccess

    root = _deck_examples(tmp_path)
    copied = copy_example(get_example(root, "deckcase"), tmp_path / "copies")
    assert not any(p.suffix in CODE_SUFFIXES for p in copied.folder.rglob("*"))
    assert not any(p.suffix in (".py", ".ipynb") for p in copied.folder.rglob("*"))
    # Every copied configuration names the template in the root, not only the entries'.
    for other in copied.folder.rglob("*.toml"):
        text = other.read_text("utf-8")
        assert '"deck.mako"' not in text, other
        assert '"../deck.mako"' not in text, other
    [(_, _, esmda), *_] = copied.configs
    text = esmda.read_text("utf-8")
    template = str((root / "deckcase" / "deck.mako").resolve())
    assert f'"{template}"' in text
    assert "# " in text  # the configuration's comments are kept
    (tmp_path / "service").mkdir()
    (tmp_path / "service" / "service.toml").write_text(f'[examples]\nroot = "{root}"\n')
    service = ApexService(tmp_path / "service")
    config: HistoryMatchConfig = service.resolve(esmda)
    rules = UIAccess.model_validate({"workspace": {"folders": ["*"]}})
    paths = PathRules(Access(rules, []), service.workspace(), copied.folder.parent,
                      service.store.entries())  # fmt: skip
    refused = Access(rules, []).refused_names(config, paths)
    assert not [r for r in refused if "deck_template_file" in r], refused


def test_a_copied_deck_reads_its_includes_from_the_copy(tmp_path: Path) -> None:
    """The copy's data files are the ones simulated (2026-10-06: ``${case_dir}`` was the
    template's folder in the examples root, so editing a copy's grid changed nothing)."""
    from apex.services.access import Access, PathRules, UIAccess
    from apex.simulation.deck import case_folder, deck_layout

    root = _deck_examples(tmp_path)
    copied = copy_example(get_example(root, "deckcase"), tmp_path / "copies")
    assert (copied.folder / "synthetic" / "config_esmda.toml").exists()
    service = ApexService(tmp_path / "service")
    rules = UIAccess.model_validate({"workspace": {"folders": ["*"]}})
    for path in sorted(copied.folder.rglob("config_*.toml")):
        config = service.resolve(path)
        assert case_folder(config.simulator.options) == copied.folder, path
        paths = PathRules(Access(rules, []), service.workspace(), copied.folder.parent,
                          service.store.entries())  # fmt: skip
        refused = Access(rules, []).refused_names(config, paths)
        assert not [r for r in refused if "case_dir" in r], refused
    assert "case_dir" in (copied.folder / "config_esmda.toml").read_text("utf-8")
    template = (root / "deckcase" / "deck.mako").resolve()
    files = deck_layout(template, copied.folder)["files"]
    assert str(copied.folder / "grid" / "Schdl.sch") in files  # type: ignore[operator]


def test_a_parent_example_copies_into_the_folder_its_nested_examples_made(
    tmp_path: Path,
) -> None:
    # The release tester, 2026-10-19: optimization/3well first put optimization in
    # "optimization-2".
    nested = copy_example(get_example(ROOT, "optimization/3well"), tmp_path)
    parent = copy_example(get_example(ROOT, "optimization"), tmp_path)
    assert nested.folder == tmp_path.resolve() / "optimization" / "3well"
    assert parent.folder == tmp_path.resolve() / "optimization"
    again = copy_example(get_example(ROOT, "optimization"), tmp_path)
    assert again.folder.name == "optimization-2"  # a real copy is still never shared


@pytest.mark.parametrize(
    "config",
    sorted(p for p in ROOT.rglob("*.toml") if '"eclipse"' in p.read_text("utf-8")),
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_eclipse_examples_cap_by_licences_not_workers(config: Path) -> None:
    """Release-tester N31: ``licenses`` caps Eclipse; a study that moves the case to
    ``opm_flow`` drops it and keeps the executor, so ``workers`` must not repeat the cap."""
    data = tomllib.loads(config.read_text("utf-8"))
    licenses = data["simulator"]["options"]["licenses"]
    assert data["execution"]["options"]["workers"] > licenses
