"""Synthetic cases created from a description: the rules' five-spot, the model's
description held to its checks, the truth drawn from its own seed, and the observations of
one truth run with their noise (a fake simulator here; OPM Flow in the integration test)."""

import datetime as dt
import json
import tomllib
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.agents.converter import CaseConverter
from apex.agents.providers.stub import StubProvider
from apex.core.plugins import simulators
from apex.core.simulation import MemberInput
from apex.domain.deck_spec import five_spot
from apex.domain.observations import DataKey
from apex.services.api import ApexService
from apex.services.settings import CaseConverterSection
from apex.services.synthetic import observe_truth

SMALL = CaseConverterSection(ensemble_size=10, iterations=1)


class FakeDeck:
    """Rates and pressures from the truth's mean permeability; no water is produced."""

    def __init__(self, deck_template_file: str, **_: object) -> None:
        self.template = deck_template_file

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        level = float(np.exp(np.mean(member.state["permx"])))
        values = {"WOPR": level, "WWPR": 0.0, "WBHP": 150.0, "WWIR": 2 * level}
        return np.array([values[k.datatype.split(":")[0]] for k in keys])


@pytest.fixture
def fake_deck() -> Iterator[str]:
    simulators.add("fake_deck", FakeDeck)
    yield "fake_deck"
    simulators.remove("fake_deck")


def test_observing_the_truth_adds_seeded_noise(tmp_path: Path, fake_deck: str) -> None:
    truth = {"permx": np.full(4, np.log(100.0))}
    dates = [dt.date(2025, 2, 1), dt.date(2025, 3, 1)]

    def observe(name: str) -> str:
        observe_truth(simulator=fake_deck, options={"deck_template_file": "x"}, truth=truth,
                      vectors=["WOPR:P", "WWPR:P", "WBHP:P"], dates=dates, relative_error=0.1,
                      floor_fraction=0.01, seed=7, work_dir=tmp_path / "work",
                      output=tmp_path / name, executor="serial")  # fmt: skip
        return (tmp_path / name).read_text()

    first, again = observe("a.csv"), observe("b.csv")

    rows = first.splitlines()
    assert first == again  # seeded
    assert rows[0] == "time,datatype,value,variance"
    # A water rate of 0 is observed, its floor 1 % of the well's largest rate (WOPR 100).
    assert len(rows) == 1 + 6
    zero = next(r for r in rows if ",WWPR:P," in r).split(",")
    assert float(zero[3]) == pytest.approx((0.01 * 100.0) ** 2)
    assert not (tmp_path / "work").exists()
    value, variance = (float(x) for x in rows[1].split(",")[2:])
    assert variance == pytest.approx((0.1 * 100.0) ** 2)
    assert value != 100.0  # noise added


def test_the_rules_create_a_five_spot_that_validates_once_observed(
    tmp_path: Path, fake_deck: str
) -> None:
    service = ApexService(tmp_path / "service")
    converter = CaseConverter(service, None, settings=SMALL)
    converter.truth_executor = "serial"  # the fake simulator lives in this test module

    created = converter.create(tmp_path / "case", name="box")
    config = tmp_path / "case" / "config.toml"
    raw = tomllib.loads(config.read_text())
    raw["simulator"]["name"] = fake_deck  # the fake stands in for OPM Flow
    import tomli_w

    config.write_text(tomli_w.dumps(raw))
    observed = converter.observe(tmp_path / "case")

    assert not created.valid
    assert "apex agent convert observe" in created.validation.errors[0]
    assert created.spec == five_spot("box")
    with np.load(tmp_path / "case" / "truth.npz") as truth:
        assert truth["permx"].shape == (21 * 21 * 3,)
    assert raw["case"] == {"kind": "synthetic", "truth_file": "truth.npz"}
    assert raw["prior"]["variables"]["permx"]["name"] == "gaussian_field"
    assert observed.valid, observed.validation.errors
    assert "Not valid yet" not in (tmp_path / "case" / "CONVERSION.md").read_text()


def test_the_models_description_is_used(tmp_path: Path) -> None:
    spec = five_spot("tiny", n=7, layers=1).model_copy(update={"name": "tiny"})
    changed = json.loads(spec.model_dump_json())
    changed["properties"]["permx_md"] = 500.0
    service = ApexService(tmp_path / "service")

    created = CaseConverter(service, StubProvider([json.dumps(changed)]), settings=SMALL).create(
        tmp_path / "case", request="a small five-spot, 500 mD"
    )

    assert created.decided_by == "model"
    assert created.spec.grid.nx == 7
    raw = tomllib.loads((tmp_path / "case" / "config.toml").read_text())
    assert raw["prior"]["variables"]["permx"]["options"]["mean"] == pytest.approx(np.log(500.0))


def test_a_failed_truth_run_says_how_to_go_on(tmp_path: Path) -> None:
    """The case stays written, its notes say why, and the error names `observe`."""
    from apex.core.errors import ConfigError

    service = ApexService(tmp_path / "service")
    converter = CaseConverter(service, None, settings=SMALL)
    converter.truth_executor = "serial"

    def broken(*args: object, **kwargs: object) -> None:
        raise ConfigError("the truth run failed: flow exited 1")

    import apex.services.synthetic as synthetic

    original = synthetic.observe_truth
    synthetic.observe_truth = broken  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigError, match="apex agent convert observe"):
            converter.create(tmp_path / "case", name="box", simulate=True)
    finally:
        synthetic.observe_truth = original

    assert (tmp_path / "case" / "truth.npz").is_file()
    assert "flow exited 1" in (tmp_path / "case" / "CONVERSION.md").read_text()


def test_a_name_is_checked_as_the_descriptions_own(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="name"):
        CaseConverter(ApexService(tmp_path / "s"), None, settings=SMALL).create(
            tmp_path / "case", name="${evil}"
        )


def test_a_relative_output_folder_runs_the_truth_where_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reported issue (2026-10-24): `convert create --out box` ran the truth in
    `box/truth_run`, relative to the simulator's own folder: 'Cannot find input case'."""
    import apex.services.synthetic as synthetic
    from apex.core.errors import ConfigError

    seen: dict[str, object] = {}

    def capture(**kwargs: object) -> None:
        seen.update(kwargs)
        raise ConfigError("stopped by the test")

    monkeypatch.setattr(synthetic, "observe_truth", capture)
    monkeypatch.chdir(tmp_path)
    converter = CaseConverter(ApexService(tmp_path / "service"), None, settings=SMALL)

    with pytest.raises(ConfigError, match="stopped by the test"):
        converter.create(Path("box"), name="box", simulate=True)
    assert seen["work_dir"] == tmp_path / "box" / "truth_run"
    seen.clear()
    with pytest.raises(ConfigError, match="stopped by the test"):
        converter.observe(Path("box"))
    assert seen["work_dir"] == tmp_path / "box" / "truth_run"
    assert seen["output"] == tmp_path / "box" / "observations.csv"


def test_a_request_the_rules_replaced_is_named_and_every_value_listed(tmp_path: Path) -> None:
    """A reported issue (2026-10-24): with three empty replies, "a 30 x 30 x 4 five-spot" became
    the rules' 21 x 21 x 3 case, and only "decided by: fallback" said so; CONVERSION.md listed
    none of the values assumed."""
    from apex.core.llm import LLMUnavailable

    service = ApexService(tmp_path / "service")
    request = "a 30 x 30 x 4 five-spot, monthly for three years"

    created = CaseConverter(service, StubProvider([LLMUnavailable("down")]), settings=SMALL).create(
        tmp_path / "case", request=request
    )

    assert created.decided_by == "fallback"
    assert created.notes[0].startswith("the request was not applied (")
    assert f"{request!r}; the rules' case is written instead: 21 x 21 x 3 cells, 5 wells" in (
        created.notes[0])  # fmt: skip
    text = (tmp_path / "case" / "CONVERSION.md").read_text()
    spec = created.spec
    assert "## Values" in text
    assert f"- grid: 21 x 21 x 3 cells of {spec.grid.dx:g} x" in text
    assert f"- correlation range: {spec.properties.range_cells:g} cells" in text
    assert f"kv/kh {spec.properties.kv_kh:g}" in text
    assert "- fluid: preset oil_water, initial_pressure" in text
    assert sum(line.startswith("- well ") for line in text.splitlines()) == len(spec.wells)
    described = CaseConverter(service, None, settings=SMALL).create(tmp_path / "plain")
    assert not any("request" in note for note in described.notes)
