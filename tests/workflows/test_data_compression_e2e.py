"""History matching with field data compressed by the ``wavelet`` transform."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.algorithms.data_compression import WaveletCompression
from apex.cli.main import app
from apex.core.errors import ConfigError, WorkflowError
from apex.domain.observations import DataKey, ObservationSet
from apex.services import composition

GRID = (16, 16)
N_CELLS = 16 * 16
N_STATE, N_SCALAR = 3, 4
FIELD_VARIANCE = 0.02**2
ESMDA = (
    'name = "esmda"\noptions = { inflation = [2, 2] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)
# energy < 1: the linear model's predictions have rank 3, and keeping the ~37 rounding-level
# singular values would amplify rounding differences at GIES's small dampings.
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 0.99 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 6"
)
WAVELET = {"datatype": "avo", "grid": list(GRID), "wavelet": "haar", "level": 2}


def _basis() -> np.ndarray:
    """Three smooth spatial modes, ``(n_cells, n_state)``: the field is ``Phi x``."""
    i, j = np.meshgrid(*[np.linspace(0, 1, n) for n in GRID], indexing="ij")
    modes = [np.cos(np.pi * i), np.sin(np.pi * (i + j)), np.cos(2 * np.pi * j) * i]
    return np.column_stack([m.ravel() for m in modes])


def _write_case(
    directory: Path, *, transform: str, ensemble_size: int = 40, method: str = ESMDA
) -> Path:
    """A linear case: 4 scalar data and one 16x16 field at t1; returns the config path."""
    rng = np.random.default_rng(11)
    forward = rng.standard_normal((N_SCALAR, N_STATE))
    truth = np.array([0.5, -1.0, 0.8])
    scalars = forward @ truth + 0.1 * rng.standard_normal(N_SCALAR)
    field = _basis() @ truth + np.sqrt(FIELD_VARIANCE) * rng.standard_normal(N_CELLS)
    directory.mkdir()
    np.savez(directory / "forward.npz", matrix=np.vstack([forward, _basis()]))
    np.savez(directory / "avo_t1.npz", avo=field)
    _write_csv(directory, [(f"t{i}", "y", v, 0.01) for i, v in enumerate(scalars)])
    field_section = f"""
[[observations.fields]]
time = "t1"
datatype = "avo"
file = "avo_t1.npz"
variance = {FIELD_VARIANCE}
"""
    return _write_config(
        directory,
        outputs=[(f"t{i}", "y") for i in range(N_SCALAR)],
        simulator_fields=f'[["t1", "avo", {N_CELLS}]]',
        observation_fields=field_section,
        transform=transform,
        ensemble_size=ensemble_size,
        method=method,
    )


def _write_csv(directory: Path, rows: list[tuple[str, str, float, float]]) -> None:
    lines = ["time,datatype,value,variance"] + [
        f"{t},{d},{float(v)!r},{float(var)!r}" for t, d, v, var in rows
    ]
    (directory / "observations.csv").write_text("\n".join(lines) + "\n")


def _write_config(
    directory: Path,
    *,
    outputs: list[tuple[str, str]],
    simulator_fields: str = "[]",
    observation_fields: str = "",
    transform: str = "",
    ensemble_size: int = 40,
    method: str = ESMDA,
) -> Path:
    listed = ", ".join(f'["{t}", "{d}"]' for t, d in outputs)
    config = directory / "config.toml"
    config.write_text(
        f"""
[run]
name = "case"
seed = 7

[[state.variables]]
name = "x"
size = {N_STATE}

[prior]
ensemble_size = {ensemble_size}

[observations]
file = "observations.csv"
{observation_fields}

[case]
kind = "real"

[simulator]
name = "linear"

[simulator.options]
forward_file = "forward.npz"
variables = ["x"]
outputs = [{listed}]
fields = {simulator_fields}

[method]
{method}

{transform}
"""
    )
    return config


def _wavelet_section(**options: object) -> str:
    items = {**WAVELET, **options}
    body = ", ".join(f"{k} = {_toml(v)}" for k, v in items.items())
    return f'[[data_transforms]]\nname = "wavelet"\noptions = {{ {body} }}'


def _toml(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return repr(value)


@pytest.mark.parametrize("method", [ESMDA, GIES], ids=["esmda", "gies"])
def test_compressed_run_equals_the_explicitly_compressed_linear_problem(
    tmp_path: Path, method: str
) -> None:
    """Hard thresholding selects wavelet coefficients: a linear map ``S W`` of the field.

    So a run with the transform must equal a plain run on the linear model ``[G; S W Phi]``
    with the compressed observation and the estimated noise variances, built here by hand
    from an independently fitted representation (same seed and data count, so the same
    random draws).
    """
    directory = tmp_path / "compressed"
    compressed = composition.start(
        _write_case(directory, transform=_wavelet_section(), method=method)
    )

    field = np.load(directory / "avo_t1.npz")["avo"]
    transform = WaveletCompression(**WAVELET)  # type: ignore[arg-type]
    transform.fit(
        ObservationSet([DataKey("t1", "avo", i) for i in range(N_CELLS)], field, np.ones(N_CELLS))
    )
    fitted = transform.fields["t1"]
    assert 0 < fitted.n_retained < N_CELLS  # thresholding dropped coefficients

    explicit = tmp_path / "explicit"
    explicit.mkdir()
    forward = np.load(directory / "forward.npz")["matrix"]
    reduced = fitted.coefficients(forward[N_SCALAR:])[fitted.retained]  # S W Phi
    np.savez(explicit / "forward.npz", matrix=np.vstack([forward[:N_SCALAR], reduced]))
    observed = np.loadtxt(directory / "observations.csv", delimiter=",", skiprows=1, usecols=2)
    rows = [(f"t{i}", "y", v, 0.01) for i, v in enumerate(observed)] + [
        ("t1", f"c{k}", v, s**2)
        for k, (v, s) in enumerate(zip(fitted.observed, fitted.noise_std, strict=True))
    ]
    _write_csv(explicit, rows)
    config = _write_config(explicit, outputs=[(t, d) for t, d, _, _ in rows], method=method)
    plain = composition.start(config)

    # The same numbers through the same code path; only the matrix products are grouped
    # differently (one stacked G versus scalar rows plus transformed field), so rounding.
    np.testing.assert_allclose(compressed.posterior, plain.posterior, rtol=1e-10, atol=1e-12)
    assert (
        compressed.history[-1].data_mismatch_mean < 0.2 * compressed.history[0].data_mismatch_mean
    )


@pytest.mark.parametrize("method", [ESMDA, GIES], ids=["esmda", "gies"])
def test_compressed_runs_resume_identically(tmp_path: Path, method: str) -> None:
    section = _wavelet_section()
    whole = composition.start(_write_case(tmp_path / "whole", transform=section, method=method))
    config = _write_case(tmp_path / "split", transform=section, method=method)
    composition.start(config, max_steps=2)
    resumed = composition.resume(tmp_path / "split" / "runs" / "case")
    np.testing.assert_array_equal(resumed.posterior, whole.posterior)


def test_resume_refuses_a_changed_data_space(tmp_path: Path) -> None:
    """Refitting on resume must reproduce the checkpointed coefficients, or stop."""
    directory = tmp_path / "changed"
    composition.start(_write_case(directory, transform=_wavelet_section()), max_steps=1)
    field = np.load(directory / "avo_t1.npz")["avo"]
    np.savez(directory / "avo_t1.npz", avo=field + 0.5 * np.sign(field))
    with pytest.raises(WorkflowError, match="differ from those the run was started with"):
        composition.resume(directory / "runs" / "case")


def test_soft_thresholding_and_coloured_noise_run(tmp_path: Path) -> None:
    section = _wavelet_section(thresholding="soft", noise="coloured", keep_approximation=True)
    result = composition.start(_write_case(tmp_path / "soft", transform=section))
    assert result.complete
    assert result.history[-1].data_mismatch_mean < result.history[0].data_mismatch_mean


def test_uncompressed_fields_are_assimilated_as_they_are(tmp_path: Path) -> None:
    result = composition.start(_write_case(tmp_path / "raw", transform=""))
    assert result.history[-1].data_mismatch_mean < result.history[0].data_mismatch_mean


def test_report_plots_scalar_data_only(tmp_path: Path) -> None:
    config = _write_case(tmp_path / "report", transform=_wavelet_section())
    composition.start(config)
    result = CliRunner().invoke(
        app, ["report", str(tmp_path / "report" / "runs" / "case"), "--plots"]
    )
    assert result.exit_code == 0, result.output
    data = tmp_path / "report" / "runs" / "case" / "report" / "data"
    # Scalar data only (the y type), not the avo field nor its wavelet coefficients.
    assert sorted(p.name for p in data.glob("*.png")) == ["y_final.png", "y_initial.png"]


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (_wavelet_section(datatype="seis"), "no field observations of 'seis'"),
        (_wavelet_section(grid=[8, 8]), "one value per active cell"),
        ('[[data_transforms]]\nname = "nope"', "nope"),
        (_wavelet_section(threshold_rule="bayesian"), "coloured"),
    ],
)
def test_transform_configuration_errors(tmp_path: Path, section: str, message: str) -> None:
    config = _write_case(tmp_path / "bad", transform=section)
    with pytest.raises(ConfigError, match=message):
        composition.start(config)


def test_a_mask_file_is_loaded(tmp_path: Path) -> None:
    config = _write_case(tmp_path / "masked", transform=_wavelet_section(mask_file="mask.npz"))
    np.savez(tmp_path / "masked" / "mask.npz", mask=np.ones(GRID, dtype=bool))
    run = composition.build_run(composition.load_config(config))
    transform = run.workflow.transforms[0]
    assert isinstance(transform, WaveletCompression)
    assert transform.mask is not None
    assert transform.mask.shape == GRID


@pytest.mark.parametrize("method", [ESMDA, GIES], ids=["esmda", "gies"])
def test_raw_forecasts_are_saved_on_request(tmp_path: Path, method: str) -> None:
    section = _wavelet_section() + "\n\n[output]\nsave_raw_forecasts = true"
    directory = tmp_path / "raw"
    composition.start(_write_case(directory, transform=section, method=method))
    checkpoints = directory / "runs" / "case" / "checkpoints"
    run = composition.build_run(composition.load_config(directory / "config.toml"))
    [transform] = run.workflow.transforms

    files = sorted(checkpoints.glob("iter_*.npz"))
    assert files
    for path in files:
        arrays = np.load(path)
        raw, predicted = arrays["raw_predicted"], arrays["predicted"]
        assert raw.shape[0] == N_SCALAR + N_CELLS  # the simulator's values, untransformed
        np.testing.assert_array_equal(transform.apply(raw), predicted)


def test_raw_forecasts_are_not_saved_by_default(tmp_path: Path) -> None:
    directory = tmp_path / "default"
    composition.start(_write_case(directory, transform=_wavelet_section()))
    arrays = np.load(directory / "runs" / "case" / "checkpoints" / "iter_1.npz")
    assert "raw_predicted" not in arrays
