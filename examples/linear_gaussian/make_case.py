"""Generate the linear-Gaussian example case: ``forward.npz``, ``observations.csv``, ``truth.npz``.

The model is ``y = G x`` with 3 unknowns and 8 data, a N(0, I) prior and Gaussian data
errors, so the exact posterior is known and printed for comparison with APEX's results::

    uv run python examples/linear_gaussian/make_case.py
    uv run apex run examples/linear_gaussian/config_esmda.toml
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_STATE, N_DATA = 3, 8


def main() -> None:
    """Write the case files and print the exact posterior mean and standard deviation."""
    rng = np.random.default_rng(99)
    forward = rng.standard_normal((N_DATA, N_STATE))
    variances = rng.uniform(0.05, 0.2, N_DATA)
    truth = np.array([0.5, -1.0, 0.8])
    observations = forward @ truth + np.sqrt(variances) * rng.standard_normal(N_DATA)

    np.savez(HERE / "forward.npz", matrix=forward)
    np.savez(HERE / "truth.npz", x=truth)  # the synthetic case's true model
    rows = ["time,datatype,value,variance"] + [
        f"t{i},y,{float(observations[i])!r},{float(variances[i])!r}" for i in range(N_DATA)
    ]
    (HERE / "observations.csv").write_text("\n".join(rows) + "\n")

    gain = forward.T @ np.linalg.inv(forward @ forward.T + np.diag(variances))
    mean = gain @ observations
    std = np.sqrt(np.diag(np.eye(N_STATE) - gain @ forward))
    print(f"truth:               {truth}")
    print(f"exact posterior mean {np.round(mean, 4)}")
    print(f"exact posterior std  {np.round(std, 4)}")


if __name__ == "__main__":
    main()
