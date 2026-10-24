---
audience: public
---

# CHOP: estimating an iterative smoother's hyper-parameters as it runs

## Goal

The method `chop` (continuous hyper-parameter optimization) wraps an iterative smoother on GIES's loop, and estimates hyper-parameters of its update during the history match: localization length scales, inflation factors of the gain, weights of the data. By the end of this page you will be able to:

- run GIES with CHOP-estimated localization length scales (P-AutoAdaLoc);
- estimate inflation factors and data weights as well;
- choose how the hyper-parameter update itself is localized;
- reproduce the published configuration;
- find the estimated hyper-parameters in a run.

The method is published: X. Luo, W. C. Cruz, X.-L. Zhang and H. Xiao (2023). Hyper-parameter optimization for improving the performance of localization in an iterative ensemble smoother. *Geoenergy Science and Engineering* 231, 212404. [doi:10.1016/j.geoen.2023.212404](https://doi.org/10.1016/j.geoen.2023.212404)

## How it works

Every member `j` carries its own hyper-parameter vector `θ_j`. In each iteration:

1. **The model update uses each member's hyper-parameters.** RLM-MAC's gain `K` is shared, and each component multiplies it element-wise for member `j`: `Δm_j = (F(θ_j) ∘ K) Δd_j`.
2. **The models are forecast**, as in GIES. A trial is accepted or rejected as in GIES.
3. **After an accepted trial, the hyper-parameters are updated** by a second RLM-MAC step: from the correlations between the hyper-parameters and the new forecast, with the same perturbed observations and the same damping `λ`. It costs no extra simulation.

## Running it

```toml
[method]
name = "chop"
max_iterations = 20
schedule = { initial = 1.0, factor = 2.0, reduction = 0.9 }
strategy = { name = "rlmmac", options = { energy = 0.99 } }

[method.options]
inner = "gies"                     # or lm_enrml (gn_enrml refuses the rlmmac strategy CHOP needs)
max_trials_per_iteration = 4       # the inner method's own options stay here too
components = [
  { name = "length_scale", mode = "data" },   # one length scale per datum (P-AutoAdaLoc-0)
]
hyper_localization = "same"        # the default
```

The inner method keeps its strategy, damping, schedule and stopping rules. CHOP needs an RLM-MAC strategy (`rlmmac`), and replaces `[method.localization]`: leave it out.

## Components

| Component | `mode` | Acts on member `j`'s update | Prior (default) |
|---|---|---|---|
| `length_scale` | `data` (one per datum), `scalar` (one per member) | Gaspari-Cohn taper `f_GC((1 − |ρ|) / (factor · ℓ_j))` on the gain | uniform `[0.23, 0.43]` |
| `gain_inflation` | `variable` (one per state variable), `scalar`, `variable_datatype` (a block matrix: per variable and data type) | multiplies the gain's rows (or blocks) | uniform `[0.5, 1.5]` |
| `data_weights` | `datatype`, `datum` | weights the innovations within the gain | uniform `[0.5, 1.5]` |

Several components may be estimated together: list them all in `components`. Their hyper-parameters are stacked, and one update estimates them all.

Every component takes:

- `low`, `high`: the uniform prior;
- `transform`: `none` (the default, natural units floored at `floor = 1e-6`), `log`, or `logit` (on `[lower, upper]`).

`length_scale` also takes:

- `correlation`: `prior` (the default, the initial ensemble's correlations ρ⁰, as the paper's formula) or `current` (recomputed before each update, as the published code);
- `factor`: 2 (the default, the published code's `2(ℓ + 1e-6)`), or 1 (the paper's formula). With 1 and the default prior, typical small correlations put `(1 − |ρ|)/ℓ` beyond Gaspari-Cohn's cut-off of 2, so the taper zeroes most of the gain: on 5Spots, one iteration barely moved the data mismatch.

## Localizing the hyper-parameter update

`hyper_localization` sets how the hyper-parameter update is tapered:

- `same` (default): each component tapers its own rows. `length_scale` uses the same Gaspari-Cohn rule on the correlations between the length scales and the new forecast, scaled by member `j`'s own length scales (the code's `update_hyper`). The other components keep the correlations above `threshold / √N` (`threshold = 3`).
- `none`: no taper (the code's `update_hyper_no_loc`).
- the name of a localization, e.g. `"adaptive_per_update"`, or `{ name = ..., options = {...} }`: applied to the hyper-parameter block.

## Reproducing the published 5Spots configuration

The paper's 5Spots P-AutoAdaLoc-0 run used the code's choices, which differ from the defaults. It also stopped when an accepted step reduced the data mismatch by less than 0.01% (`data_mismatch_tolerance = 1e-4`):

```toml
[method]
name = "chop"
max_iterations = 20
schedule = { initial = 1.0, factor = 2.0, reduction = 0.9, data_mismatch_tolerance = 1e-4 }
strategy = { name = "rlmmac", options = { energy = 0.99 } }

[method.options]
inner = "gies"
max_trials_per_iteration = 4
components = [{ name = "length_scale", mode = "data", correlation = "current" }]
hyper_localization = "none"
```

For P-AutoAdaLoc-1, use `mode = "scalar"`. The published Brugge run localized the hyper-parameter update: use `hyper_localization = "same"`. Two small differences remain from the code. The code's correlations are `(N − 1)/N` times the sample correlations, and it floors RLM-MAC's `α` at `1e-3`.

## The results

Each iterate's hyper-parameters are saved in its checkpoint, `checkpoints/iter_<k>.npz`, under the key `hyperparameters`, of shape `(n_theta, n_members)`, in the components' order.

```python
import numpy as np

first = np.load("runs/my_run/checkpoints/iter_0.npz")["hyperparameters"]
last = np.load("runs/my_run/checkpoints/iter_12.npz")["hyperparameters"]
print(first.mean(), last.mean(), last.min(), last.max())
```

## Checking it

APEX's CHOP pieces are tested against the author's code (`tests/unit/algorithms/test_chop_reference.py`):

- the per-member taper loop of `RLM_MAC_CHOP.m`;
- `update_hyper.m` and `update_hyper_no_loc.m`.

The reference is a line-by-line transcription of the MATLAB, `tools/chop_reference/`.
