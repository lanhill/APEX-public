---
audience: public
items:
  - cli:apex run
  - cli:apex stop
  - config:history_match:validation
  - plugin:methods:gies
  - plugin:methods:gn_enrml
  - plugin:methods:lm_enrml
  - plugin:stopping_rules:patience
  - plugin:stopping_rules:select
---

# Stopping rules

## Goal

A stopping rule decides when APEX stops an iterative history-matching run. By the end of this page you will know the two stopping rules, `patience` and `select`, and which one fits your run.

The procedure is published: X. Luo and P. Ranazzi (2024). Better stopping through cross validation in an iterative ensemble smoother: A perspective from supervised machine learning. *Geoenergy Science and Engineering* 243, 213298. [doi:10.1016/j.geoen.2024.213298](https://doi.org/10.1016/j.geoen.2024.213298)

## Prerequisites

- A working APEX setup and a run configuration file in TOML.
- An iterative method in that configuration: `gies`, `lm_enrml` or `gn_enrml`. ES-MDA's fixed schedule takes no cross validation, so it has no stopping rule.

## Key terms

- **Iterate:** the ensemble after one analysis step. An iterative method produces a path of iterates, one after another.
- **Accepted iterate:** an iterate whose step the method keeps. Some methods try a step (a *trial*) and reject it if it is not good enough. A rejected trial is not kept, and the method retries with more damping (for example `max_inner_iterations` in `lm_enrml` and `gn_enrml` is the number of rejected trials in a row after which the run stops). Only accepted iterates count for `patience`.
- **Cross-validation (CV) data:** data you hold out. They are forecast at every iteration but never assimilated.
- **CV score:** the mean-square normalized data mismatch of the CV data. Lower is better. The earlier text called this the "CV mismatch"; it is the same quantity.
- **CV minimum:** the lowest CV score seen so far.
- **Posterior:** the iterate where the CV score is lowest. This is the iterate the run keeps as its result.

## Setting up cross validation

A stopping rule works on CV data, which you set up in the `[validation]` section. It puts the scalar data into roles:

| Key | Role |
|---|---|
| `cv` | Forecast at every iteration but never assimilated. Their mismatch picks the posterior, and the stopping rule watches it. |
| `test` | Only scored, never assimilated nor used to stop: a check that cross validation works. |
| `history_matching` | The data assimilated. By default every scalar datum in neither `cv` nor `test`. |
| `stopping` | The stopping rule: `{ name = "patience", options = { patience = 2 } }` by default, or `"select"`. Needs `cv`. |

Without `cv` the method runs unchanged.

Each role is a *data selection*, a table with these keys:

| Key | Selects | Example |
|---|---|---|
| `wells` | every datum of these wells | `["PRO3"]` |
| `datatypes` | these exact datatypes | `["WOPR:PRO3"]` |
| `keywords` | these keywords, for every well | `["WWPR"]` |
| `times` | restricts the selection to these times, as the observations name them | `["2024-01-01"]` |

A datum `KEYWORD:WELL` at a report time is selected if it matches any of `wells`, `datatypes` or `keywords`, **and** it is at one of the `times` when `times` is given. In short: (`wells` OR `datatypes` OR `keywords`) AND `times`. Give at least one of `wells`, `datatypes` and `keywords`; `times` alone is not enough. A datatype without `:WELL` is its own well for `wells`.

Field data (for example seismic) are always history-matching data and cannot be selected for `cv` or `test`.

## The two rules

| Rule | What it does | Options |
|---|---|---|
| `patience` | Stops after `patience` accepted iterates without a new CV minimum. | `patience` (integer, default `2`, at least 1) |
| `select` | Never stops early. It keeps the minimum-CV iterate of the whole path, which is the paper's protocol. | none |

The `patience` option is the number of accepted iterates to wait after the last minimum, so that a local minimum is recognised before stopping.

## Steps

1. **Choose the CV data.** Put some wells or keywords in `cv` in `[validation]`.
2. **Decide whether you want an early stop.**
   - To save time by ending the run once the CV score stops improving, choose `patience`.
   - To examine the whole path and take its best iterate, or to follow the paper's protocol, choose `select`.
3. **If you choose `patience`, set its one option.** A larger value tolerates more accepted iterates without improvement before stopping. A smaller value stops sooner.
4. **If you choose `select`, set nothing.** It has no options.
5. **Run it** with `uv run apex run config.toml`.

## Worked example

This configuration holds the wells PRO3 and PRO4 out for cross validation and sets `patience` to `3`:

```toml
[validation]
cv = { wells = ["PRO3", "PRO4"] }
stopping = { name = "patience", options = { patience = 3 } }
```

`select` has no options, so its name is enough:

```toml
[validation]
cv = { wells = ["PRO3", "PRO4"] }
stopping = "select"
```

A selection can also combine a well with times. This one takes only the datum of PRO3 at one report time as CV data:

```toml
[validation]
cv = { wells = ["PRO3"], times = ["2024-01-01"] }
stopping = "select"
```

Now suppose the accepted iterates of the `patience = 3` run have these CV scores, in order:

| Iterate | CV score | New CV minimum? |
|---|---|---|
| 1 | 1.00 | yes |
| 2 | 0.80 | yes |
| 3 | 0.70 | yes |
| 4 | 0.72 | no (1 without improvement) |
| 5 | 0.71 | no (2) |
| 6 | 0.73 | no (3) |

Iterate 3 is the last new minimum. Iterates 4, 5 and 6 are three accepted iterates in a row without a new minimum, so `patience` with `patience = 3` stops after iterate 6. With the default of `2`, it would stop after iterate 5. The posterior is iterate 3.

With `select`, the same run continues to the end of the path (until the usual criteria of the method stop it). Afterwards the minimum-CV iterate of the whole path is kept. Here that is iterate 3, unless a later iterate beats it.

## Pitfalls

- **`patience` counts accepted iterates only.** Rejected trials do not count as iterates without improvement.
- **`patience` assumes a roughly V-shaped CV curve.** If the CV score may first rise and then fall, a small `patience` can stop early and keep the prior. In that case use `select` or a larger `patience`.
- **`select` never stops early.** It runs the whole path, so it can take longer than `patience`.
- **`select` takes no options.** Do not add any to it.
- **`stopping` needs `cv`.** Without CV data there is nothing to watch.
- **`times` does not select by itself.** It only narrows what `wells`, `datatypes` or `keywords` select.
- **Do not mix up the names.** The rule is `patience`, and so is its option. In `stopping = { name = "patience", options = { patience = 3 } }`, `patience = 3` is the number of accepted iterates to wait.

## What to read next

- Try your configuration for a few steps with `uv run apex run config.toml --max-steps 2`, then continue it later with `apex resume`.
- To end a running run cleanly at its next checkpoint, use `uv run apex stop RUN_DIR`, where `RUN_DIR` is the directory of the running run. `apex resume` continues it.
- Read the pages for the iterative methods `gies`, `lm_enrml` and `gn_enrml` in this guide: each has its own iteration budget and rejection limits that also end a run, next to the stopping rule.
