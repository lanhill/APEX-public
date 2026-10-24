---
audience: public
items:
  - plugin:stopping_rules:patience
  - plugin:stopping_rules:select
---

# Stopping rules

## Goal

A stopping rule decides when APEX stops walking along the path of iterates. By the end of this page you will know the two stopping rules, `patience` and `select`, and which one fits your run.

Both rules judge iterates by their cross-validation (CV) score. The lowest CV score seen so far is the "CV minimum".

The procedure is published: X. Luo and P. Ranazzi (2024). Better stopping through cross validation in an iterative ensemble smoother: A perspective from supervised machine learning. *Geoenergy Science and Engineering* 243, 213298. [doi:10.1016/j.geoen.2024.213298](https://doi.org/10.1016/j.geoen.2024.213298)

## Prerequisites

- A working APEX setup and a configuration file in TOML.
- A basic idea of what a CV score is: lower is better, and the best iterate is the one at the CV minimum.

## Cross validation

A stopping rule works on cross-validation (CV) data, which you set up in the `[validation]` section of a history-matching configuration. Cross validation needs an iterative method (`gies`, `lm_enrml`, `gn_enrml`); ES-MDA's fixed schedule takes none. `[validation]` puts the scalar data into roles:

| Key | Role |
|---|---|
| `cv` | Forecast at every iteration but never assimilated. The iterate with the lowest CV mismatch is the posterior, and the stopping rule watches it. |
| `test` | Only scored, never assimilated nor used to stop: a check that cross validation works. |
| `history_matching` | The data assimilated. By default every datum in neither `cv` nor `test`. |
| `stopping` | The stopping rule: `{ name = "patience", options = { patience = 2 } }` by default, or `"select"`. Needs `cv`. |

Each role is a *data selection*, a table with these keys. A datum `KEYWORD:WELL` is selected if it matches any of `wells`, `datatypes` or `keywords`, and one of `times` when `times` is given. Give at least one of the first three.

| Key | Selects | Example |
|---|---|---|
| `wells` | every datum of these wells | `["PRO3"]` |
| `datatypes` | these exact datatypes | `["WOPR:PRO3"]` |
| `keywords` | these keywords, for every well | `["WWPR"]` |
| `times` | only these times, as the observations name them | `["2024-01-01"]` |

Field data (for example seismic) are always history-matching data and cannot be selected for `cv` or `test`. `apex data-split CONFIG` ranks the wells and proposes `[validation]` sections.

## The two rules

| Rule | What it does | Options |
|---|---|---|
| `patience` | Stops after `patience` accepted iterates without a new CV minimum. | `patience` (integer, default `2`) |
| `select` | Never stops early. It keeps the minimum-CV iterate of the whole path, which is the paper's protocol. | none |

## Steps

1. **Decide whether you want an early stop.**
   - To save time by ending the run once the CV score stops improving, choose `patience`.
   - To examine the whole path and take its best iterate, or to follow the paper's protocol, choose `select`.
2. **If you choose `patience`, set its one option.** `patience` is an integer and defaults to `2`. A larger value tolerates more accepted iterates without improvement before stopping. A smaller value stops sooner.
3. **If you choose `select`, set nothing.** It has no options.

## Worked example

This configuration holds the wells PRO3 and PRO4 out for cross validation and tunes `patience` to `3`:

```toml
[validation]
cv = { wells = ["PRO3", "PRO4"] }
stopping = { name = "patience", options = { patience = 3 } }
```

`select` has no options, so its name is enough: `stopping = "select"`.

Suppose the accepted iterates have these CV scores, in order:

| Iterate | CV score | New CV minimum? |
|---|---|---|
| 1 | 1.00 | yes |
| 2 | 0.80 | yes |
| 3 | 0.70 | yes |
| 4 | 0.72 | no (1 without improvement) |
| 5 | 0.71 | no (2) |
| 6 | 0.73 | no (3) |

Iterate 3 is the last new minimum. Iterates 4, 5 and 6 are three accepted iterates in a row without a new minimum, so `patience` with `patience = 3` stops after iterate 6. With the default of `2`, it would stop after iterate 5.

With `select`, the same run continues to the end of the path. Afterwards the minimum-CV iterate of the whole path is kept. Here that is iterate 3, unless a later iterate beats it.

## Pitfalls

- **`patience` counts accepted iterates only.** It counts accepted iterates without a new CV minimum, not every iterate.
- **`patience` can stop at a local minimum.** If the CV score would improve again after a long flat stretch, a small `patience` stops too early. Raise it if that worries you.
- **`select` never stops early.** It runs the whole path, so it can take longer than `patience`.
- **`select` takes no options.** Do not add any to it.
- **`stopping` needs `cv`.** Without CV data there is nothing to watch, and the configuration is refused.
- **Do not mix up the names.** The rule is `patience`, and so is its option. In `stopping = { name = "patience", options = { patience = 3 } }`, `patience = 3` is the number of iterates to wait.

## What to read next

Read the other plugin pages in this guide to see which settings sit alongside the stopping rule in a full configuration.
