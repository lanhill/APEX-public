---
audience: public
---

# The geomodeler: priors from parameters or data

## Goal

The geomodeler gives a case's state variables a prior. It writes a new configuration beside the old one, containing the prior's sections (and, for some skills, a stored ensemble). It validates that configuration and checks a small drawn sample against what you asked for. A run never calls the geomodeler: the prior it writes is a recorded section or file, so runs stay reproducible.

After this lesson you can:

- get a prior from the model-free defaults, from well data, from a reference field, or from a plain-language request;
- tell what is estimated from your data and what is assumed;
- understand why a prior is refused, and what to change.

## Prerequisites

- A case configuration, here `case/config.toml`, with its state variables and grid.
- For `--data`: a CSV of hard data (format below).
- For `--reference`: an `.npz` of whole fields.
- For `facies_mps`: a training image in the case's folder.

## Steps

1. Pick the source of information and run the geomodeler:

```bash
uv run apex agent geomodel case/config.toml                        # the rules: Gaussian fields, values assumed
uv run apex agent geomodel case/config.toml --data case/wells.csv  # estimated from well data, conditioned to them
uv run apex agent geomodel case/config.toml --reference ref.npz    # statistics and variogram of a reference field
uv run apex agent geomodel case/config.toml --request "porosity correlated with ln PERMX, 0.7"
```

"The rules" means the default prior that is written without a model: Gaussian fields with assumed values.

With a model, the case's facts (its variables, grid and the request) go to it; see [what a provider receives, and how to keep it local](agents.md#what-a-provider-receives-and-how-to-keep-it-local).

2. Read the notes the command prints. They say which values were estimated and which were assumed.
3. Open the new configuration written beside your old one and check the prior's sections. Use that configuration for your runs.

The same function is available as the REST route `POST /v1/agents/geomodel` (`config`, `request`, `reference`, `data`), as the web UI's *Give it a prior* in the editor, and as the assistant's tool `draft_prior`.

## Skills

Each modeling scenario is a skill, a plugin of the registry `geomodeling_skills` (entry point `apex.geomodeling_skills`). The model chooses one skill and its parameters. Code checks them against the case and the skill's own rules, and refuses what does not fit, with the reason.

| Skill | Scenario | The prior it writes |
|---|---|---|
| `continuous` | Gaussian fields per property (ln PERMX, PORO, ...), each with its variogram, conditioned to hard data | `gaussian_field` per variable (`geostatspy_sgsim` for layers over 10,000 cells) |
| `correlated_continuous` | properties that vary together, with one variogram and a correlation matrix, cokriged to hard data | a stored ensemble beside the new configuration (`<config>_prior_ensemble.npz`), read by `file` |
| `facies_pluri` | facies by proportion from a truncated latent Gaussian field, conditioned to wells | `gstools_pgs` and `pgs_truncation` |
| `facies_mps` | facies with the shapes of a training image, by SNESIM, as signed distances | `mpslib_snesim` and `facies_distance` |

Values are in each variable's state space. A log variable (one the deck template exponentiates, such as `permx` in ln mD) takes its mean and spread in ln units, and a facies' permeability is given as ln(mD).

### Correlated properties

`correlated_continuous` draws one unit Gaussian field per variable, in order, and mixes them with the Cholesky factor `L` of the correlation matrix: variable `a` is `mean_a + std_a (L z)_a`. For two variables this is `mean + std (rho z_1 + sqrt(1 - rho^2) z_2)`, so `rho` is their correlation. The generators draw variables independently, so the prior is drawn once, into a stored ensemble named after the new configuration (`config_prior_ensemble.npz` beside `config_prior.toml`), and clipped to the variables' bounds (a run does not clip a stored prior). Ask for another `seed` in your request to get another draw.

### Facies from a training image

`facies_mps` needs a training image of the facies' shapes (`.npy` codes `[i, j(, k)]` or a GSLIB grid, codes `0..K-1`) in the case's folder, and one distance state variable per facies. Its check compares the drawn facies' proportions with the training image's. It runs MPSlib's `mps_snesim_tree` (the `facies` extra).

## Hard data

`--data` is a CSV with the columns `i,j,k,property,value`: 1-based cells and one value per cell, for example well logs averaged over the cells they cross.

```text
i,j,k,property,value
5,5,1,poro,0.21
5,5,1,permx,180
5,5,1,facies,1
```

- `property` names a state variable, in the deck's units. A log variable's values (e.g. `permx` in mD) are taken as ln, the state's space.
- `facies` gives integer codes `0..K-1`.

The geomodeler's code estimates the following from the data. The model uses these estimates unless you give other values in `--request` (for example `--request "porosity mean 0.18"`):

- per property, the data's mean and standard deviation, and a variogram (model, ranges, azimuth, nugget);
- for facies, the proportions (the codes' counts), the latent field's length scale, and the cells as conditioning data.

What the data cannot tell (too few values, all equal, a single column) is left out with a note, and the value is assumed instead.

### Conditioning to the data

The continuous skills also condition their priors to the data: every member takes the data's values at their cells, and near them the field follows the data. The generators `gaussian_field` and `geostatspy_sgsim` take the data as their `conditioning` option:

```toml
[prior.variables.poro]
name = "gaussian_field"

[prior.variables.poro.options]
grid = [40, 40, 3]
mean = 0.2
variance = 0.0009
major_range = 8.0
conditioning = { cells = [[5, 5, 1], [30, 12, 2]], values = [0.21, 0.17] }
```

Each unconditional realisation `z` is corrected by kriging its residuals at the data, with the generator's own variogram:

```text
z_c = z + C_xd (C_dd + R)^-1 (d - z_d - e),   e ~ N(0, R)
```

Here `z_d` is the realisation at the data cells, `d` the data values, `C_xd` the covariances between every grid cell and the data cells, `C_dd` the covariances among the data cells, and `R` the data's error variance (a diagonal matrix, zero for exact data). For a Gaussian field this is an exact draw from the field given the data: its mean is the simple-kriging estimate around the prior's mean, and its spread shrinks to zero at exact data. Layers are independent fields, so a datum conditions its own layer. For `geostatspy_sgsim` the correction is exact at the data and approximate away from them, because SGSIM's fields have the variogram only approximately.

#### Noisy data and `error_std`

With `error_std` (a number, or one per value) the data are treated as noisy: the members scatter around them by that much. The geomodeler sets it from the variogram's nugget. The nugget share is the nugget divided by the sill (the total variance), and

```text
error_std = sqrt(nugget share) x std
```

The generators have no nugget term of their own, so the nugget is represented as measurement error instead. The members then vary around the data as much as the data vary around the smooth field. The field keeps the whole variance as its sill.

Worked example: take `std = 0.03` (variance 0.0009) and a nugget share of 0.1.

- `error_std = sqrt(0.1) x 0.03`, about 0.0095.
- At a datum, the members' spread is the error's alone: about 0.0095.
- Far from every datum, the spread is the prior's: 0.03.
- In between, it grows from the first value to the second as the distance approaches the range.

#### Several properties at once

`correlated_continuous` conditions all its variables at once (simple cokriging through the correlation matrix): a well's ln PERMX moves the porosity around it as the correlation says, and the reverse.

#### Data that are refused

Two kinds of data cannot be honoured sensibly:

- data outside a variable's bounds, because the prior is clipped to the bounds;
- data improbable under the variogram, such as rough values a few cells apart under a smooth, long-range model. The exact conditional field would then swing far outside the prior.

Code measures the second harm itself. It computes the kriged mean `m + C_xd (C_dd + R)^-1 (d - m)`, where `m` is the prior mean. In this formula `d` is the data and the other symbols are as above. The kriged mean is evaluated at every grid cell, and it may depart from the prior mean by at most ten prior standard deviations at any cell. (A statistical test of the data would refuse real wells whenever the estimated variance is a little low, as estimates from correlated wells usually are.) For `correlated_continuous` all variables' data are judged together, through the correlation.

The skills refuse both kinds of data with the reason, and the rules then leave the conditioning out and say so in the notes. If you are refused, you can:

- give fewer data, or data that are smoother;
- ask for another variogram (for example a shorter range or a larger nugget) in `--request`;
- turn conditioning off, as described next.

#### Sharing statistics without conditioning

To share the data's statistics without conditioning to them, ask for it in the request. The skill's parameter is `condition = false`: in `continuous` it is set per field, and in `correlated_continuous` once for all variables.

```bash
uv run apex agent geomodel case/config.toml --data case/wells.csv --request "use the well statistics but do not condition to the wells"
```

Used directly, the generators log a warning when conditioning moves a field by more than ten prior standard deviations, and when the data's correlation matrix is nearly singular (close cells of a smooth model). In the second case a nugget of 1e-6 of each datum's variance is added, and the largest miss is logged.

A history-matching update (the step that adjusts the ensemble members to the observations during a run) leaves a cell unchanged when all members agree there. Exactly conditioned cells therefore keep the data's values through the run.

## A reference field

`--reference` is an `.npz` of whole fields in state space, one array per variable (for example the values of a deck you trust). Code estimates each field's mean and standard deviation from every active cell, and its variogram from about 3000 of them, drawn per layer (the estimator pairs cells within a layer only) with a seed of the variable's own, so the same reference always gives the same estimate. Active cells are the finite values; zeros count as inactive except for log variables, whose 0 is ln 1.

The prior shares these statistics and is not conditioned to the reference: its members are new fields that look like it, not copies of it.

With both `--reference` and `--data`, the statistics and variogram come from the reference, which shows them better than a few wells, and the prior is conditioned to the hard data.

## The variogram estimator

Variograms are estimated by a plugin of `variogram_estimators`, `wls` by default. You choose it in the `[agents.geomodeler]` section:

```toml
[agents.geomodeler]
estimator = "wls"
```

The estimator works in two steps:

1. **The experimental semivariogram** (Matheron's estimator), `gamma(h) = 1/(2 N(h)) * sum (z_i - z_j)^2` over the `N(h)` pairs in each lag class. A lag class is a band of distances between cell pairs; the largest lag is the longest such distance used. Only pairs in one layer count, as the priors are drawn per layer. An anisotropic estimate uses eight directions of 22.5 degrees each.
2. **The fit** of `spherical`, `exponential` and `gaussian` models with a nugget and geometric anisotropy (major and minor ranges, azimuth counter-clockwise from the i axis), by Cressie's weighted least squares: `sum N(h) (gamma(h) / model(h) - 1)^2` is minimised. The model with the least residual wins.

Ranges are practical ranges in cells (where the correlation falls to 5 %), as the generators read them. A fitted range is capped at twice the largest lag. When a fitted range hits that cap, the notes say so: the data do not show where the variogram levels off, so the range is only a lower bound. In that case, give a range yourself in `--request`, for example `--request "poro major range 12 cells"`.

A facies' length scale comes from the latent field, not the indicator. Each facies boundary's indicator is a thresholded Gaussian, whose variogram is mapped back to the latent field's through the bivariate normal distribution, and a gaussian model is fitted to that.

### How it was checked

- The experimental semivariogram equals GSTools' `vario_estimate` to round-off, in every direction.
- On fields drawn from known variograms, the median fitted range over 20 seeds is within 20 % of the truth and the sill within 15 %; the right model wins in most seeds.
- A thresholded field's latent range is recovered within 20 % (median of 20 seeds), at proportions of 50 % and 25 %.
- `correlated_continuous` draws exactly what the hand-written prior of the 3dBox example drew, given the same parameters and seed.
- Conditioned draws agree with the conditional Gaussian in closed form (mean and covariance, over 40,000 draws, with exact and noisy data), take exact data to round-off, and leave layers without data and the random stream as they were.
- Cokriged correlated fields reproduce the bivariate normal's conditional mean and spread at a datum.

## The drawn sample's check

Code draws a small sample of the new prior (`qc_members` in `[agents.geomodeler]`) and compares its means, spreads, correlations or facies proportions with what was asked, allowing for the sample's own noise. Bounds that clip a field show up here.

A prior that fails goes back to the model, up to `max_revisions` times (also in `[agents.geomodeler]`). If it still fails, the rules' prior (the model-free default described above) is written instead, and the notes printed by the command say so. Read the notes, then open the new configuration beside your old one to see which sections were written.

## Pitfalls

- Giving `permx` in ln units in the CSV: write it in mD; it is taken as ln for you.
- Using 0-based cells: `i,j,k` are 1-based.
- More than one value per cell: give one value per cell (average the logs over the cells they cross).
- Data outside the variable's bounds: the conditioning is refused and left out.
- Expecting `--reference` to copy the field: the members are new fields with the same statistics.
- Expecting a run to call the geomodeler: it does not; rerun the command to change the prior.

## What to read next

- [What a provider receives, and how to keep it local](agents.md#what-a-provider-receives-and-how-to-keep-it-local), before you use a model with your case.
