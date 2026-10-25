---
audience: public
---

# The geomodeler: priors from parameters or data

The geomodeler gives a case's state variables a prior. It writes a new configuration beside the old one, with the prior's sections (and, for some skills, a stored ensemble), validates it, and checks a small drawn sample against what was asked. A run never calls it: the prior it writes is a recorded section or file, so runs stay reproducible.

```bash
apex agent geomodel case/config.toml                        # the rules: Gaussian fields, values assumed
apex agent geomodel case/config.toml --data case/wells.csv  # estimated from well data, conditioned to them
apex agent geomodel case/config.toml --reference ref.npz    # statistics and variogram of a reference field
apex agent geomodel case/config.toml --request "porosity correlated with ln PERMX, 0.7"
```

The same is the REST route `POST /v1/agents/geomodel` (`config`, `request`, `reference`, `data`), the web UI's *Give it a prior* in the editor, and the assistant's tool `draft_prior`.

## Skills

Each modeling scenario is a skill, a plugin of the registry `geomodeling_skills` (entry point `apex.geomodeling_skills`). The model chooses one and its parameters; code checks them against the case and the skill's own rules, and refuses what does not fit with the reason.

| Skill | Scenario | The prior it writes |
|---|---|---|
| `continuous` | Gaussian fields per property (ln PERMX, PORO, ...), each with its variogram, conditioned to hard data | `gaussian_field` per variable (`geostatspy_sgsim` for layers over 10,000 cells) |
| `correlated_continuous` | properties that vary together, with one variogram and a correlation matrix, cokriged to hard data | a stored ensemble beside the new configuration (`<config>_prior_ensemble.npz`), read by `file` |
| `facies_pluri` | facies by proportion from a truncated latent Gaussian field, conditioned to wells | `gstools_pgs` and `pgs_truncation` |
| `facies_mps` | facies with the shapes of a training image, by SNESIM, as signed distances | `mpslib_snesim` and `facies_distance` |

Values are in each variable's state space: a log variable (one the deck template exponentiates, such as `permx` in ln mD) takes its mean and spread in ln units, and a facies' permeability as ln(mD).

### Correlated properties

`correlated_continuous` draws one unit Gaussian field per variable, in order, and mixes them with the Cholesky factor `L` of the correlation matrix: variable `a` is `mean_a + std_a (L z)_a`. For two variables this is `mean + std (rho z_1 + sqrt(1 - rho^2) z_2)`, so `rho` is their correlation. The generators draw variables independently, so the prior is drawn once, into a stored ensemble named after the new configuration (`config_prior_ensemble.npz` beside `config_prior.toml`), and clipped to the variables' bounds (a run does not clip a stored prior). Give another `seed` for another draw.

### Facies from a training image

`facies_mps` needs a training image of the facies' shapes (`.npy` codes `[i, j(, k)]` or a GSLIB grid, codes `0..K-1`) in the case's folder, and one distance state variable per facies. Its check compares the drawn facies' proportions with the training image's. It runs MPSlib's `mps_snesim_tree` (the `facies` extra).

## Hard data

`--data` is a CSV with the columns `i,j,k,property,value`: 1-based cells and one value a cell, e.g. well logs averaged over the cells they cross.

```text
i,j,k,property,value
5,5,1,poro,0.21
5,5,1,permx,180
5,5,1,facies,1
```

- `property` names a state variable, in the deck's units: a log variable's values (e.g. `permx` in mD) are taken as ln, the state's space.
- `facies` gives integer codes `0..K-1`.

From them code estimates, and the model takes unless you give other values:

- per property, the data's mean and standard deviation, and a variogram (model, ranges, azimuth, nugget);
- for facies, the proportions (the codes' counts), the latent field's length scale, and the cells as conditioning data.

What the data cannot tell (too few, all equal, a single column) is left out with a note, and the value is assumed instead. The notes say which values were estimated and which assumed.

### Conditioning to the data

The continuous skills also condition their priors to the data: every member takes the data's values at their cells, and near them the field follows the data. The generators `gaussian_field` and `geostatspy_sgsim` take the data as their `conditioning` option:

```toml
[prior.variables.poro]
name = "gaussian_field"
options = { grid = [40, 40, 3], mean = 0.2, variance = 0.0009, major_range = 8.0,
            conditioning = { cells = [[5, 5, 1], [30, 12, 2]], values = [0.21, 0.17] } }
```

Each unconditional realisation `z` is corrected by kriging its residuals at the data, with the generator's own variogram:

```text
z_c = z + C_xd (C_dd + R)^-1 (d - z_d - e),   e ~ N(0, R)
```

For a Gaussian field this is an exact draw from the field given the data: its mean is the simple-kriging estimate around the prior's mean, and its spread shrinks to zero at the data. Layers are independent fields, so a datum conditions its own layer. With `error_std` (a number, or one per value) the data are noisy: the members scatter around them by that much. The geomodeler sets it from the variogram's nugget, `sqrt(nugget share) × std`, since the generators have none: the members then vary around the data as much as the data vary around the smooth field. The field keeps the whole variance as its sill, so at a datum the members' spread is the error's alone, and away from the data the prior's. For `geostatspy_sgsim` the correction is approximate away from the data (SGSIM's fields have the variogram only approximately) and exact at them.

`correlated_continuous` conditions all its variables at once (simple cokriging through the correlation matrix): a well's ln PERMX moves the porosity around it as the correlation says, and the reverse.

Two kinds of data cannot be honoured sensibly:

- data outside a variable's bounds, because the prior is clipped to the bounds;
- data improbable under the variogram, such as rough values a few cells apart under a smooth, long-range model. The exact conditional field would then swing far outside the prior. Code measures this harm itself: the kriged mean, `m + C_xd (C_dd + R)^-1 (d − m)`, may depart from the prior mean by at most ten prior standard deviations anywhere. (A statistical test of the data would refuse real wells whenever the estimated variance is a little low, as estimates from correlated wells usually are.) For `correlated_continuous` all variables' data are judged together, through the correlation.

The skills refuse both with the reason, and the rules then leave the conditioning out and say so. To share the data's statistics without conditioning to them, set `condition = false`: per field in `continuous`, once for all variables in `correlated_continuous`. Used directly, the generators log a warning when conditioning moves a field by more than ten prior standard deviations, and when the data's correlation matrix is near singular (close cells of a smooth model; a nugget of 1e-6 of each datum's variance is then added, and the largest miss logged). A history-matching update leaves a cell where all members agree unchanged, so exactly conditioned cells keep the data's values through the run.

## A reference field

`--reference` is an `.npz` of whole fields in state space, one array per variable (for example the values of a deck you trust). Code estimates each field's mean and standard deviation from every active cell and its variogram from about 3000 of them, drawn per layer (the estimator pairs cells within a layer only) with a seed of the variable's own, so the same reference always gives the same estimate. Active cells are the finite values; zeros count as inactive except for log variables, whose 0 is ln 1. The prior shares these statistics and is not conditioned to the reference: its members are new fields that look like it, not copies of it.

With both `--reference` and `--data`, the statistics and variogram come from the reference, which shows them better than a few wells, and the prior is conditioned to the hard data.

## The variogram estimator

Variograms are estimated by a plugin of `variogram_estimators`, `wls` by default (`[agents.geomodeler] estimator`):

1. **The experimental semivariogram** (Matheron's estimator), `gamma(h) = 1/(2 N(h)) * sum (z_i - z_j)^2` over the `N(h)` pairs in each lag class. Only pairs in one layer count, as the priors are drawn per layer. An anisotropic estimate uses eight directions of 22.5 degrees each.
2. **The fit** of `spherical`, `exponential` and `gaussian` models with a nugget and geometric anisotropy (major and minor ranges, azimuth counter-clockwise from the i axis), by Cressie's weighted least squares: `sum N(h) (gamma(h) / model(h) - 1)^2` is minimised. The model with the least residual wins.

Ranges are practical ranges in cells (where the correlation falls to 5 %), as the generators read them. A range is bounded by twice the largest lag: one that reaches it is noted, since the data do not show where the variogram levels off.

A facies' length scale comes from the latent field, not the indicator: each facies boundary's indicator is a thresholded Gaussian, whose variogram is mapped back to the latent field's through the bivariate normal distribution, and a gaussian model is fitted to that.

### How it was checked

- The experimental semivariogram equals GSTools' `vario_estimate` to round-off, in every direction.
- On fields drawn from known variograms, the median fitted range over 20 seeds is within 20 % of the truth and the sill within 15 %; the right model wins in most seeds.
- A thresholded field's latent range is recovered within 20 % (median of 20 seeds), at proportions of 50 % and 25 %.
- `correlated_continuous` draws exactly what the hand-written prior of the 3dBox example drew, given the same parameters and seed.
- Conditioned draws agree with the conditional Gaussian in closed form (mean and covariance, over 40,000 draws, with exact and noisy data), take exact data to round-off, and leave layers without data and the random stream as they were.
- Cokriged correlated fields reproduce the bivariate normal's conditional mean and spread at a datum.

## The drawn sample's check

Code draws a small sample of the new prior (`[agents.geomodeler] qc_members`) and compares its means, spreads, correlations or facies proportions with what was asked, allowing for the sample's own noise. Bounds that clip a field show up here. A prior that fails goes back to the model, up to `max_revisions` times; then the rules decide.
