---
audience: public
items:
  - cli:apex plugins
  - plugin:data_transforms:wavelet
---

# Data transforms

This page covers the `wavelet` plugin in the `data_transforms` registry. It replaces each observed field of the chosen `datatype` by its leading wavelet coefficients.

The method is the sparse representation of Luo et al. (2017): X. Luo, T. Bhakta, M. Jakobsen and G. Nævdal, An ensemble 4D-seismic history-matching framework with sparse representation based on wavelet multiresolution analysis. *SPE Journal* 22(3), 985–1010. [doi:10.2118/180025-PA](https://doi.org/10.2118/180025-PA). Each field is transformed by a discrete wavelet transform, and the coefficients above a noise-based threshold are kept as the data to match.

## Goal

By the end of this lesson you will be able to:

- configure the `wavelet` transform with its four required options;
- tune the thresholding with `noise`, `threshold_rule`, `threshold_multiplier`, `thresholding`, `keep_approximation` and `min_noise`;
- choose how the grid is read (`order`), which times are used (`times`), and how inactive cells are handled (`mask`, `inactive_value`).

## Prerequisites

- You know the name of the `datatype` whose observed fields you want to transform.
- You know the shape of the `grid`: 1 to 3 positive sizes, e.g. `[nx, ny, nz]`.
- You have chosen a wavelet name for `wavelet`. It must be a discrete PyWavelets wavelet name, such as `"haar"` or `"db2"`.
- You have chosen the number of decomposition levels for `level`. It must be at least 1.

To list every plugin with its options, run:

```bash
uv run apex plugins
```

For full descriptions as JSON, add `--json`:

```bash
uv run apex plugins --json
```

## Options at a glance

The plugin accepts only the options below. Any other option is rejected.

| Option | Type | Default | Required |
|---|---|---|---|
| `datatype` | string | none | yes |
| `grid` | list of integers | none | yes |
| `wavelet` | string | none | yes |
| `level` | integer | none | yes |
| `times` | list of strings, or unset | unset (all times) | no |
| `mask` | array-like, or unset | unset (all cells active) | no |
| `inactive_value` | number, or unset | unset (each field's mean) | no |
| `order` | `"C"` or `"F"` | `"F"` | no |
| `noise` | `"white"` or `"coloured"` | `"white"` | no |
| `threshold_rule` | `"universal"` or `"bayesian"` | `"universal"` | no |
| `threshold_multiplier` | number | `1.0` | no |
| `thresholding` | `"hard"` or `"soft"` | `"hard"` | no |
| `keep_approximation` | boolean | `false` | no |
| `min_noise` | number | `1e-09` | no |

## Steps

1. **Name the data.** Set `datatype` to the data type whose observed fields should be replaced.
2. **Describe the grid.** Set `grid` to the shape of the grid the fields live on, for example `[64, 64]`.
3. **Pick the wavelet and depth.** Set `wavelet` to a PyWavelets wavelet name (for example `"haar"` or `"db2"`) and `level` to the number of decomposition levels (at least 1).
4. **Choose the cell order.** `order` says how the flattened fields (and the mask) list their cells. `"F"` (the default) is the Eclipse order, where the first index, i, runs fastest. `"C"` is the opposite: the last index runs fastest. On a 2D grid `[nx, ny]` with `"F"`, cells (0,0), (1,0), (2,0) and so on come first; with `"C"`, cells (0,0), (0,1), (0,2) and so on come first. Set `"C"` only if your data is stored that way.
5. **Choose the noise model.**
   - `"white"` (default): one noise level, estimated from the finest diagonal sub-band, is used for every coefficient.
   - `"coloured"`: one noise level per sub-band.
   - `min_noise` is a floor on the estimated noise standard deviations. It must be positive (default `1e-09`).
6. **Choose the threshold.**
   - `threshold_rule` is `"universal"` (default), which uses `sqrt(2 ln n)` times the noise level, or `"bayesian"`, which needs `noise = "coloured"`.
   - Non-zero coefficients at least as large as the threshold in magnitude are retained.
   - `threshold_multiplier` scales every threshold. It must be at least 0. The default 1 uses the rule's threshold as is, and 0 keeps every non-zero coefficient.
   - `thresholding` is `"hard"` (default), which keeps the retained coefficients as they are, or `"soft"`, which shrinks them towards zero by the threshold.
7. **Decide on the approximation.** Set `keep_approximation` to `true` to keep every approximation coefficient, unthresholded. It is `false` by default.
8. **Optionally restrict times and cells.**
   - `times` is a list of strings. Only the fields at these times are compressed; by default all times are.
   - `mask` marks the active cells. It is either grid-shaped or flat in the chosen `order`, and is shared by the fields. By default all cells are active. In a configuration file you give the mask as `mask_file`, which APEX loads for you.
   - `inactive_value` is the value used for inactive cells. By default each observed field's mean is used (as in PET).

## Worked example

This configuration transforms a hypothetical data type called `pressure` on a 64 by 64 grid. It uses the `"db2"` wavelet with 3 levels, coloured noise, the Bayesian rule, and soft thresholding, and it keeps the approximation coefficients. The `datatype` value is an example only; use your own.

```toml
[[data_transforms]]
name = "wavelet"

[data_transforms.options]
datatype = "pressure"
grid = [64, 64]
wavelet = "db2"
level = 3
order = "F"
noise = "coloured"
threshold_rule = "bayesian"
threshold_multiplier = 1.0
thresholding = "soft"
keep_approximation = true
min_noise = 1e-9
```

`[[data_transforms]]` is an array of tables (one entry per transform, in order), so its brackets are doubled; `[data_transforms.options]` holds the options of the entry just above it.

The four required options (`datatype`, `grid`, `wavelet`, `level`) are all set. `noise = "coloured"` is there because `threshold_rule = "bayesian"` needs it. The other lines are shown only to make the choices visible. Remove the ones whose values equal the defaults and the behaviour stays the same.

## Pitfalls

- **Missing required options.** `datatype`, `grid`, `wavelet` and `level` have no defaults. Leave none of them out.
- **Unknown options.** The plugin accepts only the options in the table above, so check the spelling of each name. Run `uv run apex plugins` to see the list.
- **Bayesian rule without coloured noise.** `threshold_rule = "bayesian"` needs `noise = "coloured"`. Set both together.
- **Wrong `order`.** The default is `"F"`. If your data lists its cells with the last index running fastest, set `"C"`. Otherwise the cells are read in the wrong order.
- **Wrong value types.** `grid` must be a list of integers, `times` a list of strings, and `keep_approximation` a boolean (`true` or `false`, not quoted).
- **Restricted choices.** `order`, `noise`, `threshold_rule` and `thresholding` accept only the values in the table.
- **Out-of-range numbers.** `level` must be at least 1, `threshold_multiplier` at least 0, and `min_noise` positive.
- **Changing several threshold settings at once.** Change `threshold_rule`, `threshold_multiplier` and `thresholding` one at a time and compare the transformed data after each change, so you can tell which setting caused a difference.

## What to read next

- Run `uv run apex plugins` to see this plugin's options next to those of the other plugins, and add `--json` for the full descriptions.
