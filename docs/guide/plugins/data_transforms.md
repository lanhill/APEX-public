---
audience: public
items:
  - plugin:data_transforms:wavelet
---

# Data transforms

This page covers the `wavelet` plugin in the `data_transforms` registry. It replaces each observed field of the chosen `datatype` by its leading wavelet coefficients.

The method is the sparse representation of Luo et al. (2017): X. Luo, T. Bhakta, M. Jakobsen and G. Nævdal, An ensemble 4D-seismic history-matching framework with sparse representation based on wavelet multiresolution analysis. *SPE Journal* 22(3), 985–1010. [doi:10.2118/180025-PA](https://doi.org/10.2118/180025-PA). Each field is transformed by a discrete wavelet transform, and the coefficients above a noise-based threshold are kept as the data to match.

## Goal

By the end of this lesson you will be able to:

- configure the `wavelet` transform with its four required options;
- tune thresholding with `noise`, `threshold_rule`, `threshold_multiplier`, `thresholding` and `min_noise`;
- choose how the grid is read (`order`), which times are used (`times`), and how inactive cells are handled (`mask`, `inactive_value`).

## Prerequisites

- You know the name of the `datatype` whose observed fields you want to transform.
- You know the shape of the `grid`, given as a list of integers.
- You have picked a wavelet name for `wavelet` and a decomposition depth for `level`.

## Options at a glance

The plugin accepts only the options below. Any other option is rejected.

| Option | Type | Default | Required |
|---|---|---|---|
| `datatype` | string | none | yes |
| `grid` | list of integers | none | yes |
| `wavelet` | string | none | yes |
| `level` | integer | none | yes |
| `times` | list of strings, or unset | unset | no |
| `mask` | array-like, or unset | unset | no |
| `inactive_value` | number, or unset | unset | no |
| `order` | `"C"` or `"F"` | `"F"` | no |
| `noise` | `"white"` or `"coloured"` | `"white"` | no |
| `threshold_rule` | `"universal"` or `"bayesian"` | `"universal"` | no |
| `threshold_multiplier` | number | `1.0` | no |
| `thresholding` | `"hard"` or `"soft"` | `"hard"` | no |
| `keep_approximation` | boolean | `false` | no |
| `min_noise` | number | `1e-09` | no |

## Steps

1. **Name the data.** Set `datatype` to the data type whose observed fields should be replaced.
2. **Describe the grid.** Set `grid` to a list of integers, for example `[64, 64]`.
3. **Pick the wavelet and depth.** Set `wavelet` (a string) and `level` (an integer).
4. **Choose the memory layout.** `order` is `"F"` by default. Set it to `"C"` only if your fields are laid out that way.
5. **Describe the noise and thresholding.**
   - `noise` is `"white"` or `"coloured"`.
   - `threshold_rule` is `"universal"` or `"bayesian"`; `"bayesian"` needs `noise = "coloured"`.
   - `thresholding` is `"hard"` or `"soft"`.
   - `threshold_multiplier` scales the threshold (default `1.0`).
   - `min_noise` sets a lower bound for the noise level (default `1e-09`).
6. **Decide on the approximation.** Set `keep_approximation` to `true` if you want the approximation coefficients kept. It is `false` by default.
7. **Optionally restrict times and cells.** `times` takes a list of strings. `mask` and `inactive_value` deal with inactive cells.

## Worked example

This configuration transforms a hypothetical data type called `pressure` on a 64 by 64 grid. It uses a soft threshold and the Bayesian rule (which needs coloured noise), and keeps the approximation coefficients. The values of `datatype` and `wavelet` are examples only; use your own.

```toml
[[data_transforms]]
name = "wavelet"

[data_transforms.options]
datatype = "pressure"
grid = [64, 64]
wavelet = "db4"
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

The four required options (`datatype`, `grid`, `wavelet`, `level`) are all set. Everything else is shown only to make the choices visible. Remove the lines whose values equal the defaults and the behaviour stays the same.

## Pitfalls

- **Missing required options.** `datatype`, `grid`, `wavelet` and `level` have no defaults. Leaving any of them out is an error.
- **Unknown options.** The plugin rejects options not listed in the table above, so a misspelled name fails rather than being ignored.
- **Wrong `order`.** The default is `"F"`. If your data is stored in the other layout, set `"C"`, otherwise the grid is read in the wrong order.
- **Wrong value types.** `grid` must be a list of integers, `times` a list of strings, and `keep_approximation` a boolean (`true` or `false`, not quoted).
- **Restricted choices.** `order`, `noise`, `threshold_rule` and `thresholding` accept only the values in the table. Anything else is invalid.
- **Changing several threshold settings at once.** Change `threshold_rule`, `threshold_multiplier` and `thresholding` one at a time so you can see which one has an effect.

## What to read next

- The other pages in the plugins section of the guide, to see how data transforms fit with the other plugin registries.
