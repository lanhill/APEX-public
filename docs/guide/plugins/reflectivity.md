---
audience: public
items:
  - cli:apex plugins
  - plugin:data_transforms:wavelet
  - plugin:gmac_constraints:linear
  - plugin:reflectivity:aki_richards
  - plugin:reflectivity:approx_zoeppritz_pp
  - plugin:reflectivity:fatti
  - plugin:reflectivity:shuey
  - plugin:reflectivity:zoeppritz
  - plugin:simulators:linear
---

# Reflectivity

## Goal

By the end of this lesson you will know which reflectivity plugins APEX offers, how they differ, which one to choose, and how to set the one option they expose. All five plugins belong to the `reflectivity` registry and compute PP (P-wave to P-wave) reflectivity.

## Prerequisites

- A basic idea of what PP reflectivity is: the amplitude of a reflected P-wave at an interface between two layers, as a function of incidence angle.
- A history-matching configuration with AVO data: a `[[geophysics.data]]` entry with `kind = "avo"`. Its `options.reflectivity` chooses the plugin (default `zoeppritz`). The entry's other AVO options are described on the guide's AVO data page.

## A few terms

- **Exact vs linearised.** `zoeppritz` solves the reflection problem exactly. A linearised form is a first-order expression in the contrasts (the relative jumps in density, Vp and Vs across the interface), so it is simpler but only agrees with the exact result to first order.
- **Impedance contrast.** Impedance is density times velocity; `fatti` writes the reflectivity in terms of the jump in impedance.
- **Shuey's terms.** In `R₀ + G sin²θ + F (tan²θ - sin²θ)`, `R₀` is the reflectivity at normal incidence, `G` is the gradient, and `F` is the term that matters at larger angles; `θ` is the incidence angle.

## The plugins

| Plugin | What it computes |
| --- | --- |
| `zoeppritz` | Exact PP reflectivity: Aki & Richards' (1980) explicit solution, eq. 5.40. |
| `approx_zoeppritz_pp` | PyLops' `approx_zoeppritz_pp`, the 3dBox example's choice. It is the same as `zoeppritz`. |
| `aki_richards` | Aki & Richards' (1980) linearised PP reflectivity, eq. 5.44. |
| `shuey` | Shuey's (1985) three-term approximation, `R₀ + G sin²θ + F (tan²θ - sin²θ)`. |
| `fatti` | Fatti et al.'s (1994) form of the linearised reflectivity, in impedance contrasts. |

Why are there two exact-looking plugins? `approx_zoeppritz_pp` is the name PyLops gives its function, and the 3dBox example uses it, so APEX offers it under that name. APEX documents it as the same as `zoeppritz`. If you are not following the 3dBox example, choose `zoeppritz`, which is also the default.

## Steps

1. **Decide between exact and linearised.** `zoeppritz` is exact and is the default, so keep it when in doubt. `aki_richards`, `shuey` and `fatti` are linearised or approximate forms, which only agree with the exact result to first order in the contrasts. Choose one of them when you want that specific formulation.
2. **Pick the plugin that matches the formulation you need.** Use `shuey` for the three-term form, `fatti` for the impedance-contrast form, and `aki_richards` for Aki & Richards' linearised form.
3. **Check its options.** Only `aki_richards` has an option. The other four (`approx_zoeppritz_pp`, `fatti`, `shuey`, `zoeppritz`) take none, and unknown options are rejected for every plugin.
4. **Set `form` if you use `aki_richards`.** It is a string with two allowed values: `"textbook"` (the default), which uses the mean of the incidence and transmission angles and the ray parameter, and `"linear"`, which uses PyLops' coefficients, linear in the contrasts. The two agree to first order in the contrasts.
5. **Check that APEX sees the plugin.** List the plugins and their options with:

```bash
uv run apex plugins
```

Add `--json` for the full descriptions as JSON:

```bash
uv run apex plugins --json
```

## Worked example

This configures `aki_richards` to use the `linear` form instead of the default `textbook`:

```toml
[[geophysics.data]]
kind = "avo"
datatype = "avo"
vintages = ["2025-01-01"]

[geophysics.data.options]
reflectivity = { name = "aki_richards", options = { form = "linear" } }
```

The entry also needs the other AVO options described on the AVO data page; this example shows only how the reflectivity plugin is chosen.

A plugin without options is given by its name alone: `reflectivity = "shuey"`.

If you leave `form` out, `aki_richards` uses `"textbook"`. The other four plugins need no options, so there is nothing to set for them.

## Pitfalls

- **Misspelled or extra options fail.** The options of every plugin are closed: nothing beyond the listed ones is accepted. In particular, `shuey`, `fatti`, `zoeppritz` and `approx_zoeppritz_pp` accept no options at all.
- **`form` accepts only two values.** Anything other than `"textbook"` or `"linear"` is invalid.
- **Do not expect `zoeppritz` and `approx_zoeppritz_pp` to differ.** They are documented as the same.
- **The two `aki_richards` forms are not identical.** They agree only to first order in the contrasts, so switching `form` can change your numbers.
- **Do not confuse registries.** Other plugins have similar or generic names but are not reflectivity plugins: `wavelet` belongs to `data_transforms`, and the `linear` plugins belong to `gmac_constraints` and `simulators`. Only the five above go in `options.reflectivity`.

## What to read next

- The guide's AVO data page, for the other options of the `[[geophysics.data]]` entry that uses reflectivity.
- The 3dBox example, which uses `approx_zoeppritz_pp`.
- The other plugin pages in the guide, to see what reflectivity feeds into.
