---
audience: public
items:
  - plugin:reflectivity:aki_richards
  - plugin:reflectivity:approx_zoeppritz_pp
  - plugin:reflectivity:fatti
  - plugin:reflectivity:shuey
  - plugin:reflectivity:zoeppritz
---

# Reflectivity

## Goal

By the end of this lesson you will know which reflectivity plugins APEX offers, how they differ, and how to set the one option they expose. All five plugins belong to the `reflectivity` registry and compute PP (P-wave to P-wave) reflectivity.

## Prerequisites

- A basic idea of what PP reflectivity is: the amplitude of a reflected P-wave at an interface between two layers, as a function of incidence angle.
- A history-matching configuration with AVO data: a `[[geophysics.data]]` entry with `kind = "avo"`. Its `options.reflectivity` chooses the plugin (default `zoeppritz`).

## The plugins

| Plugin | What it computes |
| --- | --- |
| `zoeppritz` | Exact PP reflectivity: Aki & Richards' (1980) explicit solution, eq. 5.40. |
| `approx_zoeppritz_pp` | PyLops' `approx_zoeppritz_pp`, the 3dBox example's choice. It is the same as `zoeppritz`. |
| `aki_richards` | Aki & Richards' (1980) linearised PP reflectivity, eq. 5.44. |
| `shuey` | Shuey's (1985) three-term approximation, `R₀ + G sin²θ + F (tan²θ - sin²θ)`. |
| `fatti` | Fatti et al.'s (1994) form of the linearised reflectivity, in impedance contrasts. |

## Steps

1. **Decide between exact and approximate.** `zoeppritz` is the exact solution. `aki_richards`, `shuey` and `fatti` are linearised or approximate forms. `approx_zoeppritz_pp` is listed as the same as `zoeppritz`.
2. **Pick the plugin that matches the formulation you need.** Use `shuey` for the three-term form, `fatti` if you want the impedance-contrast form, and `aki_richards` for Aki & Richards' linearised form.
3. **Check its options.** Only `aki_richards` has an option. The other four (`approx_zoeppritz_pp`, `fatti`, `shuey`, `zoeppritz`) take none, and unknown options are rejected for every plugin.
4. **Set `form` if you use `aki_richards`.** It is a string with two allowed values, `"textbook"` (the default) and `"linear"`.

## Worked example

This configures `aki_richards` to use the `linear` form instead of the default `textbook`:

```toml
[[geophysics.data]]
kind = "avo"
datatype = "avo"
vintages = ["2025-01-01"]

[geophysics.data.options]
# ... the entry's other AVO options (angles, wavelet, ...)
reflectivity = { name = "aki_richards", options = { form = "linear" } }
```

A plugin without options is given by its name alone: `reflectivity = "shuey"`.

If you leave `form` out, `aki_richards` uses `"textbook"`. The other four plugins need no options, so there is nothing to set for them.

## Pitfalls

- **Misspelled or extra options fail.** The options of every plugin are closed: nothing beyond the listed ones is accepted. In particular, `shuey`, `fatti`, `zoeppritz` and `approx_zoeppritz_pp` accept no options at all.
- **`form` accepts only two values.** Anything other than `"textbook"` or `"linear"` is invalid.
- **Do not expect `zoeppritz` and `approx_zoeppritz_pp` to differ.** They are documented as the same.
- **Do not expect linearised plugins to match the exact one everywhere.** `aki_richards`, `shuey` and `fatti` are approximations, so results can differ from `zoeppritz`.

## What to read next

Continue with the other plugin pages in the guide to see what reflectivity feeds into, and with the 3dBox example, which uses `approx_zoeppritz_pp`.
