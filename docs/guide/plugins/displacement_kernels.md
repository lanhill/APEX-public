---
audience: public
items:
  - plugin:displacement_kernels:geertsma
  - plugin:displacement_kernels:van_opstal
---

# Displacement kernels

## Goal

A displacement kernel is the elastic response used to turn a source of strain into displacement. APEX provides two in the `displacement_kernels` registry:

- `geertsma`: Geertsma's (1973) nucleus of strain in an elastic half-space.
- `van_opstal`: a nucleus of strain over a rigid basement (van Opstal 1974; Tempone et al. 2010).

By the end of this page you will be able to choose between them and choose each in a configuration.

## Prerequisites

- A history-matching configuration with seafloor displacement data: a `[[geophysics.data]]` entry with `kind = "seafloor_displacement"`.
- A decision about whether your model is a plain half-space (`geertsma`) or has a rigid basement at a known depth (`van_opstal`).
- For `van_opstal`, a value for the basement depth. It is required.

## Steps

1. **Pick the kernel.** Use `geertsma` for an elastic half-space. Use `van_opstal` when a rigid basement lies below the source.
2. **Choose it in the data's options.** The kernel is the `kernel` option of the entry's `[geophysics.data.options]`: `kernel = { name = "geertsma", options = { ... } }`, or `kernel = "geertsma"` to keep a kernel's defaults. Without `kernel`, the entry uses `geertsma` with its defaults.
3. **Set the options.** Only the options below exist. Any other option is rejected, because neither plugin accepts additional properties.
4. **Leave out what you don't need.** Every option except `basement_depth` has a default.

### Options for `geertsma`

| Option | Type | Default | Notes |
|---|---|---|---|
| `poisson` | number | `0.25` | Poisson's ratio. |
| `truncation_factor` | number or unset | unset | Optional. Leave it out to keep it unset. |
| `free_surface` | `"stations"` or `"datum"` | `"stations"` | Which free-surface choice to use. |

### Options for `van_opstal`

| Option | Type | Default | Notes |
|---|---|---|---|
| `basement_depth` | number | none (required) | Depth of the rigid basement. You must set it. |
| `poisson` | number | `0.25` | Poisson's ratio. |
| `free_surface` | `"stations"` or `"datum"` | `"stations"` | Which free-surface choice to use. |

Note that `truncation_factor` belongs to `geertsma` only. `van_opstal` does not accept it.

## Worked example

First, a half-space kernel with a different Poisson's ratio and the free surface at the datum. The default `truncation_factor` is kept by leaving it out.

```toml
[[geophysics.data]]
kind = "seafloor_displacement"
datatype = "subsidence"
vintages = ["2025-01-01"]

[geophysics.data.options]
baseline = "2020-01-01"
stations = { spacing = 500.0, depth = 300.0 }
compressibility = 1e-10            # 1/Pa
kernel = { name = "geertsma", options = { poisson = 0.3, free_surface = "datum" } }
```

Next, a rigid-basement kernel. Only the required `basement_depth` is set, so `poisson` stays `0.25` and `free_surface` stays `"stations"`.

```toml
[[geophysics.data]]
kind = "seafloor_displacement"
datatype = "subsidence"
vintages = ["2025-01-01"]

[geophysics.data.options]
baseline = "2020-01-01"
stations = { spacing = 500.0, depth = 300.0 }
compressibility = 1e-10            # 1/Pa
kernel = { name = "van_opstal", options = { basement_depth = 3000.0 } }
```

A half-space kernel with every default needs only its name: `kernel = "geertsma"` (or leave `kernel` out).

Replace the example depth with the value for your own model, in the same length unit as the rest of your configuration.

## Pitfalls

- **Missing `basement_depth`.** `van_opstal` has no default for it, so `kernel = "van_opstal"` alone is refused.
- **Wrong option for the kernel.** `truncation_factor` is valid for `geertsma` only, and `basement_depth` for `van_opstal` only. Unknown options are rejected.
- **Misspelled `free_surface`.** Only `"stations"` and `"datum"` are accepted, in lowercase and quoted.
- **Assuming the two kernels give the same answer.** They model different media, so switching from `geertsma` to `van_opstal` changes the physics, not just the settings.
- **Forgetting the defaults.** If you do not set `poisson`, it is `0.25`. Set it explicitly if your medium differs.

## What to read next

Continue with the other plugin pages in the guide. Together they show how the geophysical data fit into a complete configuration.
